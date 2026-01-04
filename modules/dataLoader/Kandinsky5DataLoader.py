
import copy
import os
import torch

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.Kandinsky5Model import Kandinsky5Model
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS, TrainDataLoader
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.MapData import MapData
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize
from mgds.pipelineModules.VariationSorting import VariationSorting
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.DecodeTokens import DecodeTokens

class Kandinsky5DataLoader(BaseDataLoader, DataLoaderText2ImageMixin):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: Kandinsky5Model,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        super().__init__(train_device, temp_device)
        self.model = model

        if is_validation:
            config = copy.copy(config)
            config.batch_size = 1
            config.multi_gpu = False

        self.__ds = self.create_dataset(config, model, train_progress, is_validation)
        self.__dl = TrainDataLoader(self.__ds, config.batch_size)

    def get_data_set(self) -> MGDS:
        return self.__ds

    def get_data_loader(self) -> TrainDataLoader:
        return self.__dl

    def _preparation_modules(self, config: TrainConfig, model: Kandinsky5Model):
        rescale_image = RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)
        
        # VAE
        encode_image = EncodeVAE(
            in_name='image', 
            out_name='latent_image_distribution', 
            vae=model.vae, 
            autocast_contexts=[model.transformer_autocast_context], 
            dtype=model.transformer_train_dtype.torch_dtype()
        )
        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')

        # Text Tokenization (Dual)
        # Qwen
        tokenize_qwen = Tokenize(
            in_name='prompt',
            tokens_out_name='tokens_qwen',
            mask_out_name='tokens_mask_qwen',
            tokenizer=model.tokenizer_qwen,
            max_token_length=model.text_len_qwen,
        )
        
        # CLIP
        tokenize_clip = Tokenize(
            in_name='prompt',
            tokens_out_name='tokens_clip',
            mask_out_name='tokens_mask_clip',
            tokenizer=model.tokenizer_clip,
            max_token_length=model.text_len_clip,
        )

        # Text Encoding moved to output modules (avoid caching issues)
        encode_qwen = None
        encode_clip = None
        
        # RoPE Generation - create from latent image
        def create_rope_fn(latent):
            visual_rope_pos, text_rope_pos = model.create_rope_inputs(latent, None)
            return {'visual_rope_pos': visual_rope_pos, 'text_rope_pos': text_rope_pos}

        create_rope = MapData(
            in_name='latent_image',
            out_name='rope_positions',
            map_fn=create_rope_fn
        )

        modules = [rescale_image, encode_image, image_sample]

        if model.tokenizer_qwen:
            modules.append(tokenize_qwen)
        if model.tokenizer_clip:
            modules.append(tokenize_clip)

        # Note: Text encoding is NOT done during caching to avoid thread-safety issues
        # Text embeddings are computed on-the-fly during training in the predict method

        modules.append(create_rope)

        return modules

    def _cache_modules(self, config: TrainConfig, model: Kandinsky5Model):
        # Cache latents and tokens (NOT embeddings - computed on-the-fly)
        image_split_names = ['latent_image', 'rope_positions']
        text_split_names = ['tokens_qwen', 'tokens_clip']
        # Text embeddings are NOT cached - they're computed during training

        image_aggregate_names = ['crop_resolution', 'image_path']
        
        image_cache_dir = os.path.join(config.cache_dir, "image")
        text_cache_dir = os.path.join(config.cache_dir, "text")

        def before_cache_image():
             model.to(self.temp_device)
             model.vae_to(self.train_device)
             model.eval()
        
        def before_cache_text():
             model.to(self.temp_device)
             if not config.train_text_encoder_or_embedding():
                 model.text_encoder_to(self.train_device)
             model.eval()

        image_disk_cache = DiskCache(
            cache_dir=image_cache_dir, split_names=image_split_names, aggregate_names=image_aggregate_names,
            variations_in_name='concept.image_variations', balancing_in_name='concept.balancing', 
            balancing_strategy_in_name='concept.balancing_strategy', 
            variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.image'], 
            group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_image
        )

        text_disk_cache = DiskCache(
            cache_dir=text_cache_dir, split_names=text_split_names, aggregate_names=[],
            variations_in_name='concept.text_variations', balancing_in_name='concept.balancing', 
            balancing_strategy_in_name='concept.balancing_strategy', 
            variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'], 
            group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_text
        )

        modules = []
        if config.latent_caching:
            modules.append(image_disk_cache)
            if not config.train_text_encoder_or_embedding():
                modules.append(text_disk_cache)
        
        # Sort logic ... TBD
        
        return modules

    def _output_modules(self, config: TrainConfig, model: Kandinsky5Model):
        # Output latents, tokens, and prompt (for on-the-fly text encoding)
        output_names = [
             'latent_image', 'rope_positions',
             'tokens_qwen', 'tokens_clip',
             'prompt',  # For on-the-fly text encoding in predict()
        ]
             
        def before_cache_image():
             pass # Already handled or similar logic

        return self._output_modules_from_out_names(
             output_names=output_names,
             config=config,
             before_cache_image_fun=before_cache_image,
             use_conditioning_image=False, # T2V focus for now
             vae=model.vae,
             autocast_context=[model.transformer_autocast_context],
             train_dtype=model.transformer_train_dtype,
        )

    def _debug_modules(self, config, model):
        return [] # Empty for now

    def create_dataset(self, config, model, train_progress, is_validation):
        # Reuse mixin logic for data loading pipeline
        enumerate_input = self._enumerate_input_modules(config, allow_videos=True)
        load_input = self._load_input_modules(config, model.transformer_train_dtype, allow_video=True)
        mask_augmentation = self._mask_augmentation_modules(config)
        aspect_bucketing_in = self._aspect_bucketing_in(config, 64, frame_dim_enabled=True)  # 64 = VAE spatial factor
        crop = self._crop_modules(config)
        augmentation = self._augmentation_modules(config)
        preparation = self._preparation_modules(config, model)
        cache = self._cache_modules(config, model)
        output = self._output_modules(config, model)

        return self._create_mgds(
            config,
            [enumerate_input, load_input, mask_augmentation, aspect_bucketing_in, crop, augmentation, preparation, cache, output],
            train_progress,
            is_validation
        )
