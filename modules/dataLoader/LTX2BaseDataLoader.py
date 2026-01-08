"""
LTX-2 Video Data Loader for OneTrainer

Handles video data loading and preprocessing for LTX-2 training.
"""
import copy
import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.LTX2Model import LTX2Model
from modules.util.config.TrainConfig import TrainConfig
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS, TrainDataLoader
from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
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

import torch


class LTX2BaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: LTX2Model,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        super().__init__(
            train_device,
            temp_device,
        )

        if is_validation:
            config = copy.copy(config)
            config.batch_size = 1
            config.multi_gpu = False

        self.__ds = self.create_dataset(
            config=config,
            model=model,
            train_progress=train_progress,
            is_validation=is_validation,
        )
        self.__dl = TrainDataLoader(self.__ds, config.batch_size)

    def get_data_set(self) -> MGDS:
        return self.__ds

    def get_data_loader(self) -> TrainDataLoader:
        return self.__dl

    def create_dataset(
            self,
            config: TrainConfig,
            model: LTX2Model,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        """Create the MGDS dataset with all pipeline modules."""
        enumerate_input = self._enumerate_input_modules(config, allow_videos=True)
        load_input = self._load_input_modules(config, model.train_dtype, allow_video=True)
        mask_augmentation = self._mask_augmentation_modules(config)
        aspect_bucketing_in = self._aspect_bucketing_in(config, 32, True)  # LTX-2 uses 32x spatial compression
        crop_modules = self._crop_modules(config)
        augmentation_modules = self._augmentation_modules(config)
        inpainting_modules = self._inpainting_modules(config)
        preparation_modules = self._preparation_modules(config, model)
        cache_modules = self._cache_modules(config, model)
        output_modules = self._output_modules(config, model)

        debug_modules = self._debug_modules(config, model)

        return self._create_mgds(
            config,
            [
                enumerate_input,
                load_input,
                mask_augmentation,
                aspect_bucketing_in,
                crop_modules,
                augmentation_modules,
                inpainting_modules,
                preparation_modules,
                cache_modules,
                output_modules,

                debug_modules if config.debug_mode else None,
            ],
            train_progress,
            is_validation
        )

    def _preparation_modules(self, config: TrainConfig, model: LTX2Model):
        """Prepare video data for LTX-2 training."""
        modules = []

        # Rescale video frames from [0,1] to [-1,1]
        rescale_image = RescaleImageChannels(
            image_in_name='image',
            image_out_name='image',
            in_range_min=0,
            in_range_max=1,
            out_range_min=-1,
            out_range_max=1
        )
        modules.append(rescale_image)

        # Encode video through VAE
        if model.vae_encoder is not None:
            encode_image = EncodeVAE(
                in_name='image',
                out_name='latent_image_distribution',
                vae=model.vae_encoder,
                autocast_contexts=[model.autocast_context],
                dtype=model.train_dtype.torch_dtype()
            )
            modules.append(encode_image)

            image_sample = SampleVAEDistribution(
                in_name='latent_image_distribution',
                out_name='latent_video',
                mode='mean'
            )
            modules.append(image_sample)

        # Handle masked training
        if config.masked_training or config.model_type.has_mask_input():
            # LTX-2 uses 32x spatial compression
            downscale_mask = ScaleImage(
                in_name='mask',
                out_name='latent_mask',
                factor=1.0 / model.vae_spatial_compression
            )
            modules.append(downscale_mask)

        # Text encoding - LTX-2 uses Gemma text encoder
        if model.text_encoder is not None and not config.train_text_encoder_or_embedding():
            # Text encoder encoding will happen during training
            # since LTX-2 uses a custom Gemma encoder
            pass

        return modules

    def _cache_modules(self, config: TrainConfig, model: LTX2Model):
        """Setup caching for latents and text embeddings."""
        image_split_names = ['latent_video', 'original_resolution', 'crop_offset']

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        if config.model_type.has_conditioning_image_input():
            image_split_names.append('latent_conditioning_image')

        image_aggregate_names = ['crop_resolution', 'image_path']

        text_split_names = []

        sort_names = image_aggregate_names + image_split_names + [
            'prompt', 'text_encoder_hidden_state', 'text_encoder_mask',
            'concept'
        ]

        if not config.train_text_encoder_or_embedding():
            text_split_names.append('text_encoder_hidden_state')
            text_split_names.append('text_encoder_mask')

        image_cache_dir = os.path.join(config.cache_dir, "image")
        text_cache_dir = os.path.join(config.cache_dir, "text")

        def before_cache_image_fun():
            model.to(self.temp_device)
            model.vae_to(self.train_device)
            model.eval()
            torch_gc()

        def before_cache_text_fun():
            model.to(self.temp_device)

            if not config.train_text_encoder_or_embedding():
                model.text_encoder_to(self.train_device)

            model.eval()
            torch_gc()

        image_disk_cache = DiskCache(
            cache_dir=image_cache_dir,
            split_names=image_split_names,
            aggregate_names=image_aggregate_names,
            variations_in_name='concept.image_variations',
            balancing_in_name='concept.balancing',
            balancing_strategy_in_name='concept.balancing_strategy',
            variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.image'],
            group_enabled_in_name='concept.enabled',
            before_cache_fun=before_cache_image_fun
        )

        text_disk_cache = DiskCache(
            cache_dir=text_cache_dir,
            split_names=text_split_names,
            aggregate_names=[],
            variations_in_name='concept.text_variations',
            balancing_in_name='concept.balancing',
            balancing_strategy_in_name='concept.balancing_strategy',
            variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'],
            group_enabled_in_name='concept.enabled',
            before_cache_fun=before_cache_text_fun
        )

        modules = []

        if config.latent_caching:
            modules.append(image_disk_cache)

        if config.latent_caching:
            sort_names = [x for x in sort_names if x not in image_aggregate_names]
            sort_names = [x for x in sort_names if x not in image_split_names]

            if not config.train_text_encoder_or_embedding():
                modules.append(text_disk_cache)
                sort_names = [x for x in sort_names if x not in text_split_names]

        if len(sort_names) > 0:
            variation_sorting = VariationSorting(
                names=sort_names,
                balancing_in_name='concept.balancing',
                balancing_strategy_in_name='concept.balancing_strategy',
                variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'],
                group_enabled_in_name='concept.enabled'
            )
            modules.append(variation_sorting)

        return modules

    def _output_modules(self, config: TrainConfig, model: LTX2Model):
        """Configure output modules for the data loader."""
        output_names = [
            'image_path', 'latent_video',
            'prompt',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        if config.model_type.has_conditioning_image_input():
            output_names.append('latent_conditioning_image')

        if not config.train_text_encoder_or_embedding():
            output_names.append('text_encoder_hidden_state')
            output_names.append('text_encoder_mask')

        def before_cache_image_fun():
            model.to(self.temp_device)
            model.vae_to(self.train_device)
            model.eval()
            torch_gc()

        return self._output_modules_from_out_names(
            output_names=output_names,
            config=config,
            before_cache_image_fun=before_cache_image_fun,
            use_conditioning_image=True,
            vae=model.vae_decoder,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: LTX2Model):
        """Debug modules for visualizing training data."""
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        modules = []

        if model.vae_decoder is not None:
            decode_image = DecodeVAE(
                in_name='latent_video',
                out_name='decoded_image',
                vae=model.vae_decoder,
                autocast_contexts=[model.autocast_context],
                dtype=model.train_dtype.torch_dtype()
            )
            modules.append(decode_image)

        if config.masked_training or config.model_type.has_mask_input():
            upscale_mask = ScaleImage(
                in_name='latent_mask',
                out_name='decoded_mask',
                factor=model.vae_spatial_compression
            )
            modules.append(upscale_mask)

            save_mask = SaveImage(
                image_in_name='decoded_mask',
                original_path_in_name='image_path',
                path=debug_dir,
                in_range_min=0,
                in_range_max=1,
                before_save_fun=before_save_fun
            )
            modules.append(save_mask)

        save_prompt = SaveText(
            text_in_name='prompt',
            original_path_in_name='image_path',
            path=debug_dir,
            before_save_fun=before_save_fun
        )
        modules.append(save_prompt)

        return modules
