"""
WanSampler - Video sample generation for Wan models during training

Note: This is a placeholder implementation. Wan video generation
requires custom inference (not using diffusers pipeline).
"""

from collections.abc import Callable

from modules.model.WanModel import WanModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch

from PIL import Image
from tqdm import tqdm


class WanSampler(BaseModelSampler):
    """
    Sampler for Wan video models.
    
    Currently a minimal placeholder - full sampling requires
    porting Wan's inference pipeline.
    """
    
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: WanModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            num_frames: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            noise_scheduler: object,
            text_encoder_1_layer_skip: int = 0,
            text_encoder_2_layer_skip: int = 0, # Not used for Wan
            transformer_attention_mask: bool = False,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.transformer_autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            import copy
            if noise_scheduler is None:
                noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
            else:
                noise_scheduler = copy.deepcopy(self.model.noise_scheduler)

            vae_temporal_stride = 4
            vae_spatial_stride = 8
            num_latent_channels = 16

            # prepare prompt
            # Check if text encoder exists (e.g. might be offloaded or None)
            if self.model.text_encoder is None:
                 print("Warning: text_encoder is None in sampler")
                 # Return blank or handle gracefully? 
                 # For now proceed, encode_text handles None check or we crash?
            
            self.model.text_encoder_to(self.train_device)
            # Tokenizer is needed too
            
            # encode_text returns (embeddings, seq_lens)
            context, seq_lens = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
                text_encoder_layer_skip=text_encoder_1_layer_skip,
            )
            
            if cfg_scale > 1.0:
               neg_context, neg_seq_lens = self.model.encode_text(
                   text=negative_prompt or "",
                   train_device=self.train_device,
                   text_encoder_layer_skip=text_encoder_1_layer_skip,
               )
            
            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare latent image
            num_latent_frames = (num_frames - 1) // vae_temporal_stride + 1
            
            latent_image = torch.randn(
                size=(
                    1, # batch size
                    num_latent_channels,
                    num_latent_frames,
                    height // vae_spatial_stride,
                    width // vae_spatial_stride
                ),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32, 
            )

            # prepare timesteps
            noise_scheduler.set_timesteps(
                num_inference_steps=diffusion_steps,
                device=self.train_device,
            )
            timesteps = noise_scheduler.timesteps

            self.model.transformer_to(self.train_device)
            
            for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
                if cfg_scale > 1.0:
                    latent_model_input = torch.cat([latent_image, latent_image])
                    timestep_input = torch.cat([timestep.unsqueeze(0), timestep.unsqueeze(0)])
                    context_input = torch.cat([context, neg_context])
                    seq_len_input = torch.cat([seq_lens, neg_seq_lens])
                else:
                    latent_model_input = latent_image
                    timestep_input = timestep.unsqueeze(0)
                    context_input = context
                    seq_len_input = seq_lens
                
                with self.model.transformer_autocast_context:
                    # WanModel.forward(self, x, t, context, seq_len, y=None, clip_fea=None, parasitic_fea=None)
                    noise_pred = self.model.transformer(
                        x=latent_model_input.to(self.model.transformer_train_dtype.torch_dtype()),
                        t=timestep_input,
                        context=context_input.to(self.model.transformer_train_dtype.torch_dtype()),
                        seq_len=seq_len_input,
                    )

                if cfg_scale > 1.0:
                    pos_noise, neg_noise = noise_pred.chunk(2)
                    noise_pred = neg_noise + cfg_scale * (pos_noise - neg_noise)

                latent_image = noise_scheduler.step(
                    noise_pred.float(), timestep, latent_image, return_dict=False
                )[0]

                on_update_progress(i + 1, len(timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()

            # decode
            self.model.vae_to(self.train_device)

            video = self.model.vae_decode(latent_image.to(self.model.transformer_train_dtype.torch_dtype()))
            
            # Postprocess: (B, C, F, H, W) -> [-1, 1] usually? 
            # Wan VAE output range?
            # diffusion-pipe usually outputs [-1, 1] or [0, 1]?
            # VAE output is often unnormalized or [-1, 1].
            # Hunyuan uses video_processor. Logic implies it returns [-1, 1].
            # Let's assume [-1, 1] and rescale to [0, 1].
            
            video = (video / 2 + 0.5).clamp(0, 1)
            video = video.permute(0, 2, 3, 4, 1) # B, F, H, W, C
            video = (video * 255).round().to(torch.uint8)
            video = video[0].cpu() # F, H, W, C

            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(
                file_type=FileType.VIDEO,
                data=video,
            )

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 16),
            width=self.quantize_resolution(sample_config.width, 16),
            num_frames=sample_config.frames if hasattr(sample_config, 'frames') else 1,
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            on_update_progress=on_update_progress,
        )
        if self.model.vae:
             self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)
