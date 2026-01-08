"""
LTX-2 Video Sampler for OneTrainer

Generates video samples for validation during training.
"""
from collections.abc import Callable

from modules.model.LTX2Model import LTX2Model
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch
from tqdm import tqdm


class LTX2Sampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: LTX2Model,
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
            noise_scheduler: NoiseScheduler,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        """Generate video using LTX-2 model."""

        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            # Calculate latent dimensions
            # LTX-2: spatial compression 32x, temporal compression 8x
            latent_height = height // self.model.vae_spatial_compression
            latent_width = width // self.model.vae_spatial_compression
            latent_frames = (num_frames - 1) // self.model.vae_temporal_compression + 1
            num_channels = 128  # LTX-2 latent channels

            # Encode text prompt
            self.model.text_encoder_to(self.train_device)

            text_embeddings, text_mask = self.model.encode_text(
                train_device=self.train_device,
                text=prompt,
            )

            # Handle CFG with negative prompt
            if cfg_scale > 1.0 and negative_prompt:
                neg_embeddings, neg_mask = self.model.encode_text(
                    train_device=self.train_device,
                    text=negative_prompt,
                )
                # Stack for batched inference
                text_embeddings = torch.cat([text_embeddings, neg_embeddings], dim=0)
                if text_mask is not None and neg_mask is not None:
                    text_mask = torch.cat([text_mask, neg_mask], dim=0)

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # Initialize random latents
            latent_video = torch.randn(
                size=(1, num_channels, latent_frames, latent_height, latent_width),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            # Setup timesteps for diffusion
            # LTX-2 uses timesteps in (0, 1) range with flow matching
            timesteps = torch.linspace(1.0, 0.0, diffusion_steps + 1, device=self.train_device)

            # Denoising loop
            self.model.transformer_to(self.train_device)

            for i, (t_curr, t_next) in enumerate(tqdm(
                zip(timesteps[:-1], timesteps[1:]),
                total=diffusion_steps,
                desc="sampling"
            )):
                # Prepare input
                latent_input = latent_video.to(dtype=self.model.transformer_train_dtype.torch_dtype())

                if cfg_scale > 1.0:
                    # Duplicate for CFG
                    latent_input = torch.cat([latent_input, latent_input], dim=0)

                # Current timestep
                t_batch = t_curr.expand(latent_input.shape[0])

                # Create modality for transformer
                video_modality = self.model.create_modality(
                    latents=latent_input,
                    timesteps=t_batch,
                    text_embeddings=text_embeddings,
                    text_mask=text_mask,
                )

                # Forward pass
                with self.model.transformer_autocast_context:
                    predicted_video, _ = self.model.transformer(
                        video=video_modality,
                        audio=None,
                        perturbations=None,
                    )

                # Unpatchify
                if self.model.patchifier is not None:
                    predicted_flow = self.model.patchifier.unpatchify(
                        predicted_video,
                        latent_input.shape[2:],
                    )
                else:
                    predicted_flow = predicted_video

                # Apply CFG
                if cfg_scale > 1.0:
                    pred_cond, pred_uncond = predicted_flow.chunk(2)
                    predicted_flow = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

                # Euler step: x_next = x_curr - (t_curr - t_next) * velocity
                dt = t_curr - t_next
                latent_video = latent_video - dt * predicted_flow

                on_update_progress(i + 1, diffusion_steps)

            self.model.transformer_to(self.temp_device)
            torch_gc()

            # Decode latents to video
            self.model.vae_to(self.train_device)

            latents = self.model.unscale_latents(latent_video)
            video = self.model.decode_video(latents)

            self.model.vae_to(self.temp_device)
            torch_gc()

            # Convert to frames list
            # video shape: [B, C, T, H, W] -> list of PIL images
            video = video[0]  # Remove batch dim
            video = video.permute(1, 0, 2, 3)  # [T, C, H, W]
            video = ((video + 1) / 2).clamp(0, 1)  # Denormalize

            frames = []
            for frame in video:
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
                frame_np = (frame_np * 255).astype('uint8')
                from PIL import Image
                frames.append(Image.fromarray(frame_np))

            return ModelSamplerOutput(
                file_type=FileType.VIDEO,
                data=frames,
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
        # Default video parameters for LTX-2
        num_frames = getattr(sample_config, 'num_frames', 41)  # ~5 seconds at 24fps

        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 32),  # LTX-2 uses 32x compression
            width=self.quantize_resolution(sample_config.width, 32),
            num_frames=num_frames,
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format or VideoFormat.MP4, audio_format,
        )

        on_sample(sampler_output)
