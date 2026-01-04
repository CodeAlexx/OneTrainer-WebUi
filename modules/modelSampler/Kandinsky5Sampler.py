"""
Kandinsky 5 Sampler

Generates images/videos using Kandinsky 5 model during validation/sampling.
"""

import torch
from typing import Callable
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from modules.model.Kandinsky5Model import Kandinsky5Model
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.VideoFormat import VideoFormat


class Kandinsky5Sampler(BaseModelSampler):
    def __init__(self, train_device: torch.device, temp_device: torch.device, model: Kandinsky5Model):
        super().__init__(train_device, temp_device)
        self.model = model

    def __sample_base(
            self,
            prompt: str,
            height: int,
            width: int,
            num_frames: int,
            seed: int,
            diffusion_steps: int,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        """Generate images/video using the Kandinsky 5 model."""
        model = self.model
        device = self.train_device
        dtype = model.transformer_train_dtype.torch_dtype() if model.transformer_train_dtype else torch.bfloat16

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # Encode text - move encoders to GPU first
        model.text_encoder_to(device)
        text_embed, pooled_text_embed = model.encode_text([prompt], device)
        # Take first element (unbatch) and move to device with correct dtype
        text_embed = text_embed[0].to(dtype=dtype, device=device)
        pooled_text_embed = pooled_text_embed[0].to(dtype=dtype, device=device)
        # Move text encoders back to save VRAM
        model.text_encoder_to(self.temp_device)
        torch.cuda.empty_cache()

        # Latent dimensions
        latent_h = height // 8
        latent_w = width // 8
        latent_t = num_frames  # For video, use temporal frames

        # Initial noise
        latents = torch.randn(
            (latent_t, latent_h, latent_w, 16),
            device=device, dtype=dtype, generator=generator
        )

        # Visual Condition Padding
        has_visual_cond = model.dit_config.get('visual_cond', False)
        if has_visual_cond:
            visual_cond = torch.zeros_like(latents)
            visual_cond_mask = torch.zeros((latent_t, latent_h, latent_w, 1), device=device, dtype=dtype)
            latents = torch.cat([latents, visual_cond, visual_cond_mask], dim=-1)

        # Scheduler
        scheduler = model.noise_scheduler
        scheduler.set_timesteps(diffusion_steps, device=device)

        # RoPE positions - use PATCHED dimensions
        patch_size = model.dit_config.get('patch_size', (1, 2, 2))
        patched_t = max(1, latent_t // patch_size[0])
        patched_h = latent_h // patch_size[1]
        patched_w = latent_w // patch_size[2]
        visual_rope_pos = (
            torch.arange(patched_t, device=device),
            torch.arange(patched_h, device=device),
            torch.arange(patched_w, device=device),
        )
        text_rope_pos = torch.arange(text_embed.shape[0], device=device)

        # Move transformer to device
        model.transformer.to(device)
        if hasattr(model, 'transformer_lora') and model.transformer_lora is not None:
            model.transformer_lora.to(device)

        # Enable block swapping for memory-efficient inference
        use_block_swap = True  # Can be made configurable later
        if use_block_swap and hasattr(model.transformer, 'block_swap_enabled'):
            model.transformer.block_swap_enabled = True
            model.transformer.blocks_in_memory = 6  # Keep 6 blocks in VRAM at a time
            model.transformer.pin_first_n_blocks = 2
            model.transformer.pin_last_n_blocks = 2
            if model.transformer.swap_stream is None and torch.cuda.is_available():
                model.transformer.swap_stream = torch.cuda.Stream()
            model.transformer.setup_block_swapping(device, torch.device('cpu'))

        # Sampling loop
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Sampling")):
            on_update_progress(i, len(scheduler.timesteps))

            # Timestep to 0-1 range
            t_input = t / 1000.0 if t > 1.0 else t
            t_tensor = t_input.to(device=device, dtype=torch.float32).view(1)

            with torch.no_grad():
                model_out = model.transformer(
                    x=latents,
                    text_embed=text_embed,
                    pooled_text_embed=pooled_text_embed,
                    visual_rope_pos=visual_rope_pos,
                    text_rope_pos=text_rope_pos,
                    time=t_tensor,
                )

                if isinstance(model_out, tuple):
                    noise_pred = model_out[0]
                elif hasattr(model_out, 'sample'):
                    noise_pred = model_out.sample
                else:
                    noise_pred = model_out

            # Scheduler step
            if has_visual_cond:
                latents_main = latents[..., :16]
                latents_cond = latents[..., 16:]
                latents_main = scheduler.step(noise_pred, t, latents_main).prev_sample
                latents = torch.cat([latents_main, latents_cond], dim=-1)
            else:
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        on_update_progress(len(scheduler.timesteps), len(scheduler.timesteps))

        # Clear loaded blocks to free VRAM for VAE
        if use_block_swap and hasattr(model.transformer, 'clear_loaded_blocks'):
            model.transformer.clear_loaded_blocks()
            model.transformer.block_swap_enabled = False  # Reset for next call

        # Extract main latents
        if has_visual_cond:
            latents = latents[..., :16]

        # Decode with VAE
        # Latents: [T, H, W, C] -> [1, C, T, H, W] for VAE
        latents_for_vae = latents.permute(3, 0, 1, 2).unsqueeze(0)

        with torch.no_grad():
            model.vae.to(device)
            decoded = model.vae.decode(latents_for_vae).sample
            model.vae.to(self.temp_device)
            torch.cuda.empty_cache()

        # Output: [1, C, T, H, W]
        if num_frames == 1:
            # Single image: [1, C, H, W] -> PIL Image
            image = decoded[:, :, 0, :, :]
            image = (image / 2 + 0.5).clamp(0, 1)
            pil_image = to_pil_image(image[0].cpu().float())  # Remove batch dim, convert to float32, then PIL

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=pil_image,
            )
        else:
            # Video: [F, H, W, C] uint8 for write_video
            video = decoded.cpu().permute(0, 2, 3, 4, 1).float()  # [1, T, H, W, C]
            video = (video / 2 + 0.5).clamp(0, 1)
            video = (video * 255).round().to(dtype=torch.uint8)
            video = video[0]  # [T, H, W, C]

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
        # Get number of frames from config
        num_frames = getattr(sample_config, 'frames', 1)
        if isinstance(num_frames, str):
            num_frames = int(num_frames)

        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            height=self.quantize_resolution(sample_config.height, 64),
            width=self.quantize_resolution(sample_config.width, 64),
            num_frames=num_frames,
            seed=sample_config.seed,
            diffusion_steps=sample_config.diffusion_steps,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)
