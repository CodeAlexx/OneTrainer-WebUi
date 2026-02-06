"""Standalone test for SD3 sample generation."""
import torch
from PIL import Image
from tqdm import tqdm

def main():
    print("Loading SD3.5-medium...")

    # Load model components directly
    from diffusers import SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection

    model_path = "stabilityai/stable-diffusion-3.5-medium"
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Load tokenizers
    tokenizer_1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", local_files_only=True)

    # Load text encoders
    print("Loading CLIP encoders...")
    text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_path, subfolder="text_encoder_2", torch_dtype=dtype, local_files_only=True
    ).to(device)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype, local_files_only=True
    ).to(device)

    # Load transformer
    print("Loading transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype, local_files_only=True
    ).to(device)

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler", local_files_only=True
    )

    # Settings
    height, width = 512, 512  # Small for fast test
    num_steps = 20
    cfg_scale = 5.0
    seed = 42

    prompt = "a photograph of a cat sitting on a windowsill, natural lighting"
    negative_prompt = "blurry, low quality"

    # Encode prompts
    print("Encoding prompts...")

    def encode_clip(tokenizer, encoder, text, device):
        tokens = tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        output = encoder(**tokens, output_hidden_states=True)
        hidden = output.hidden_states[-2]  # Penultimate layer
        pooled = output.text_embeds
        return hidden, pooled

    # Positive
    h1_pos, p1_pos = encode_clip(tokenizer_1, text_encoder_1, prompt, device)
    h2_pos, p2_pos = encode_clip(tokenizer_2, text_encoder_2, prompt, device)

    # Negative
    h1_neg, p1_neg = encode_clip(tokenizer_1, text_encoder_1, negative_prompt, device)
    h2_neg, p2_neg = encode_clip(tokenizer_2, text_encoder_2, negative_prompt, device)

    # Combine CLIP embeddings [CLIP-L 768 || CLIP-G 1280] = 2048
    prompt_embeds_pos = torch.cat([h1_pos, h2_pos], dim=-1)  # [1, 77, 2048]
    pooled_embeds_pos = torch.cat([p1_pos, p2_pos], dim=-1)  # [1, 2048]

    prompt_embeds_neg = torch.cat([h1_neg, h2_neg], dim=-1)
    pooled_embeds_neg = torch.cat([p1_neg, p2_neg], dim=-1)

    # Pad to transformer's expected dim if needed
    joint_dim = transformer.config.joint_attention_dim
    if prompt_embeds_pos.shape[-1] < joint_dim:
        pad_size = joint_dim - prompt_embeds_pos.shape[-1]
        prompt_embeds_pos = torch.nn.functional.pad(prompt_embeds_pos, (0, pad_size))
        prompt_embeds_neg = torch.nn.functional.pad(prompt_embeds_neg, (0, pad_size))

    # Offload text encoders
    text_encoder_1.to("cpu")
    text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    # Create noise
    print("Creating noise...")
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_h, latent_w = height // 8, width // 8
    latent_channels = transformer.config.in_channels

    latents = torch.randn(
        1, latent_channels, latent_h, latent_w,
        generator=generator, device=device, dtype=dtype
    )

    # Denoise with Euler
    print(f"Denoising ({num_steps} steps)...")

    # Linear timestep schedule from 1 to 0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    num_train_timesteps = scheduler.config.num_train_timesteps

    for i in tqdm(range(num_steps)):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t

        # Integer timestep for transformer
        int_t = (t * num_train_timesteps).long()

        # CFG: concat negative and positive
        latent_input = torch.cat([latents, latents], dim=0)
        int_t_batch = int_t.expand(2)
        prompt_input = torch.cat([prompt_embeds_neg, prompt_embeds_pos], dim=0)
        pooled_input = torch.cat([pooled_embeds_neg, pooled_embeds_pos], dim=0)

        # Predict flow
        with torch.no_grad():
            flow_pred = transformer(
                hidden_states=latent_input,
                timestep=int_t_batch,
                encoder_hidden_states=prompt_input,
                pooled_projections=pooled_input,
                return_dict=True,
            ).sample

        # Apply CFG
        flow_neg, flow_pos = flow_pred.chunk(2)
        flow_pred = flow_neg + cfg_scale * (flow_pos - flow_neg)

        # Euler step
        latents = latents + dt * flow_pred

    # Decode
    print("Decoding...")
    transformer.to("cpu")
    torch.cuda.empty_cache()

    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0.0)

    latents_scaled = latents / vae_scale + vae_shift
    with torch.no_grad():
        decoded = vae.decode(latents_scaled).sample

    # Save
    print("Saving...")
    image = decoded[0].cpu().float()
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype("uint8")
    Image.fromarray(image).save("test_sd3_sample.png")
    print("Saved to test_sd3_sample.png")

if __name__ == "__main__":
    main()
