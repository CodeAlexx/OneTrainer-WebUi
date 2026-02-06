#!/usr/bin/env python3
"""
Test FLUX.2 edit mode LoRA inference on old photos.
Uses OneTrainer's custom diffusers classes for proper FLUX.2 support.
"""
import os
import sys
import torch
from PIL import Image
import glob
import numpy as np

sys.path.insert(0, "/home/alex/OneTrainer")

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from transformers import Qwen3ForCausalLM, AutoTokenizer
from safetensors.torch import load_file

# Paths
BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
LORA_PATH = "/home/alex/OneTrainer/workspace/run/backup/2026-01-17_16-16-14-backup-2400-25-0/lora/lora.safetensors"
INPUT_DIR = "/home/alex/OneTrainer/oldpics"
OUTPUT_DIR = "/home/alex/OneTrainer/oldpics/baseline"
PROMPT = "restore and colorize this old photograph, high quality, detailed, sharp, vibrant colors"

NUM_STEPS = 30
GUIDANCE_SCALE = 3.5
CONDITIONING_SCALE = 10.0

def load_and_preprocess_image(path, size=512):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2.0 - 1.0
    return img_tensor, img

def patchify_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
    return latents

def unpatchify_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
    return latents

def pack_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

def unpack_latents(latents, height, width):
    batch_size, seq_len, num_channels = latents.shape
    return latents.reshape(batch_size, height, width, num_channels).permute(0, 3, 1, 2)

def prepare_latent_image_ids(latents, device):
    """Prepare 4D position IDs for target latents (t=0)."""
    batch_size, _, height, width = latents.shape
    # For FLUX.2, position IDs are (t, h, w, l) where l=1
    t = torch.zeros(1, device=device, dtype=torch.long)  # t=0 for target
    h = torch.arange(height, device=device)
    w = torch.arange(width, device=device)
    l_ = torch.arange(1, device=device)
    latent_ids = torch.cartesian_prod(t, h, w, l_)  # [height*width, 4]
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

def prepare_conditioning_ids(latents, device, scale=10.0, ref_idx=0):
    """Prepare 4D position IDs for conditioning latents (t=scale+scale*ref_idx)."""
    batch_size, _, height, width = latents.shape
    t_value = int(scale + scale * ref_idx)
    t = torch.full((1,), t_value, device=device, dtype=torch.long)
    h = torch.arange(height, device=device)
    w = torch.arange(width, device=device)
    l_ = torch.arange(1, device=device)
    latent_ids = torch.cartesian_prod(t, h, w, l_)  # [height*width, 4]
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("Loading VAE...")
    vae = AutoencoderKLFlux2.from_pretrained(
        BASE_MODEL,
        subfolder="vae",
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        subfolder="tokenizer",
        local_files_only=True,
    )

    print("Loading text encoder (Qwen3)...")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        BASE_MODEL,
        subfolder="text_encoder",
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    text_encoder.eval()

    print("Loading transformer...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        BASE_MODEL,
        subfolder="transformer",
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)

    print(f"Transformer config: {transformer.config.num_attention_heads} heads")

    # Skip LoRA for baseline comparison
    print("\nRunning WITHOUT LoRA (baseline)")

    # Ensure entire model is in consistent dtype after LoRA merge
    transformer = transformer.to(dtype=dtype)
    transformer.eval()

    # Setup scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
    )

    # VAE scaling
    latent_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype=dtype)
    latent_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(device, dtype=dtype)

    def scale_latents(latents):
        return (latents - latent_bn_mean) / latent_bn_std

    def unscale_latents(latents):
        return latents * latent_bn_std + latent_bn_mean

    # Get images
    images = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.png"))
    images = [f for f in images if "restored" not in f and OUTPUT_DIR not in f]

    if not images:
        print("No images found!")
        return

    print(f"\nFound {len(images)} images to process")

    for img_path in images:
        print(f"\nProcessing: {os.path.basename(img_path)}")

        cond_tensor, cond_pil = load_and_preprocess_image(img_path, 512)
        cond_tensor = cond_tensor.to(device, dtype=torch.float32)

        with torch.no_grad():
            # Encode conditioning image
            cond_latent = vae.encode(cond_tensor).latent_dist.sample()
            cond_latent_patchified = patchify_latents(cond_latent)
            scaled_cond = scale_latents(cond_latent_patchified.to(dtype))
            packed_cond = pack_latents(scaled_cond)

            cond_ids = prepare_conditioning_ids(
                cond_latent_patchified, device,
                scale=CONDITIONING_SCALE, ref_idx=0
            )

            # Encode prompt
            chat = [{"role": "user", "content": PROMPT}]
            text = tokenizer.apply_chat_template(
                chat, tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            tokens = tokenizer(
                text,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            text_output = text_encoder(
                tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )
            # FLUX.2 Klein uses concatenated hidden states from layers 9, 18, 27
            # Total dim = 2560 * 3 = 7680
            QWEN3_HIDDEN_STATES_LAYERS = [9, 18, 27]
            text_embeds = torch.cat([text_output.hidden_states[k] for k in QWEN3_HIDDEN_STATES_LAYERS], dim=2)

            # Prepare text IDs - needs 4D coordinates (t, h, w, l)
            seq_len = text_embeds.shape[1]
            t = torch.arange(1, device=device)
            h = torch.arange(1, device=device)
            w = torch.arange(1, device=device)
            l_ = torch.arange(seq_len, device=device)
            txt_ids = torch.cartesian_prod(t, h, w, l_).unsqueeze(0)  # [1, seq_len, 4]

            # Initialize latents
            latent_height = cond_latent_patchified.shape[-2]
            latent_width = cond_latent_patchified.shape[-1]
            latent_channels = cond_latent_patchified.shape[1]

            latents = torch.randn(
                (1, latent_channels, latent_height, latent_width),
                device=device, dtype=dtype,
                generator=torch.Generator(device).manual_seed(42)
            )
            packed_latents = pack_latents(latents)
            target_seq_len = packed_latents.shape[1]
            image_ids = prepare_latent_image_ids(latents, device)

            scheduler.set_timesteps(NUM_STEPS)

            for i, t in enumerate(scheduler.timesteps):
                print(f"  Step {i+1}/{NUM_STEPS}", end="\r")

                combined_latents = torch.cat([packed_latents, packed_cond], dim=1)
                combined_ids = torch.cat([image_ids, cond_ids], dim=1)

                # Use bfloat16 for all inputs
                timestep = torch.tensor([t.item() / 1000], device=device, dtype=dtype)
                guidance = torch.tensor([GUIDANCE_SCALE], device=device, dtype=dtype)

                pred = transformer(
                    hidden_states=combined_latents,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states=text_embeds.to(dtype),
                    txt_ids=txt_ids,
                    img_ids=combined_ids,
                    return_dict=True,
                ).sample

                pred = pred[:, :target_seq_len, :]
                packed_latents = scheduler.step(pred, t, packed_latents, return_dict=False)[0]

            print()

            # Decode
            latents_unpacked = unpack_latents(packed_latents, latent_height, latent_width)
            latents_unscaled = unscale_latents(latents_unpacked)
            latents_unpatchified = unpatchify_latents(latents_unscaled)

            image = vae.decode(latents_unpatchified.to(torch.float32)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image[0].permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            result = Image.fromarray(image)

        out_name = os.path.basename(img_path).rsplit(".", 1)[0] + "_restored.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        result.save(out_path)
        print(f"  Saved: {out_path}")

        orig_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path).rsplit(".", 1)[0] + "_original.png")
        cond_pil.save(orig_path)

    print("\nDone!")

if __name__ == "__main__":
    main()
