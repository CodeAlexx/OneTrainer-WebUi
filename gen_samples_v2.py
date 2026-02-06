#!/usr/bin/env python3
"""Generate 5 sample images using OneTrainer's LoRA loading."""
import torch
import sys
sys.path.insert(0, "/home/alex/OneTrainer")

from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from transformers import Qwen3ForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import os
import numpy as np
from PIL import Image

LORA_PATH = "/home/alex/OneTrainer/workspace/run/backup/2026-01-17_16-16-14-backup-2400-25-0/lora/lora.safetensors"
MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
OUTPUT_DIR = "/home/alex/OneTrainer/lady_samples"

PROMPTS = [
    "a beautiful woman in a red dress standing in a garden at sunset, elegant, photorealistic",
    "a young woman with curly hair reading a book in a cozy cafe, warm lighting, candid portrait",
    "a professional woman in a business suit giving a presentation, confident, modern office",
    "a woman hiking on a mountain trail at sunrise, athletic wear, scenic landscape background",
    "a woman artist painting in her studio, creative, colorful paint splatters, natural light",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda")
dtype = torch.bfloat16

print("Loading VAE...")
vae = AutoencoderKLFlux2.from_pretrained(MODEL, subfolder="vae", local_files_only=True, torch_dtype=torch.float32).to(device)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer", local_files_only=True)

print("Loading text encoder...")
text_encoder = Qwen3ForCausalLM.from_pretrained(MODEL, subfolder="text_encoder", local_files_only=True, torch_dtype=dtype).to(device)
text_encoder.eval()

print("Loading transformer...")
transformer = Flux2Transformer2DModel.from_pretrained(MODEL, subfolder="transformer", local_files_only=True, torch_dtype=dtype).to(device)

print("Loading LoRA weights...")
lora_state_dict = load_file(LORA_PATH)

# Convert OneTrainer format to PEFT format
peft_state_dict = {}
for key, value in lora_state_dict.items():
    # Remove lora_transformer. prefix and convert to PEFT naming
    if key.startswith("lora_transformer."):
        new_key = key.replace("lora_transformer.", "")
        # Convert to PEFT format: base_model.model.{layer}.lora_{A/B}.weight
        if ".lora_down." in new_key:
            new_key = new_key.replace(".lora_down.weight", ".lora_A.weight")
            new_key = f"base_model.model.{new_key}"
            peft_state_dict[new_key] = value
        elif ".lora_up." in new_key:
            new_key = new_key.replace(".lora_up.weight", ".lora_B.weight")
            new_key = f"base_model.model.{new_key}"
            peft_state_dict[new_key] = value

# Get target modules from LoRA keys
target_modules = set()
for key in lora_state_dict.keys():
    if ".lora_down." in key:
        # Extract module name
        module = key.replace("lora_transformer.", "").replace(".lora_down.weight", "")
        target_modules.add(module)

print(f"Found {len(target_modules)} LoRA target modules")

# For now, just merge LoRA weights directly into transformer
print("Merging LoRA into transformer...")
for key, value in lora_state_dict.items():
    if not key.startswith("lora_transformer."):
        continue
    if ".alpha" in key:
        continue

    # Parse the key
    parts = key.replace("lora_transformer.", "").split(".")

    if ".lora_down." in key:
        # Get corresponding up weight and alpha
        up_key = key.replace(".lora_down.", ".lora_up.")
        alpha_key = key.replace(".lora_down.weight", ".alpha")

        down = value.to(device, dtype=dtype)
        up = lora_state_dict[up_key].to(device, dtype=dtype)
        alpha = lora_state_dict.get(alpha_key, torch.tensor(down.shape[0]))
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.item()

        # Get the target module
        module_path = key.replace("lora_transformer.", "").replace(".lora_down.weight", "")

        # Navigate to module
        module = transformer
        for part in module_path.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        # Compute merged weight: W + (up @ down) * (alpha / rank)
        rank = down.shape[0]
        scale = alpha / rank
        delta = (up @ down) * scale

        with torch.no_grad():
            module.weight.add_(delta)

print("LoRA merged!")
transformer.eval()

# Setup scheduler
scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)

# VAE scaling
latent_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype=dtype)
latent_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(device, dtype=dtype)

def scale_latents(latents):
    return (latents - latent_bn_mean) / latent_bn_std

def unscale_latents(latents):
    return latents * latent_bn_std + latent_bn_mean

def patchify(latents):
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)

def unpatchify(latents):
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)

def pack(latents):
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)

def unpack(latents, h, w):
    b, s, c = latents.shape
    return latents.reshape(b, h, w, c).permute(0, 3, 1, 2)

def prepare_image_ids(latents):
    b, _, h, w = latents.shape
    t = torch.arange(1, device=device)
    hs = torch.arange(h, device=device)
    ws = torch.arange(w, device=device)
    l = torch.arange(1, device=device)
    ids = torch.cartesian_prod(t, hs, ws, l)
    return ids.unsqueeze(0).expand(b, -1, -1)

def prepare_text_ids(embeds):
    b, seq, _ = embeds.shape
    t = torch.arange(1, device=device)
    h = torch.arange(1, device=device)
    w = torch.arange(1, device=device)
    l = torch.arange(seq, device=device)
    ids = torch.cartesian_prod(t, h, w, l)
    return ids.unsqueeze(0).expand(b, -1, -1)

QWEN_LAYERS = [9, 18, 27]

for i, prompt in enumerate(PROMPTS):
    print(f"\nGenerating {i+1}/5: {prompt[:50]}...")

    with torch.no_grad():
        # Encode text
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer(text, padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device)

        text_out = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask, output_hidden_states=True)
        text_embeds = torch.cat([text_out.hidden_states[k] for k in QWEN_LAYERS], dim=2)
        txt_ids = prepare_text_ids(text_embeds)

        # Init latents (512x512 -> 64x64 latent -> 32x32 patchified)
        latents = torch.randn((1, 64, 32, 32), device=device, dtype=dtype, generator=torch.Generator(device).manual_seed(42 + i))
        packed = pack(latents)
        img_ids = prepare_image_ids(latents)

        scheduler.set_timesteps(30)

        for t in scheduler.timesteps:
            timestep = torch.tensor([t.item() / 1000], device=device, dtype=dtype)

            pred = transformer(
                hidden_states=packed,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states=text_embeds.to(dtype),
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=True,
            ).sample

            packed = scheduler.step(pred, t, packed, return_dict=False)[0]

        # Decode
        latents = unpack(packed, 32, 32)
        latents = unscale_latents(latents)
        latents = unpatchify(latents)

        image = vae.decode(latents.to(torch.float32)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image[0].permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        result = Image.fromarray(image)

    out_path = os.path.join(OUTPUT_DIR, f"lora_sample_{i+1}.png")
    result.save(out_path)
    print(f"Saved: {out_path}")

print(f"\nDone! Images in {OUTPUT_DIR}")
