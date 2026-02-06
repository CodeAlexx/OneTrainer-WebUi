#!/usr/bin/env python3
"""Generate 5 sample images using FLUX.2 Klein 9B with LoKR adapter."""
import torch
import os
import sys

sys.path.insert(0, "/home/alex/OneTrainer")

from diffusers import Flux2KleinPipeline
from safetensors.torch import load_file
import lycoris

# Paths
BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
LOKR_PATH = "/home/alex/flux2_9b_lokr/backup/2026-01-18_20-03-20-backup-9215-78-11/lora/lora.safetensors"
OUTPUT_DIR = "/home/alex/flux2_lokr_samples_1024"

PROMPTS = [
    "a beautiful lady with flowing dark hair, elegant portrait, soft lighting",
    "a lady in a red dress standing in a garden at sunset, photorealistic",
    "portrait of a young lady with blue eyes, professional headshot",
    "a lady reading a book in a cozy library, warm lighting",
    "elegant lady at a cafe, Parisian style, candid photo",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading FLUX.2 Klein 9B pipeline...")
pipe = Flux2KleinPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

print("Loading LoKR adapter...")
lokr_state_dict = load_file(LOKR_PATH)

# Create LyCORIS wrapper and apply to transformer
wrapper, _ = lycoris.create_lycoris_from_weights(
    1.0,  # multiplier
    LOKR_PATH,
    pipe.transformer,
)
wrapper.merge_to()

print("Moving to GPU with CPU offload...")
pipe.enable_model_cpu_offload()

for i, prompt in enumerate(PROMPTS):
    print(f"\nGenerating {i+1}/5: {prompt[:50]}...")
    image = pipe(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=3.5,
        height=1024,
        width=1024,
        generator=torch.Generator("cuda").manual_seed(42 + i),
    ).images[0]

    out_path = os.path.join(OUTPUT_DIR, f"lokr_sample_{i+1}.png")
    image.save(out_path)
    print(f"Saved: {out_path}")

print(f"\nDone! Images in {OUTPUT_DIR}")
