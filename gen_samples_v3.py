#!/usr/bin/env python3
"""Generate 5 sample images using converted diffusers LoRA."""
import torch
import sys
sys.path.insert(0, "/home/alex/OneTrainer")

from diffusers import Flux2KleinPipeline
import os

LORA_PATH = "/home/alex/OneTrainer/lady_lora_diffusers.safetensors"
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

print("Loading pipeline...")
pipe = Flux2KleinPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.to("cuda")

print("Loading LoRA...")
pipe.load_lora_weights(LORA_PATH)

for i, prompt in enumerate(PROMPTS):
    print(f"\nGenerating {i+1}/5: {prompt[:50]}...")
    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=3.5,
        generator=torch.Generator("cuda").manual_seed(42 + i),
    ).images[0]

    out_path = os.path.join(OUTPUT_DIR, f"lora_sample_{i+1}.png")
    image.save(out_path)
    print(f"Saved: {out_path}")

print(f"\nDone! Images in {OUTPUT_DIR}")
