#!/usr/bin/env python3
"""Test Z-Image sampling using diffusers pipeline as reference."""

import torch
from pathlib import Path
from PIL import Image


def main():
    print("=" * 60)
    print("Z-Image Sampling Test (using diffusers pipeline)")
    print("=" * 60)

    # Use HF cached model
    model_path = "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/78771b7e11b922c868dd766476bda1f4fc6bfc96"

    output_dir = Path("/home/alex/OneTrainer/output/zimage_sample_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    from diffusers import ZImagePipeline

    print("\n[1/2] Loading Z-Image pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.to(device)

    print("\n[2/2] Generating samples...")
    prompts = [
        "a portrait photo of a woman",
        "a woman standing in a city street, natural lighting",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n  Generating: {prompt[:50]}...")

        image = pipe(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=8,
            guidance_scale=1.0,
            generator=torch.Generator(device).manual_seed(42),
        ).images[0]

        save_path = output_dir / f"sample_{i:02d}.png"
        image.save(save_path)
        print(f"  Saved: {save_path}")

    print("\n" + "=" * 60)
    print("Sampling test complete!")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
