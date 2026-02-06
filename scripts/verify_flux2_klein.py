#!/usr/bin/env python3
"""
Verify FLUX.2 Klein sampling/training by generating images.

This script tests:
1. Base model sampling works
2. LoRA loading and sampling works
3. Various resolutions work

Usage:
    # Basic sampling
    python scripts/verify_flux2_klein.py --prompt "a cat" --output cat.png

    # With LoRA
    python scripts/verify_flux2_klein.py --prompt "a portrait of sks person" \
        --lora path/to/lora.safetensors

    # Custom settings
    python scripts/verify_flux2_klein.py --prompt "a cat" \
        --model /path/to/FLUX.2-klein-4B \
        --steps 50 --seed 42 --width 768 --height 1024

SAFETY: This script NEVER downloads models. All models must be local.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify FLUX.2 Klein sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output",
        default="output.png",
        help="Output image path (default: output.png)",
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="Path to trained LoRA safetensors file",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA weight scale (default: 1.0)",
    )
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="Model path or HF ID (default: FLUX.2-klein-4B)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of diffusion steps (auto-detected if not set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Data type (default: bf16)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--use-diffusers-transformer",
        action="store_true",
        help="Use diffusers FluxTransformer2DModel instead of Flux2Transformer2DModel",
    )
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype_str]


def main():
    args = parse_args()

    # Validate output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get dtype
    dtype = get_dtype(args.dtype)
    device = torch.device(args.device)

    print(f"FLUX.2 Klein Verification Script")
    print(f"=" * 50)
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Steps: {args.steps or 'auto'}")
    print(f"Seed: {args.seed}")
    print(f"Dtype: {args.dtype}")
    if args.lora:
        print(f"LoRA: {args.lora} (scale={args.lora_scale})")
    print()

    # Import eritrainer modules
    try:
        from eritrainer.models.flux2_klein import Flux2KleinModelLoader, Flux2KleinSampler
    except ImportError as e:
        print(f"Error: Could not import eritrainer modules: {e}")
        print("Make sure you're running from the OneTrainer directory.")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = Flux2KleinModelLoader.load(
            model_path=args.model,
            dtype=dtype,
            device="cpu",  # Load to CPU first
            use_flux2_transformer=not args.use_diffusers_transformer,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print()
        print("Common issues:")
        print("  - Model not found locally (this script NEVER downloads)")
        print("  - Missing model components (VAE, text encoder)")
        print("  - Incorrect model path")
        sys.exit(1)

    # Create sampler
    print(f"Creating sampler...")
    sampler = Flux2KleinSampler(model, device=device, dtype=dtype)

    # Report model info
    variant = "4B" if model.is_4b() else "9B"
    mode = "distilled" if model.is_distilled() else "base"
    print(f"Model variant: FLUX.2 Klein {variant} ({mode})")

    # Generate image
    print()
    if args.lora:
        lora_path = Path(args.lora)
        if not lora_path.exists():
            print(f"Error: LoRA file not found: {args.lora}")
            sys.exit(1)

        print(f"Loading LoRA: {args.lora}")
        print(f"Generating with LoRA (scale={args.lora_scale})...")
        image = sampler.sample_with_lora(
            prompt=args.prompt,
            lora_path=str(lora_path),
            lora_scale=args.lora_scale,
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            seed=args.seed,
        )
    else:
        print(f"Generating: {args.prompt}")
        image = sampler.sample(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            seed=args.seed,
        )

    # Convert tensor to PIL and save
    from PIL import Image
    import numpy as np

    # Image is [1, 3, H, W] or [3, H, W] in [0, 1] range
    if image.dim() == 4:
        image = image[0]  # Remove batch dim

    # [3, H, W] -> [H, W, 3]
    img_np = image.float().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)

    # Save
    pil_image.save(output_path)
    print()
    print(f"Saved to: {output_path}")
    print(f"Image size: {pil_image.size}")
    print()
    print("Verification complete!")


if __name__ == "__main__":
    main()
