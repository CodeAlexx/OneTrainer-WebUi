# Qwen Image Edit Training Guide

Qwen Image Edit is an image-to-image editing model that takes a source image and a text prompt to generate an edited version.

## Dataset Format

Qwen Image Edit requires **paired images**: a source (control) image and a target image.

### OneTrainer Format

Place files in your concept folder using the `-condlabel` suffix pattern:

```
my_dataset/
  image1.jpg           # Target image (what you want to learn)
  image1-condlabel.png # Source/control image (input to the model)
  image1.txt           # Edit prompt (e.g., "remove the background")

  image2.jpg
  image2-condlabel.png
  image2.txt
  ...
```

**Important**: Conditioning images must be `.png` format.

### SimpleTuner Format Conversion

If you have a SimpleTuner-format dataset (separate `control/` and `images/` folders), use the conversion script:

```bash
python scripts/convert_qwen_edit_dataset.py \
    --input /path/to/simpletuner_dataset \
    --output /path/to/onetrainer_dataset
```

## Training Configuration

### Recommended Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Resolution | 512 | Start low, increase if VRAM allows |
| Gradient Checkpointing | ON | Required for 24GB GPUs |
| FP8 Weights | ON | Reduces VRAM by ~40% |
| LoRA Rank | 16-64 | Higher = more capacity |
| Learning Rate | 1e-4 | Standard for LoRA |
| Batch Size | 1 | Increase with gradient accumulation |

### Memory Requirements

| Resolution | Min VRAM (FP8) | Min VRAM (BF16) |
|------------|----------------|-----------------|
| 256px | ~12GB | ~20GB |
| 512px | ~18GB | ~28GB |
| 768px | ~24GB | ~40GB+ |

### Example Preset

Use the `qwen_image_edit_lora.json` preset as a starting point.

## Technical Notes

### Gradient Checkpointing

Gradient checkpointing is highly recommended for training on consumer GPUs. Note that:

- Musubi block swap is automatically disabled when gradient checkpointing is enabled
- This is due to incompatibility with Diffusers' `_gradient_checkpointing_func`
- CPU offloaded checkpointing may cause issues; use regular `ON` mode

### Control Image Handling

The training pipeline:

1. Encodes both target and control images to VAE latents
2. Adds the temporal dimension (required by Qwen's video-style VAE)
3. Packs latents using Qwen's patchification
4. Concatenates control + noisy target latents
5. Passes `img_shapes` to describe the token structure

### Flow Matching

Qwen Image Edit uses flow matching, not standard diffusion:

- Noise is added as: `noisy = sigma * noise + (1 - sigma) * latent`
- Target for training is velocity: `target = noise - latent`
- Timesteps are sampled using logit-normal distribution

## Troubleshooting

### "Expected all tensors to be on the same device"

This error typically occurs when:
- Musubi block swap conflicts with gradient checkpointing
- Solution: Ensure gradient checkpointing is `ON` (Musubi auto-disables)

### Out of Memory (OOM)

Try these in order:
1. Reduce resolution to 256px
2. Enable FP8 weights for transformer and text encoder
3. Reduce batch size to 1
4. Enable gradient checkpointing

### Control images not found

Ensure:
- Conditioning images use `-condlabel` suffix
- Conditioning images are `.png` format
- Files are in the same directory as target images

## References

- [Qwen-Image-Edit on HuggingFace](https://huggingface.co/Qwen/Qwen-Image-Edit)
- [SimpleTuner Qwen Edit Training](https://github.com/bghira/SimpleTuner)
