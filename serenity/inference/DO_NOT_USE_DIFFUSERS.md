# CRITICAL: NO DIFFUSERS IN INFERENCE

The inference engine (`serenity/inference/`) is a STANDALONE system.

## Rules

1. **DO NOT** import anything from `diffusers` in this directory
2. **DO NOT** use diffusers pipelines for inference/generation
3. **DO NOT** add diffusers as a dependency for the inference tab
4. The inference engine loads single-file checkpoints (.safetensors, .gguf) directly
5. The inference engine handles its own sampling, text encoding, VAE decode
6. This is intentional — diffusers was removed after 2 days of cleanup work

## Why

- A previous Claude session added diffusers to inference against explicit instructions
- It took 2 days to remove it
- The training sampler (`serenity/sampling/`) uses diffusers — that is SEPARATE and OK
- The inference engine (`serenity/inference/`) must remain diffusers-free

## What to use instead

- Direct model loading from safetensors/GGUF files
- Native attention, sampling, and VAE implementations in `serenity/inference/`
- ComfyUI/Forge-style single-file checkpoint workflow
