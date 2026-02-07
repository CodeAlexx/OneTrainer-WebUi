# Serenity

A diffusion model training framework for fine-tuning image and video models.

**Status: pre-alpha 0.055** -- this is early, expect rough edges and breaking changes.

---

## What Is This

Serenity is a Python tool for training LoRAs, LyCORIS adapters, embeddings, and full fine-tunes across a bunch of diffusion model architectures. It grew out of wanting a single config-driven pipeline that could handle everything from SD 1.5 to video models without switching tools.

It's not polished software yet. The config surface is large (150+ fields), some model support is more battle-tested than others, and the documentation is catching up to the code. But it works, and if you're comfortable reading config files and don't mind the occasional rough edge, it might be useful to you.

## Models

These have actual model adapter code in the repo:

- **Stable Diffusion 1.5** -- including inpainting
- **SDXL 1.0** -- base and inpainting
- **Stable Diffusion 3 / 3.5** -- flow-matching
- **Flux 1** -- Dev, Schnell, Fill
- **Flux 2** -- including Klein 4B and 9B
- **Chroma**
- **Z-Image**
- **LTX2** -- video
- **HunyuanVideo** -- video
- **Qwen** -- image generation and editing

Not all models are equally tested. SD 1.5, SDXL, and Flux get the most use. Some of the newer ones (Chroma, HunyuanVideo) have less mileage on them.

## Training Methods

- **LoRA** -- configurable rank, alpha, layer filtering
- **LyCORIS** -- LoKR, LoHa, Tucker decomposition, DoRA, RS-LoRA
- **Embedding / Textual Inversion** -- SD 1.5 and SDXL only
- **Full Fine-Tune** -- needs 24 GB+ VRAM
- **VAE Fine-Tune**

## Other Features

- 45+ optimizers (standard, 8-bit, adaptive LR, schedule-free, research)
- Multiple LR schedulers, loss functions, noise strategies
- Aspect-ratio bucketing and disk-based latent/text-encoder caching
- Gradient checkpointing, layer offloading, quantization (NF4, INT8, FP8)
- DDP multi-GPU support
- EMA weight averaging
- Mixed precision with stochastic rounding
- Sample generation during training
- YAML and JSON configs with automatic enum coercion
- Checkpoint saving in SafeTensors, Diffusers, CKPT, ComfyUI formats
- Resume from backup

---

## Installation

```bash
git clone https://github.com/CodeAlexx/Serenity.git
cd Serenity
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Needs Python 3.10+, PyTorch 2.0+ with CUDA, and an NVIDIA GPU (8 GB minimum for LoRA, 24 GB+ for full fine-tune).

---

## Quick Start

```yaml
# config.yaml
model_type: sdxl
training_method: lora
transformer_path: "stabilityai/stable-diffusion-xl-base-1.0"
output_dir: "output/"
output_model_destination: "output/my_lora.safetensors"

learning_rate: 1.0e-4
epochs: 10
batch_size: 1
resolution: "1024"
train_dtype: BFLOAT_16

lora_rank: 16
lora_alpha: 1.0

optimizer:
  optimizer: ADAMW

concepts:
  - name: "my_subject"
    path: "training_data/my_subject"
    prompt:
      source: "txt"
```

```bash
python -m serenity.cli.commands train config.yaml
```

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for the full configuration reference.

---

## Project Structure

```
serenity/
├── core/           # Config, interfaces, trainer loop
├── models/         # Model adapters
├── training/       # Optimizers, schedulers, losses, EMA, precision, distributed
├── pipeline/       # Dataset loading, bucketing, caching, augmentations, masks
├── data/           # Dataloader
├── checkpoint/     # Save, load, resume
├── memory/         # Layer offloading, allocation
├── sampling/       # Sample generation during training
├── presets/        # VRAM-tier preset configs
├── cli/            # Command-line interface
├── adapters/       # LoRA, LyCORIS adapter layer
├── utils/          # Utilities
└── tests/          # Tests
```

---

## Model Support Matrix

| Model | LoRA | Full FT | Embedding | Inpainting |
|-------|------|---------|-----------|------------|
| SD 1.5 | Yes | Yes | Yes | Yes |
| SDXL 1.0 | Yes | Yes | Yes | Yes |
| SD 3 / 3.5 | Yes | Yes | -- | -- |
| Flux 1 | Yes | Yes | -- | Yes (Fill) |
| Flux 2 | Yes | Yes | -- | -- |
| Flux 2 Klein | Yes | Yes | -- | -- |
| Chroma | Yes | Yes | -- | -- |
| Z-Image | Yes | Yes | -- | -- |
| LTX2 | Yes | Yes | -- | -- |
| HunyuanVideo | Yes | Yes | -- | -- |
| Qwen | Yes | Yes | -- | -- |

---

## Known Limitations

- Pre-alpha software. APIs and config schema will change.
- No GUI -- CLI and config files only.
- AMD/ROCm support is untested and probably broken.
- Some model/training-method combinations may have bugs that haven't been found yet.
- Documentation may lag behind actual code behavior.

---

## Contributing

Contributions welcome. This is pre-alpha, so expect things to move around. Open an issue before starting large changes so we can discuss the approach.

---

## License

See [LICENSE](LICENSE) file.

---

## Acknowledgments

Built on PyTorch, Diffusers, bitsandbytes, LyCORIS, and many other open-source libraries.
