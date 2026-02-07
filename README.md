# Serenity

**A modular diffusion model training framework for fine-tuning and adapting generative image and video models.**

<!-- Badges: CI status, Python version, License, PyPI version -->
<!-- ![CI](https://img.shields.io/github/actions/workflow/status/CodeAlexx/Serenity/ci.yml) -->
<!-- ![Python](https://img.shields.io/badge/python-3.10%2B-blue) -->
<!-- ![License](https://img.shields.io/github/license/CodeAlexx/Serenity) -->

---

## Overview

Serenity is a Python framework for fine-tuning and training diffusion models across a wide range of architectures. It provides a unified training pipeline supporting Stable Diffusion 1.5, SDXL, SD3/3.5, Flux 1 and 2 (including Klein compact models), Chroma, Z-Image, Qwen, and video models like LTX2 and HunyuanVideo. Whether you are training a small LoRA adapter on an 8 GB GPU or running a full fine-tune across multiple GPUs, Serenity offers the configuration surface and memory management tools to get it done.

The framework is built around a clean separation of concerns: model adapters, data pipeline, training utilities, and memory management each live in their own layer with well-defined interfaces. Configuration is handled through YAML or JSON files with over 150 tunable fields, automatic enum coercion, and a migration system that keeps older configs compatible as the schema evolves. A rich set of optimizers (45+), learning rate schedulers, loss functions, and noise strategies gives you fine-grained control over every aspect of the training loop.

Serenity also supports LyCORIS adapters (LoHa, LoKR, full-matrix decomposition), textual inversion / embedding training, VAE fine-tuning, EMA weight averaging, mixed-precision training with stochastic rounding, aspect-ratio bucketing, disk-based latent and text-encoder caching, and sample generation during training for visual progress tracking. It is designed for researchers, hobbyists, and production teams who need a flexible, extensible training toolkit.

## Key Features

### Model Support
- **Stable Diffusion 1.5** -- including inpainting variants
- **SDXL 1.0** -- base and inpainting
- **Stable Diffusion 3 / 3.5** -- flow-matching based architectures
- **Flux 1** -- Dev, Schnell, and Fill (inpainting) variants
- **Flux 2** -- including Klein 4B and Klein 9B compact models
- **Chroma** -- Chroma 1 diffusion model
- **Z-Image** -- high-quality image generation
- **LTX2** -- video generation model
- **HunyuanVideo** -- video generation
- **Qwen** -- image generation and image editing modes

### Training Methods
- **LoRA** -- low-rank adaptation with configurable rank, alpha, and layer filtering
- **LyCORIS** -- LoKR (Kronecker product), LoHa, full-matrix, Tucker decomposition, weight decomposition (DoRA), RS-LoRA
- **Embedding / Textual Inversion** -- train new token embeddings with norm preservation
- **Full Fine-Tune** -- unrestricted weight updates across the entire model
- **VAE Fine-Tune** -- train the variational autoencoder independently

### Optimizers (45+)
- **Standard**: Adam, AdamW, SGD, RMSprop, Adagrad
- **8-bit (bitsandbytes)**: AdamW 8-bit, Adam 8-bit, SGD 8-bit, Lion 8-bit, LAMB 8-bit, LARS 8-bit, RMSprop 8-bit, Adagrad 8-bit, AdEMAMix 8-bit, CAME 8-bit
- **Adaptive LR**: Prodigy, Prodigy+ScheduleFree, D-Adaptation (SGD, Adam, Adan, AdaGrad, Lion)
- **Schedule-Free**: ScheduleFree AdamW, ScheduleFree SGD
- **Advanced**: Lion, LAMB, LARS, AdEMAMix, CAME, Muon, AdaMuon, ADOPT, SignSGD
- **Research**: AdaBelief, Tiger, Aida, Yogi
- **Fused back-pass** support for Adafactor, CAME, Prodigy+, and advanced optimizer variants

### Learning Rate Schedulers
- Constant, Linear, Cosine, Cosine with Restarts, Cosine with Hard Restarts
- REX scheduler
- Adafactor internal scheduler
- Custom scheduler (user-defined Python callable)
- Configurable warmup steps, cycle count, and minimum LR factor

### Loss Functions & Weighting
- **Functions**: MSE (L2), MAE (L1), Huber, Log-Cosh, VB (variational bound)
- **Weighting**: Constant, MIN_SNR_GAMMA, P2, Debiased Estimation, Sigma
- **Scaling**: Per-batch, global-batch, gradient-accumulation, and combined modes
- **Masking**: Spatial loss masking with per-image masks, random circular masks, and masked prior preservation

### Noise & Timestep Control
- Offset noise with generalized mode
- Perturbation noise injection
- Zero terminal SNR rescaling
- V-prediction and epsilon-prediction forcing
- 6 timestep distributions: Uniform, Sigmoid, Logit-Normal, Heavy-Tail, CosMap, Inverted Parabola
- Configurable min/max noising strength, shift, and dynamic timestep shifting

### EMA (Exponential Moving Average)
- GPU or CPU weight averaging
- Configurable decay rate and update interval
- Non-EMA sampling option for comparison

### Precision & Quantization
- fp16, bf16, tf32, fp32 training dtypes
- bf16 stochastic rounding for improved low-precision training
- Mixed precision with GradScaler
- Quantization: NF4, FP8, INT8, GGUF (including GGUF A8 float/int variants)
- Per-component weight dtype control (transformer, text encoders, VAE)

### Data Pipeline
- Aspect ratio bucketing with configurable resolution
- Disk-based latent caching and text-encoder output caching
- Video frame extraction for video model training
- Multi-threaded data loading
- Concept-based dataset organization with per-concept captions and settings

### Augmentations
- Random crop, horizontal flip
- Rotation, circular padding shift
- Noise injection, Gaussian blur
- Per-concept augmentation configuration

### Masking
- Per-image mask support (PNG masks)
- Random circular mask generation
- Mask augmentation and normalization
- Masked prior preservation weighting
- Inpainting model support with conditioning images

### Checkpointing & Output
- SafeTensors, Diffusers directory format, CKPT, ComfyUI LoRA, internal format
- Rolling backups with configurable count
- Periodic saving by epoch, step, or time interval
- Backup-before-save safety
- Resume from last backup

### Multi-GPU
- DDP (DistributedDataParallel) training
- Fused and async gradient reduction
- Per-device index selection
- Global batch loss and LR scaling

### Memory Management
- Gradient checkpointing (on-device and CPU-offloaded)
- Layer offloading with configurable fraction
- Async activation offloading
- Pinned memory pools and allocation strategies
- Memory conductor for coordinated offloading

### Monitoring & Callbacks
- TensorBoard logging with optional port exposure
- Sample image/video generation during training
- Validation loops with configurable frequency
- Training progress tracking

### Configuration System
- YAML and JSON config file support
- 150+ configurable fields
- Automatic enum coercion with alias support
- Config migration system for schema evolution
- Preset loading for common VRAM tiers
- Per-component sub-configs (transformer, text encoders, VAE, optimizer)

---

## Installation

```bash
git clone https://github.com/CodeAlexx/Serenity.git
cd Serenity
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Prerequisites

- Python 3.10 or later
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with 8 GB+ VRAM (24 GB+ recommended for full fine-tuning)

---

## Quick Start

Create a minimal YAML configuration file:

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

Run training:

```bash
python -m serenity.cli.commands train config.yaml
```

---

## Project Structure

```
serenity/
├── core/           # Config, interfaces, enums, trainer loop
├── models/         # Model adapters (SD15, SDXL, SD3, Flux 1/2, Chroma,
│                   #   Z-Image, LTX2, HunyuanVideo, Qwen)
├── training/       # Optimizers, schedulers, losses, EMA, precision,
│                   #   gradient management, distributed, embedding training
├── pipeline/       # Dataset loading, bucketing, caching, augmentations,
│                   #   concepts, masks, video frame extraction
├── data/           # Data utilities and dataloader
├── checkpoint/     # Save, load, convert, and resume checkpoints
├── memory/         # Memory conductor, layer offloading, allocation strategies
├── sampling/       # Inference sampling during training
├── presets/        # VRAM-tier preset configurations (8GB, 16GB, 24GB)
├── cli/            # Command-line interface
├── adapters/       # Adapter layer (LoRA, LyCORIS)
├── utils/          # General utilities
└── tests/          # Test suite
```

---

## Configuration

Serenity uses a dataclass-based configuration system (`TrainConfig`) with over 150 fields organized into logical sections:

| Section | Key Fields |
|---------|-----------|
| **Model** | `model_type`, `training_method`, `transformer_path`, `base_model_name` |
| **Training** | `learning_rate`, `epochs`, `batch_size`, `gradient_accumulation_steps`, `seed` |
| **Optimizer** | Nested `optimizer` config with type, betas, eps, weight decay, and optimizer-specific params |
| **Precision** | `train_dtype`, `output_dtype`, `fallback_train_dtype`, per-component `weight_dtype` |
| **LoRA** | `lora_rank`, `lora_alpha`, `peft_type`, `lora_weight_dtype`, layer filtering |
| **LyCORIS/LoKR** | `lokr_dim`, `lokr_alpha`, `lokr_factor`, Tucker, weight decompose, full matrix |
| **Loss** | `mse_strength`, `mae_strength`, `huber_strength`, `loss_weight_fn`, `loss_scaler` |
| **Noise** | `offset_noise_weight`, `timestep_distribution`, `min_noising_strength` |
| **Data** | `concepts`, `resolution`, `aspect_ratio_bucketing`, `latent_caching` |
| **Sampling** | `sample_after`, `sample_after_unit`, `sample_image_format` |
| **Checkpointing** | `backup_after`, `rolling_backup`, `output_model_format` |
| **Multi-GPU** | `multi_gpu`, `device_indexes`, `fused_gradient_reduce` |
| **Memory** | `gradient_checkpointing`, `layer_offload_fraction`, `enable_async_offloading` |

Configs can be written in YAML or JSON. Older config files are automatically migrated to the current schema version when loaded.

---

## Supported Models

| Model Family | LoRA | Full Fine-Tune | Embedding/TI | Inpainting | Notes |
|-------------|------|---------------|-------------|------------|-------|
| SD 1.5 | Yes | Yes | Yes | Yes | Classic architecture |
| SDXL 1.0 | Yes | Yes | Yes | Yes | Dual text encoder |
| SD 3 / 3.5 | Yes | Yes | -- | -- | Flow matching |
| Flux 1 (Dev/Schnell) | Yes | Yes | -- | Yes (Fill) | Flow matching |
| Flux 2 | Yes | Yes | -- | -- | Next-gen Flux |
| Flux 2 Klein 4B/9B | Yes | Yes | -- | -- | Compact variants |
| Chroma | Yes | Yes | -- | -- | Flow matching |
| Z-Image | Yes | Yes | -- | -- | High quality |
| LTX2 | Yes | Yes | -- | -- | Video model |
| HunyuanVideo | Yes | Yes | -- | -- | Video model |
| Qwen | Yes | Yes | -- | -- | Image editing mode |

---

## Version

**pre-alpha 0.055**

This project is in active early development. APIs, configuration schemas, and behavior may change between versions.

---

## License

See [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome. Serenity is currently in pre-alpha, so expect breaking changes and rough edges. If you would like to contribute:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description of the change

Bug reports, feature requests, and documentation improvements are all appreciated. Please open an issue before starting large changes to discuss the approach.

---

## Acknowledgments

Serenity builds on the shoulders of the open-source diffusion model community. Special thanks to the teams behind PyTorch, Diffusers, bitsandbytes, LyCORIS, and the many optimizer and scheduler libraries that make this work possible.
