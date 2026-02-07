# Serenity User Guide

**Version: pre-alpha 0.055**

A comprehensive guide to configuring and running diffusion model training with Serenity.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Configuration](#2-configuration)
3. [Model Types](#3-model-types)
4. [Training Methods](#4-training-methods)
5. [Data Preparation](#5-data-preparation)
6. [Optimizers](#6-optimizers)
7. [Learning Rate Schedulers](#7-learning-rate-schedulers)
8. [Loss Functions](#8-loss-functions)
9. [Memory Optimization](#9-memory-optimization)
10. [Sampling](#10-sampling)
11. [Multi-GPU Training](#11-multi-gpu-training)
12. [Presets](#12-presets)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Getting Started

### Prerequisites

- **Python**: 3.10 or later (3.12 recommended)
- **PyTorch**: 2.0 or later with CUDA support
- **CUDA**: Compatible NVIDIA GPU driver
- **GPU**: Minimum 8 GB VRAM for LoRA training; 16-24 GB+ for full fine-tuning
- **Disk**: Sufficient space for model weights (2-20 GB per model) and latent caches

### Installation

```bash
git clone https://github.com/CodeAlexx/Serenity.git
cd Serenity
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Optional Dependencies

Some optimizers require additional packages. Install them as needed:

```bash
# 8-bit optimizers
pip install bitsandbytes

# Prodigy optimizer
pip install prodigyopt

# Prodigy+ Schedule-Free
pip install prodigyplus

# D-Adaptation optimizers
pip install dadaptation

# Lion optimizer
pip install lion-pytorch

# Schedule-Free optimizers
pip install schedulefree

# CAME optimizer
pip install came-pytorch

# Advanced optimizers (Muon, ADOPT, etc.)
pip install adv_optm

# Muon (upstream)
pip install muon

# AdaBelief (via timm)
pip install timm

# Tiger, Aida, ADOPT, Yogi
pip install pytorch-optimizer

# Adafactor (via transformers)
pip install transformers

# YAML config support
pip install pyyaml
```

### Verifying Installation

```bash
python -c "import serenity; print('Serenity loaded successfully')"
```

### Running Your First Training

```bash
python -m serenity.cli.commands train my_config.yaml
```

---

## 2. Configuration

Serenity uses a dataclass-based configuration system loaded from YAML or JSON files. The main config object is `TrainConfig`, which contains over 150 fields organized into logical groups. All enum values are coerced automatically -- you can use strings like `"ADAMW"` or `"adamw"` interchangeably.

### Minimal Config Example (YAML)

```yaml
# Required fields
model_type: sdxl
training_method: lora
transformer_path: "stabilityai/stable-diffusion-xl-base-1.0"
output_dir: "output/"
concepts:
  - name: "subject"
    path: "data/subject"
    prompt:
      source: "txt"

# Training basics
learning_rate: 1.0e-4
epochs: 10
batch_size: 1
resolution: "1024"
train_dtype: BFLOAT_16

# LoRA settings
lora_rank: 16
lora_alpha: 1.0

# Output
output_model_destination: "output/my_lora.safetensors"
output_model_format: SAFETENSORS

# Optimizer (nested object)
optimizer:
  optimizer: ADAMW
  weight_decay: 0.01
```

### Minimal Config Example (JSON)

```json
{
  "model_type": "sdxl",
  "training_method": "lora",
  "transformer_path": "stabilityai/stable-diffusion-xl-base-1.0",
  "output_dir": "output/",
  "concepts": [
    {
      "name": "subject",
      "path": "data/subject",
      "prompt": { "source": "txt" }
    }
  ],
  "learning_rate": 1e-4,
  "epochs": 10,
  "batch_size": 1,
  "resolution": "1024",
  "train_dtype": "BFLOAT_16",
  "lora_rank": 16,
  "lora_alpha": 1.0,
  "output_model_destination": "output/my_lora.safetensors",
  "optimizer": {
    "optimizer": "ADAMW"
  }
}
```

### Key Configuration Sections

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | string | Model architecture (e.g., `sdxl`, `flux_dev`, `sd15`, `ltx2`) |
| `training_method` | string | One of `lora`, `fine_tune`, `fine_tune_vae`, `embedding` |
| `transformer_path` | string | Path or HuggingFace ID for the base model weights |
| `output_dir` | string | Directory for training outputs |
| `concepts` | list | Training data concepts (see Data Preparation) |

#### Training Settings

| Field | Default | Description |
|-------|---------|-------------|
| `learning_rate` | `1e-4` | Global learning rate |
| `epochs` | `100` | Number of training epochs |
| `max_train_steps` | `null` | Optional hard step limit (overrides epochs) |
| `batch_size` | `1` | Per-device batch size |
| `gradient_accumulation_steps` | `1` | Accumulation steps before optimizer update |
| `seed` | `42` | Random seed for reproducibility |
| `resolution` | `"512"` | Training resolution (string, e.g. `"1024"` or `"512x768"`) |
| `frames` | `"25"` | Frame count for video models |

#### Precision Settings

| Field | Default | Description |
|-------|---------|-------------|
| `train_dtype` | `FLOAT_16` | Training precision: `FLOAT_16`, `BFLOAT_16`, `FLOAT_32`, `TFLOAT_32` |
| `fallback_train_dtype` | `BFLOAT_16` | Fallback if primary dtype is unsupported |
| `output_dtype` | `FLOAT_32` | Precision for saved model weights |

#### Model Part Configs

Each model component (transformer, text encoders, VAE) has its own sub-config with:
- `model_name` -- optional override path
- `train` -- whether to train this component
- `stop_training_after` / `stop_training_after_unit` -- freeze component after N epochs/steps
- `learning_rate` -- per-component LR override
- `weight_dtype` -- per-component precision
- `dropout_probability` -- component-level dropout

Example:

```yaml
text_encoder:
  train: true
  stop_training_after: 30
  stop_training_after_unit: EPOCH
  learning_rate: 5.0e-5

transformer:
  train: true
  learning_rate: 1.0e-4
  weight_dtype: BFLOAT_16
```

#### Config Migration

Serenity includes an automatic config migration system. When you load a config file that was written for an older schema version, it is automatically upgraded to the current format. This means you generally do not need to manually update old config files.

---

## 3. Model Types

Serenity supports a broad range of diffusion model architectures. Set the `model_type` field to one of the following values.

### Stable Diffusion 1.5

```yaml
model_type: sd15          # Standard SD 1.5
model_type: sd15_inpainting  # Inpainting variant
```

The original Stable Diffusion architecture. Small model size (~860 MB), fast training, low VRAM requirements. Good for learning and experimentation. Supports LoRA, full fine-tune, embedding training, and inpainting.

### SDXL 1.0

```yaml
model_type: sdxl          # SDXL 1.0
model_type: sdxl_10_base  # Explicit base variant
model_type: sdxl_inpainting  # Inpainting
```

Dual text encoder architecture (CLIP + OpenCLIP) with higher default resolution (1024px). Larger model, better quality, higher VRAM requirement.

### Stable Diffusion 3 / 3.5

```yaml
model_type: sd3           # SD 3.0
model_type: sd35          # SD 3.5
```

Flow-matching based architecture with MMDiT (multimodal diffusion transformer). Supports up to three text encoders.

### Flux 1

```yaml
model_type: flux_dev      # Flux Dev (guidance-distilled)
model_type: flux_schnell  # Flux Schnell (fast, few-step)
model_type: flux_fill_dev # Flux Fill Dev (inpainting)
```

Black Forest Labs flow-matching transformer models. High quality, efficient inference.

### Flux 2

```yaml
model_type: flux_2            # Flux 2 standard
model_type: flux_2_klein_4b   # Klein 4B (compact)
model_type: flux_2_klein_9b   # Klein 9B (compact)
```

Next-generation Flux architecture. Klein variants are compact models designed for lower VRAM usage while maintaining quality.

### Chroma

```yaml
model_type: chroma_1
```

Flow-matching based diffusion model.

### Z-Image

```yaml
model_type: zimage
```

High-quality image generation model. Aliases `z_image` and `z-image` are also accepted.

### LTX2 (Video)

```yaml
model_type: ltx2
```

Video generation model. Use the `frames` config field to set the number of output frames. Aliases `ltx`, `ltx_video`, and `ltxvideo` are accepted.

### HunyuanVideo

```yaml
model_type: hunyuan_video
```

Video generation model with dual text encoder support.

### Qwen

```yaml
model_type: qwen              # Image generation
model_type: qwen_image_edit   # Image editing mode
```

Qwen-based diffusion model with an image editing variant that accepts source images for modification.

---

## 4. Training Methods

### LoRA (Low-Rank Adaptation)

```yaml
training_method: lora
lora_rank: 16
lora_alpha: 1.0
lora_weight_dtype: FLOAT_32
```

LoRA injects small trainable matrices into the model while keeping the base weights frozen. This dramatically reduces VRAM usage and training time.

- **`lora_rank`**: The rank of the low-rank matrices. Higher rank = more capacity but more VRAM. Common values: 4, 8, 16, 32, 64.
- **`lora_alpha`**: Scaling factor for LoRA weights. Often set to 1.0 or equal to rank.
- **`layer_filter`**: Restrict LoRA to specific layers (string pattern or regex with `layer_filter_regex: true`).
- **`layer_filter_preset`**: Use `"full"` for all layers or a model-specific subset.

LoRA produces small output files (typically 1-100 MB) that can be loaded on top of any compatible base model.

### LyCORIS / LoKR

```yaml
training_method: lora
peft_type: LOKR
lokr_dim: 16
lokr_alpha: 16.0
lokr_factor: -1
lokr_use_tucker: false
lokr_weight_decompose: false
lokr_full_matrix: false
lokr_rs_lora: false
```

LyCORIS (Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion) provides advanced adapter types beyond standard LoRA:

- **LoKR (Kronecker product)**: Efficient parameter decomposition using Kronecker products. Set `peft_type: LOKR`.
- **Tucker decomposition**: Enable with `lokr_use_tucker: true` for additional compression.
- **Weight decomposition (DoRA)**: Enable with `lokr_weight_decompose: true` for direction-magnitude separation.
- **Full matrix**: Enable with `lokr_full_matrix: true` to train full-rank updates (no decomposition).
- **RS-LoRA**: Enable with `lokr_rs_lora: true` for rank-stabilized LoRA scaling.

### Embedding / Textual Inversion

```yaml
training_method: embedding
embedding_learning_rate: 5.0e-4
preserve_embedding_norm: false
embedding_weight_dtype: FLOAT_32
```

Trains new token embeddings that represent a concept. The model weights remain completely frozen -- only the embedding vectors are updated. Output is a small embedding file that can be used as a trigger word during inference.

- **`embedding_learning_rate`**: Separate LR for embeddings (typically higher than LoRA LR).
- **`preserve_embedding_norm`**: Constrains embedding magnitude to prevent drift.

### Full Fine-Tune

```yaml
training_method: fine_tune
```

Updates all model weights without restrictions. Produces the highest quality results but requires significantly more VRAM (24 GB+) and training time. The output is a full model checkpoint.

Best practices for full fine-tuning:
- Use gradient checkpointing to reduce VRAM
- Lower batch size and use gradient accumulation
- Consider CPU-offloaded EMA
- Use bf16 precision to save memory

### VAE Fine-Tune

```yaml
training_method: fine_tune_vae
```

Trains only the VAE (variational autoencoder) component. Useful for improving image decode quality or adapting the VAE to a specific domain.

---

## 5. Data Preparation

### Directory Structure

Organize your training data into concept folders. Each concept represents a distinct subject, style, or category:

```
training_data/
├── concept_a/
│   ├── image_001.png
│   ├── image_001.txt        # Caption file (same name as image)
│   ├── image_001-masklabel.png  # Optional mask
│   ├── image_002.jpg
│   ├── image_002.txt
│   └── ...
├── concept_b/
│   ├── photo_01.png
│   ├── photo_01.txt
│   └── ...
```

### Concepts Configuration

Each concept is defined in the `concepts` list of your config:

```yaml
concepts:
  - name: "my_character"
    path: "training_data/concept_a"
    prompt:
      source: "txt"              # Read captions from .txt files
    # Optional settings:
    # seed: 42
    # include_subdirectories: true
    # image_variations: 1
    # text_variations: 1
    # balancing: 1.0
    # loss_weight: 1.0
```

### Captions

Captions can come from several sources:

- **Text files** (`source: "txt"`): Place a `.txt` file next to each image with the same base name. The file content becomes the caption.
- **Filename** (`source: "filename"`): The image filename (without extension) is used as the caption.
- **Fixed prompt** (`source: "fixed"`, `text: "a photo of XYZ"`): All images in the concept share the same prompt.

### Masks (Optional)

For masked training, place mask images alongside your training images:

- Mask filename pattern: `{image_name}-masklabel.png`
- White pixels (255) indicate regions of interest
- Black pixels (0) indicate regions to ignore
- The mask is used to weight the loss spatially

Enable masked training in the config:

```yaml
masked_training: true
unmasked_probability: 0.1     # Probability of training without mask
unmasked_weight: 0.1          # Loss weight for unmasked regions
normalize_masked_area_loss: false
```

You can also use random circular masks without providing mask files:

```yaml
masked_training: true
# Random masks are generated when no mask file is found
```

### Image Requirements

- Supported formats: PNG, JPG, JPEG, WEBP, BMP
- Images are automatically resized and bucketed by aspect ratio
- For best results, provide images at or above your target training resolution
- Consistent quality and style within each concept improves results

### Video Data (LTX2, HunyuanVideo)

For video models, place video files in the concept directory. Serenity extracts frames automatically according to the `frames` config field.

### Latent Caching

Serenity supports disk-based latent caching to speed up training:

```yaml
latent_caching: true
clear_cache_before_training: true
cache_dir: "workspace-cache/run"
```

When enabled, VAE-encoded latents and text encoder outputs are cached to disk on the first pass. Subsequent epochs load from cache, skipping the encoding step entirely. This significantly speeds up multi-epoch training.

---

## 6. Optimizers

Serenity provides over 45 optimizer implementations organized into categories. The optimizer is configured as a nested object:

```yaml
optimizer:
  optimizer: ADAMW          # Optimizer type
  weight_decay: 0.01        # Weight decay
  eps: 1.0e-8               # Epsilon for numerical stability
  # ... additional optimizer-specific parameters
```

### Standard Optimizers

| Optimizer | Config Value | Best For |
|-----------|-------------|----------|
| AdamW | `ADAMW` | General purpose, most training scenarios |
| Adam | `ADAM` | When decoupled weight decay is not needed |
| SGD | `SGD` | Simple training, when momentum suffices |

AdamW is the recommended default for most training runs.

### 8-bit Optimizers (bitsandbytes)

| Optimizer | Config Value | Savings |
|-----------|-------------|---------|
| AdamW 8-bit | `ADAMW_8BIT` | ~50% optimizer VRAM |
| Adam 8-bit | `ADAM_8BIT` | ~50% optimizer VRAM |
| SGD 8-bit | `SGD_8BIT` | ~50% optimizer VRAM |
| Lion 8-bit | `LION_8BIT` | ~50% optimizer VRAM |
| LAMB 8-bit | `LAMB_8BIT` | ~50% optimizer VRAM |
| LARS 8-bit | `LARS_8BIT` | ~50% optimizer VRAM |
| RMSprop 8-bit | `RMSPROP_8BIT` | ~50% optimizer VRAM |
| Adagrad 8-bit | `ADAGRAD_8BIT` | ~50% optimizer VRAM |
| AdEMAMix 8-bit | `ADEMAMIX_8BIT` | ~50% optimizer VRAM |
| CAME 8-bit | `CAME_8BIT` | ~50% optimizer VRAM |

8-bit optimizers quantize optimizer states to 8-bit, cutting optimizer memory roughly in half with minimal quality impact. Requires the `bitsandbytes` package.

### Adaptive Learning Rate Optimizers

| Optimizer | Config Value | Notes |
|-----------|-------------|-------|
| Prodigy | `PRODIGY` | Auto-tunes learning rate; set LR to 1.0 |
| Prodigy+ ScheduleFree | `PRODIGY_PLUS_SCHEDULE_FREE` | No LR scheduler needed |
| D-Adapt Adam | `DADAPT_ADAM` | Automatic LR adaptation |
| D-Adapt SGD | `DADAPT_SGD` | Automatic LR adaptation |
| D-Adapt Adan | `DADAPT_ADAN` | Automatic LR with Adan optimizer |
| D-Adapt AdaGrad | `DADAPT_ADA_GRAD` | Automatic LR adaptation |
| D-Adapt Lion | `DADAPT_LION` | Automatic LR adaptation |

Adaptive optimizers determine their own effective learning rate. When using Prodigy, set `learning_rate: 1.0` and let the optimizer find the optimal rate.

### Schedule-Free Optimizers

| Optimizer | Config Value | Notes |
|-----------|-------------|-------|
| ScheduleFree AdamW | `SCHEDULE_FREE_ADAMW` | No scheduler needed |
| ScheduleFree SGD | `SCHEDULE_FREE_SGD` | No scheduler needed |

Schedule-free optimizers achieve good results without a learning rate schedule. Set `learning_rate_scheduler: CONSTANT` when using these.

### Advanced Optimizers

| Optimizer | Config Value | Notes |
|-----------|-------------|-------|
| Lion | `LION` | Memory-efficient, sign-based updates |
| LAMB | `LAMB` | Layer-wise adaptive, good for large batch |
| LARS | `LARS` | Large-batch training |
| AdEMAMix | `ADEMAMIX` | Exponential moving average mixture |
| CAME | `CAME` | Confidence-Aware optimizer |
| Adafactor | `ADAFACTOR` | Memory-efficient, factored second moments |
| Muon | `MUON` | Matrix-free second-order updates |

### Research Optimizers

| Optimizer | Config Value | Notes |
|-----------|-------------|-------|
| AdaBelief | `ADABELIEF` | Adapts step size by belief in gradient |
| Tiger | `TIGER` | Lightweight optimizer |
| Aida | `AIDA` | Advanced optimizer |
| ADOPT | `ADOPT` | Decoupled weight decay variant |
| Yogi | `YOGI` | Controlled adaptive learning rates |

### Advanced Variant Optimizers (adv_optm)

| Optimizer | Config Value | Notes |
|-----------|-------------|-------|
| AdamW Adv | `ADAMW_ADV` | Stochastic rounding support |
| ADOPT Adv | `ADOPT_ADV` | Stochastic rounding support |
| Prodigy Adv | `PRODIGY_ADV` | Stochastic rounding + adaptive LR |
| Lion Adv | `LION_ADV` | Stochastic rounding support |
| Lion Prodigy Adv | `LION_PRODIGY_ADV` | Lion + adaptive LR |
| Simplified AdEMAMix | `SIMPLIFIED_ADEMAMIX` | Simplified dual-EMA mixture |
| SignSGD Adv | `SIGNSGD_ADV` | Sign-based gradient with stochastic rounding |
| Muon Adv | `MUON_ADV` | Matrix orthogonalization with stochastic rounding |
| AdaMuon Adv | `ADAMUON_ADV` | Adam + Muon hybrid |

These variants add bf16 stochastic rounding support, which improves training stability in low-precision modes.

### Fused Back-Pass

Some optimizers support fused back-pass, which combines the backward pass and optimizer step for reduced memory:

```yaml
optimizer:
  optimizer: ADAFACTOR
  fused_back_pass: true
```

Supported by: Adafactor, CAME, CAME 8-bit, Adam, AdamW, and all `_ADV` variant optimizers.

---

## 7. Learning Rate Schedulers

Configure the learning rate schedule with:

```yaml
learning_rate_scheduler: COSINE
learning_rate_warmup_steps: 200
learning_rate_cycles: 1.0
learning_rate_min_factor: 0.0
```

### Available Schedulers

| Scheduler | Config Value | Description |
|-----------|-------------|-------------|
| **Constant** | `CONSTANT` | Fixed LR after warmup. Simple and reliable. |
| **Linear** | `LINEAR` | Linear decay from initial LR to `min_factor * LR`. |
| **Cosine** | `COSINE` | Smooth cosine annealing. Most popular choice. |
| **Cosine with Restarts** | `COSINE_WITH_RESTARTS` | Cosine with periodic warm restarts. Use `cycles` to set restart count. |
| **Cosine with Hard Restarts** | `COSINE_WITH_HARD_RESTARTS` | Abrupt restarts instead of smooth transitions. |
| **REX** | `REX` | Reciprocal exponential schedule for stable convergence. |
| **Adafactor** | `ADAFACTOR` | Internal scheduler for Adafactor optimizer. Use only with Adafactor. |
| **Custom** | `CUSTOM` | User-defined Python callable specified via `custom_learning_rate_scheduler`. |

### Warmup

All schedulers support a warmup phase:

```yaml
learning_rate_warmup_steps: 200    # Linear ramp from 0 to LR over this many steps
```

Warmup prevents instability in early training when weights are far from converged.

### Guidance for Scheduler Selection

- **LoRA training (short runs)**: `CONSTANT` or `COSINE` with minimal warmup
- **Full fine-tune (long runs)**: `COSINE` or `COSINE_WITH_RESTARTS`
- **Adaptive optimizers (Prodigy, D-Adapt)**: `CONSTANT` (the optimizer adapts internally)
- **Schedule-free optimizers**: `CONSTANT` (built-in scheduling)
- **Adafactor with relative_step**: `ADAFACTOR` scheduler

---

## 8. Loss Functions

Serenity supports multiple loss functions that can be blended together with strength weights:

```yaml
mse_strength: 1.0          # MSE loss weight (default)
mae_strength: 0.0          # MAE loss weight
huber_strength: 0.0        # Huber loss weight
huber_delta: 1.0           # Huber delta parameter
log_cosh_strength: 0.0     # Log-Cosh loss weight
vb_loss_strength: 1.0      # VB loss weight
```

### Loss Function Types

| Function | Field | Description | When to Use |
|----------|-------|-------------|------------|
| **MSE** | `mse_strength` | Mean squared error (L2). Penalizes large errors heavily. | Default for most training. Stable convergence. |
| **MAE** | `mae_strength` | Mean absolute error (L1). Equal penalty for all error magnitudes. | When you want sharper outputs with less smoothing. |
| **Huber** | `huber_strength` | Combination of MSE (small errors) and MAE (large errors). | Robust to outliers. Good for noisy datasets. |
| **Log-Cosh** | `log_cosh_strength` | Smooth approximation of MAE. Differentiable everywhere. | Similar to Huber but with smoother gradients. |
| **VB** | `vb_loss_strength` | Variational bound loss. | Models with variational components. |

You can combine multiple losses by setting non-zero strengths for more than one:

```yaml
mse_strength: 0.8
mae_strength: 0.2          # 80% MSE + 20% MAE blend
```

### Loss Weighting Functions

```yaml
loss_weight_fn: MIN_SNR_GAMMA
loss_weight_strength: 5.0
```

| Weighting | Config Value | Description |
|-----------|-------------|-------------|
| **Constant** | `CONSTANT` | No weighting -- all timesteps contribute equally. |
| **MIN_SNR_GAMMA** | `MIN_SNR_GAMMA` | From the Min-SNR paper. Reduces weight for very high/low SNR timesteps. Strength controls gamma. |
| **P2** | `P2` | Perception Prioritized weighting. Focuses on perceptually important timesteps. |
| **Debiased Estimation** | `DEBIASED_ESTIMATION` | Corrects for training bias in noise prediction. |
| **Sigma** | `SIGMA` | Signal-based weighting. Compatible with flow-matching models. |

### Loss Scaling

```yaml
loss_scaler: NONE          # No scaling (default)
# Other options: BATCH, GLOBAL_BATCH, GRADIENT_ACCUMULATION, BOTH, GLOBAL_BOTH
```

Loss scaling normalizes the loss by batch size, accumulation steps, or both. Use `BOTH` or `GLOBAL_BOTH` when combining large effective batch sizes with gradient accumulation to keep gradients stable.

---

## 9. Memory Optimization

### By VRAM Tier

#### 8 GB VRAM

```yaml
training_method: lora
lora_rank: 8                          # Keep rank low
batch_size: 1
gradient_accumulation_steps: 4        # Simulate larger batch
gradient_checkpointing: on
train_dtype: BFLOAT_16
latent_caching: true

# Use 8-bit optimizer
optimizer:
  optimizer: ADAMW_8BIT

# Quantize base model
transformer:
  weight_dtype: NFLOAT_4              # NF4 quantization

# Layer offloading
layer_offload_fraction: 0.5           # Offload 50% of layers to CPU
enable_async_offloading: true
```

#### 16 GB VRAM

```yaml
training_method: lora
lora_rank: 16
batch_size: 1
gradient_accumulation_steps: 2
gradient_checkpointing: on
train_dtype: BFLOAT_16
latent_caching: true

optimizer:
  optimizer: ADAMW_8BIT               # or PRODIGY
```

#### 24 GB+ VRAM

```yaml
training_method: lora                 # or fine_tune
lora_rank: 32
batch_size: 2
gradient_checkpointing: on            # Still recommended for large models
train_dtype: BFLOAT_16
latent_caching: true

optimizer:
  optimizer: ADAMW
```

### Memory Reduction Techniques

#### Gradient Checkpointing

```yaml
gradient_checkpointing: on            # Recompute activations during backward pass
# gradient_checkpointing: cpu_offloaded  # Offload checkpoints to CPU RAM
```

Trades compute for memory by recomputing intermediate activations instead of storing them. Typically reduces VRAM by 30-50% with a 15-25% speed penalty.

#### Layer Offloading

```yaml
layer_offload_fraction: 0.5           # Offload 50% of model layers to CPU
enable_async_offloading: true          # Use async transfers
enable_activation_offloading: true     # Offload activations too
```

Moves inactive model layers to CPU RAM and swaps them in as needed. The fraction controls what portion of layers are offloaded (0.0 = none, 1.0 = all).

#### Quantization

Reduce base model memory with quantization:

```yaml
transformer:
  weight_dtype: NFLOAT_4    # NF4 (4-bit) -- smallest, ~4x reduction
  # weight_dtype: INT_8     # INT8 -- ~2x reduction
  # weight_dtype: FLOAT_8   # FP8 -- ~2x reduction
  # weight_dtype: GGUF      # GGUF format quantization
```

Quantization applies to frozen base model weights. Trainable parameters (LoRA matrices, embeddings) remain in full precision.

#### 8-bit Optimizers

Switch to an 8-bit optimizer variant to halve optimizer state memory:

```yaml
optimizer:
  optimizer: ADAMW_8BIT     # Instead of ADAMW
```

#### Latent Caching

```yaml
latent_caching: true
clear_cache_before_training: true
```

Caches VAE-encoded latents to disk. Eliminates VAE from the training loop after the first pass, freeing its VRAM.

#### Effective Batch Size with Accumulation

Instead of increasing batch size (which increases VRAM), use gradient accumulation:

```yaml
batch_size: 1
gradient_accumulation_steps: 8        # Effective batch size = 8
```

### Memory Management Architecture

Serenity's memory system has several coordinated components:

- **Memory Conductor**: Orchestrates memory allocation across CPU/GPU
- **Allocation Strategies**: Pinned memory pools for efficient CPU-GPU transfers
- **Sync Utilities**: Coordinated memory synchronization for multi-device setups
- **Checkpoint Layers**: Memory-efficient gradient checkpointing implementation

---

## 10. Sampling

Serenity can generate sample images/videos during training to track progress visually.

### Configuration

```yaml
sample_after: 10
sample_after_unit: MINUTE          # EPOCH, STEP, MINUTE, HOUR, SECOND, ALWAYS, NEVER
sample_skip_first: 0               # Skip sampling for the first N intervals
sample_image_format: JPG           # JPG or PNG
samples_to_tensorboard: true       # Include samples in TensorBoard
non_ema_sampling: true             # Sample from non-EMA weights too
```

### Sample Definitions

Provide sampling prompts and settings in a separate file referenced by `sample_definition_file_name`, or inline via the `samples` field:

```yaml
samples:
  - prompt: "a photo of my_subject in a garden"
    negative_prompt: "blurry, low quality"
    width: 1024
    height: 1024
    steps: 20
    seed: 42
```

### Noise Schedulers for Sampling

The sampling system supports multiple noise schedulers:

| Scheduler | Config Value |
|-----------|-------------|
| DDIM | `DDIM` |
| Euler | `EULER` |
| Euler Ancestral | `EULER_A` |
| DPM++ | `DPMPP` |
| DPM++ SDE | `DPMPP_SDE` |
| UniPC | `UNIPC` |
| Euler (Karras) | `EULER_KARRAS` |
| DPM++ (Karras) | `DPMPP_KARRAS` |
| DPM++ SDE (Karras) | `DPMPP_SDE_KARRAS` |
| UniPC (Karras) | `UNIPC_KARRAS` |

---

## 11. Multi-GPU Training

Serenity supports DDP (DistributedDataParallel) for training across multiple GPUs.

### Configuration

```yaml
multi_gpu: true
device_indexes: "0,1"              # Comma-separated GPU indices
fused_gradient_reduce: true        # Fuse gradient all-reduce with backward pass
async_gradient_reduce: true        # Overlap gradient reduction with compute
async_gradient_reduce_buffer: 100  # Buffer size for async reduction
```

### How It Works

- Each GPU processes a separate batch independently
- Gradients are synchronized via all-reduce after each step
- Fused gradient reduction overlaps communication with computation for higher throughput
- Loss and learning rate can be scaled by world size using the `loss_scaler` and `learning_rate_scaler` fields

### Scaling Considerations

```yaml
loss_scaler: GLOBAL_BATCH          # Scale loss by total batch across GPUs
learning_rate_scaler: GLOBAL_BATCH # Scale LR accordingly
```

When using multiple GPUs, the effective batch size is `batch_size * num_gpus * gradient_accumulation_steps`. You may want to scale the learning rate linearly with the effective batch size, or use the scaler settings to handle this automatically.

---

## 12. Presets

Presets are pre-configured JSON files for common training scenarios and VRAM tiers.

### Using a Preset

Presets live in the `serenity/presets/` directory. They contain complete config overrides for specific scenarios:

```
serenity/presets/
├── sd15_lora_8gb.json       # SD 1.5 LoRA on 8 GB GPU
├── sdxl_lora_16gb.json      # SDXL LoRA on 16 GB GPU
├── flux_lora_24gb.json      # Flux LoRA on 24 GB GPU
```

You can use a preset as your base config and override specific fields:

```bash
python -m serenity.cli.commands train serenity/presets/sdxl_lora_16gb.json
```

Or create a YAML config that references a preset and overrides fields:

```yaml
# Start from a preset's settings and customize
model_type: sdxl
training_method: lora
transformer_path: "stabilityai/stable-diffusion-xl-base-1.0"
output_dir: "output/"

# Override from preset defaults
learning_rate: 5.0e-5
epochs: 20

concepts:
  - name: "my_concept"
    path: "data/my_concept"
    prompt:
      source: "txt"
```

### Creating Custom Presets

Create a JSON file with any subset of `TrainConfig` fields. Fields not specified will use their defaults:

```json
{
  "model_type": "flux_dev",
  "training_method": "lora",
  "train_dtype": "BFLOAT_16",
  "lora_rank": 16,
  "batch_size": 1,
  "gradient_checkpointing": "on",
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "optimizer": "ADAMW_8BIT",
    "weight_decay": 0.01
  },
  "learning_rate": 1e-4,
  "learning_rate_scheduler": "COSINE",
  "learning_rate_warmup_steps": 100,
  "latent_caching": true
}
```

Save the file to `serenity/presets/` or any location, then pass it as the config path.

---

## 13. Troubleshooting

### Out of Memory (OOM)

**Symptoms**: `torch.cuda.OutOfMemoryError` or training crashes with CUDA errors.

**Solutions** (in order of impact):

1. **Enable gradient checkpointing**: `gradient_checkpointing: on`
2. **Reduce batch size**: `batch_size: 1` and increase `gradient_accumulation_steps`
3. **Switch to 8-bit optimizer**: `optimizer: ADAMW_8BIT`
4. **Quantize base model**: `transformer: { weight_dtype: NFLOAT_4 }`
5. **Reduce LoRA rank**: Lower `lora_rank` (try 4 or 8)
6. **Enable layer offloading**: `layer_offload_fraction: 0.5`
7. **Lower resolution**: Use smaller `resolution` value
8. **Disable latent caching decode step**: Set `only_cache: true` to cache without training in first pass

### Training Loss Not Decreasing

**Possible causes**:

- **Learning rate too low**: Increase `learning_rate` by 2-5x
- **Learning rate too high**: Decrease `learning_rate`; check for NaN losses
- **Bad captions**: Verify captions match image content
- **Too few images**: Use data augmentation or duplicate images with varied captions
- **Wrong model type**: Ensure `model_type` matches your base model

### NaN or Inf Loss Values

**Solutions**:

- Switch to `train_dtype: BFLOAT_16` (more stable than fp16)
- Enable gradient clipping: `clip_grad_norm: 1.0`
- Reduce learning rate
- Check for corrupted images in your dataset
- Ensure loss weight strength is reasonable (`loss_weight_strength: 5.0` is typical for MIN_SNR_GAMMA)

### Slow Training

**Possible causes**:

- **No latent caching**: Enable `latent_caching: true` to avoid re-encoding every epoch
- **CPU bottleneck**: Increase `dataloader_threads` (default 2, try 4-8)
- **Gradient checkpointing overhead**: Normal -- trades ~20% speed for significant VRAM savings
- **Layer offloading**: High `layer_offload_fraction` values slow down training due to CPU-GPU transfers

### Config Errors

- **Unknown model type**: Check spelling, use lowercase. Aliases like `sdxl`, `sd15`, `flux_dev` work.
- **Unsupported training method**: Valid values are `lora`, `fine_tune`, `fine_tune_vae`, `embedding`.
- **Missing required fields**: `model_type`, `training_method`, `transformer_path`, `output_dir`, and `concepts` are required.
- **YAML parse errors**: Ensure proper indentation and quoting of string values.

### Checkpoint Issues

- **Cannot resume**: Set `continue_last_backup: true` and ensure `workspace_dir` points to the original training workspace.
- **Output format**: Check `output_model_format` matches your target (e.g., `SAFETENSORS` for most use cases, `DIFFUSERS` for directory format).

### Multi-GPU Issues

- **GPUs not detected**: Verify `device_indexes` lists valid GPU indices. Check `nvidia-smi` for available devices.
- **Uneven memory usage**: This is normal -- rank 0 typically uses slightly more memory.
- **Hang during gradient sync**: Try disabling `async_gradient_reduce` as a diagnostic step.

---

## Further Reading

- [Quick Start Guide](QuickStartGuide.md)
- [Project Structure](ProjectStructure.md)
- [Overview](Overview.md)
- [Embedding Training](EmbeddingTraining.md)
- [Captioning and Masking](CaptioningAndMasking.md)
- [RAM Offloading](RamOffloading.md)
- [CLI Training](CliTraining.md)
- [Contributing](Contributing.md)

---

*Serenity pre-alpha 0.055*
