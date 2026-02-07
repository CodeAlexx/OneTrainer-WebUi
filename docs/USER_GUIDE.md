# Serenity User Guide

**Version: pre-alpha 0.055**

Reference for configuring and running training with Serenity. This covers most of the config options but may not be 100% in sync with the code -- when in doubt, check `serenity/core/config.py`.

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

### What You Need

- **Python**: 3.10+ (3.12 works fine)
- **PyTorch**: 2.0+ with CUDA
- **GPU**: 8 GB VRAM minimum for LoRA. 16-24 GB+ for full fine-tune depending on model.
- **Disk**: Enough room for model weights (2-20 GB per model) plus latent cache files

### Installation

```bash
git clone https://github.com/CodeAlexx/Serenity.git
cd Serenity
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Optional Packages

Some optimizers need extra packages. Install as needed:

```bash
pip install bitsandbytes       # 8-bit optimizers
pip install prodigyopt          # Prodigy
pip install prodigyplus         # Prodigy+ Schedule-Free
pip install dadaptation         # D-Adaptation optimizers
pip install lion-pytorch        # Lion
pip install schedulefree        # Schedule-Free optimizers
pip install came-pytorch        # CAME
pip install adv_optm            # Advanced variants (Muon, ADOPT, etc.)
pip install muon                # Muon (upstream)
pip install timm                # AdaBelief
pip install pytorch-optimizer   # Tiger, Aida, ADOPT, Yogi
pip install transformers        # Adafactor
pip install pyyaml              # YAML config support
```

### Check It Works

```bash
python -c "import serenity; print('OK')"
```

### Run Training

```bash
python -m serenity.cli.commands train my_config.yaml
```

---

## 2. Configuration

Serenity loads config from YAML or JSON files into a `TrainConfig` dataclass. There are 150+ fields, but most have sensible defaults -- you only need to set what you care about. Enum values are case-insensitive (`"ADAMW"` and `"adamw"` both work).

### Minimal YAML Example

```yaml
model_type: sdxl
training_method: lora
transformer_path: "stabilityai/stable-diffusion-xl-base-1.0"
output_dir: "output/"
concepts:
  - name: "subject"
    path: "data/subject"
    prompt:
      source: "txt"

learning_rate: 1.0e-4
epochs: 10
batch_size: 1
resolution: "1024"
train_dtype: BFLOAT_16

lora_rank: 16
lora_alpha: 1.0

output_model_destination: "output/my_lora.safetensors"
output_model_format: SAFETENSORS

optimizer:
  optimizer: ADAMW
  weight_decay: 0.01
```

### Minimal JSON Example

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

### Required Fields

| Field | Type | What It Is |
|-------|------|------------|
| `model_type` | string | Which model (`sdxl`, `flux_dev`, `sd15`, `ltx2`, etc.) |
| `training_method` | string | `lora`, `fine_tune`, `fine_tune_vae`, or `embedding` |
| `transformer_path` | string | Path or HuggingFace ID for base model weights |
| `output_dir` | string | Where to put training outputs |
| `concepts` | list | Your training data (see Data Preparation) |

### Training Settings

| Field | Default | What It Does |
|-------|---------|--------------|
| `learning_rate` | `1e-4` | Learning rate |
| `epochs` | `100` | Training epochs |
| `max_train_steps` | `null` | Hard step limit (overrides epochs if set) |
| `batch_size` | `1` | Per-GPU batch size |
| `gradient_accumulation_steps` | `1` | Steps before optimizer update |
| `seed` | `42` | Random seed |
| `resolution` | `"512"` | Training resolution (e.g. `"1024"` or `"512x768"`) |
| `frames` | `"25"` | Frame count for video models |

### Precision

| Field | Default | What It Does |
|-------|---------|--------------|
| `train_dtype` | `FLOAT_16` | Training precision: `FLOAT_16`, `BFLOAT_16`, `FLOAT_32`, `TFLOAT_32` |
| `fallback_train_dtype` | `BFLOAT_16` | Fallback if GPU doesn't support primary dtype |
| `output_dtype` | `FLOAT_32` | Precision for saved weights |

### Per-Component Configs

Each model component (transformer, text encoders, VAE) has its own sub-config:

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

Fields per component: `model_name`, `train`, `stop_training_after`, `stop_training_after_unit`, `learning_rate`, `weight_dtype`, `dropout_probability`.

### Config Migration

Old config files are auto-upgraded to the current schema when loaded. You shouldn't need to manually update configs after version bumps.

---

## 3. Model Types

Set `model_type` in your config. Aliases are accepted for most models (case-insensitive).

### Stable Diffusion 1.5

```yaml
model_type: sd15              # or sd15_inpainting
```

The original SD architecture. Small (~860 MB), fast to train, low VRAM. Good starting point. Supports LoRA, fine-tune, embedding, and inpainting.

### SDXL 1.0

```yaml
model_type: sdxl              # or sdxl_10_base, sdxl_inpainting
```

Dual text encoder (CLIP + OpenCLIP), higher default resolution (1024px). Better quality, needs more VRAM.

### Stable Diffusion 3 / 3.5

```yaml
model_type: sd3               # or sd35
```

Flow-matching architecture with MMDiT. Up to three text encoders.

### Flux 1

```yaml
model_type: flux_dev          # or flux_schnell, flux_fill_dev
```

Black Forest Labs flow-matching models. Dev is guidance-distilled, Schnell is few-step, Fill is inpainting.

### Flux 2

```yaml
model_type: flux_2            # or flux_2_klein_4b, flux_2_klein_9b
```

Next-gen Flux. Klein variants are smaller models that use less VRAM.

### Chroma

```yaml
model_type: chroma_1
```

Less tested than the SD/Flux models.

### Z-Image

```yaml
model_type: zimage            # aliases: z_image, z-image
```

### LTX2 (Video)

```yaml
model_type: ltx2              # aliases: ltx, ltx_video, ltxvideo
```

Video model. Set `frames` to control output frame count.

### HunyuanVideo

```yaml
model_type: hunyuan_video
```

Video model. Less tested.

### Qwen

```yaml
model_type: qwen              # or qwen_image_edit for editing mode
```

Image editing variant takes source images for modification.

---

## 4. Training Methods

### LoRA

```yaml
training_method: lora
lora_rank: 16
lora_alpha: 1.0
lora_weight_dtype: FLOAT_32
```

Trains small adapter matrices, keeps base model frozen. Low VRAM, small output files (typically 1-100 MB). This is what most people want.

- `lora_rank`: Higher = more capacity, more VRAM. 4, 8, 16, 32, 64 are common.
- `lora_alpha`: Scaling factor. Usually 1.0 or equal to rank.
- `layer_filter`: Restrict LoRA to specific layers (string pattern, or regex with `layer_filter_regex: true`).

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

Advanced adapter types beyond standard LoRA. Requires the lycoris library.

- **LoKR**: Kronecker product decomposition. Set `peft_type: LOKR`.
- **Tucker**: Additional compression. `lokr_use_tucker: true`.
- **Weight decompose (DoRA)**: Direction-magnitude separation. `lokr_weight_decompose: true`.
- **Full matrix**: Full-rank updates without decomposition. `lokr_full_matrix: true`.
- **RS-LoRA**: Rank-stabilized scaling. `lokr_rs_lora: true`.

### Embedding / Textual Inversion

```yaml
training_method: embedding
embedding_learning_rate: 5.0e-4
preserve_embedding_norm: false
embedding_weight_dtype: FLOAT_32
```

Trains new token embeddings while keeping model weights frozen. Output is a small embedding file. Only works with SD 1.5 and SDXL.

### Full Fine-Tune

```yaml
training_method: fine_tune
```

Updates all model weights. Best results but needs a lot of VRAM (24 GB+). Output is a full model checkpoint. Use gradient checkpointing and bf16 to keep memory in check.

### VAE Fine-Tune

```yaml
training_method: fine_tune_vae
```

Trains just the VAE. Niche use case.

---

## 5. Data Preparation

### Folder Structure

```
training_data/
├── concept_a/
│   ├── image_001.png
│   ├── image_001.txt        # Caption (same base name)
│   ├── image_001-masklabel.png  # Optional mask
│   ├── image_002.jpg
│   ├── image_002.txt
│   └── ...
```

### Concepts Config

```yaml
concepts:
  - name: "my_character"
    path: "training_data/concept_a"
    prompt:
      source: "txt"
    # seed: 42
    # include_subdirectories: true
    # balancing: 1.0
    # loss_weight: 1.0
```

### Captions

- `source: "txt"` -- reads `.txt` file next to each image
- `source: "filename"` -- uses the filename as the caption
- `source: "fixed"` with `text: "a photo of XYZ"` -- same caption for all images

### Masks (Optional)

Place mask PNGs next to images with the pattern `{image_name}-masklabel.png`. White = region of interest, black = ignore.

```yaml
masked_training: true
unmasked_probability: 0.1
unmasked_weight: 0.1
normalize_masked_area_loss: false
```

### Image Formats

PNG, JPG, JPEG, WEBP, BMP. Images are auto-resized and bucketed by aspect ratio. Provide images at or above your target resolution.

### Video Data

For LTX2/HunyuanVideo, put video files in the concept directory. Frames are extracted based on the `frames` config field.

### Latent Caching

```yaml
latent_caching: true
clear_cache_before_training: true
cache_dir: "workspace-cache/run"
```

Caches VAE-encoded latents and text encoder outputs to disk. First epoch is slow (encoding), subsequent epochs skip encoding entirely. Worth enabling for multi-epoch runs.

---

## 6. Optimizers

```yaml
optimizer:
  optimizer: ADAMW
  weight_decay: 0.01
  eps: 1.0e-8
```

### Standard

| Optimizer | Value | Notes |
|-----------|-------|-------|
| AdamW | `ADAMW` | Default, works for most things |
| Adam | `ADAM` | No decoupled weight decay |
| SGD | `SGD` | Simple |

### 8-bit (bitsandbytes)

Halves optimizer state memory. Requires `bitsandbytes`.

`ADAMW_8BIT`, `ADAM_8BIT`, `SGD_8BIT`, `LION_8BIT`, `LAMB_8BIT`, `LARS_8BIT`, `RMSPROP_8BIT`, `ADAGRAD_8BIT`, `ADEMAMIX_8BIT`, `CAME_8BIT`

### Adaptive LR

These find their own learning rate. Set `learning_rate: 1.0` for Prodigy.

| Optimizer | Value |
|-----------|-------|
| Prodigy | `PRODIGY` |
| Prodigy+ ScheduleFree | `PRODIGY_PLUS_SCHEDULE_FREE` |
| D-Adapt Adam | `DADAPT_ADAM` |
| D-Adapt SGD | `DADAPT_SGD` |
| D-Adapt Adan | `DADAPT_ADAN` |
| D-Adapt AdaGrad | `DADAPT_ADA_GRAD` |
| D-Adapt Lion | `DADAPT_LION` |

### Schedule-Free

No LR scheduler needed. Use `learning_rate_scheduler: CONSTANT`.

`SCHEDULE_FREE_ADAMW`, `SCHEDULE_FREE_SGD`

### Advanced

`LION`, `LAMB`, `LARS`, `ADEMAMIX`, `CAME`, `ADAFACTOR`, `MUON`

### Research

`ADABELIEF`, `TIGER`, `AIDA`, `ADOPT`, `YOGI`

### Advanced Variants (adv_optm)

These add bf16 stochastic rounding support:

`ADAMW_ADV`, `ADOPT_ADV`, `PRODIGY_ADV`, `LION_ADV`, `LION_PRODIGY_ADV`, `SIMPLIFIED_ADEMAMIX`, `SIGNSGD_ADV`, `MUON_ADV`, `ADAMUON_ADV`

### Fused Back-Pass

Combines backward pass and optimizer step. Saves memory.

```yaml
optimizer:
  optimizer: ADAFACTOR
  fused_back_pass: true
```

Works with: Adafactor, CAME, CAME 8-bit, Adam, AdamW, and all `_ADV` variants.

---

## 7. Learning Rate Schedulers

```yaml
learning_rate_scheduler: COSINE
learning_rate_warmup_steps: 200
learning_rate_cycles: 1.0
learning_rate_min_factor: 0.0
```

| Scheduler | Value | Notes |
|-----------|-------|-------|
| Constant | `CONSTANT` | Fixed LR after warmup |
| Linear | `LINEAR` | Linear decay |
| Cosine | `COSINE` | Smooth annealing. Most common. |
| Cosine with Restarts | `COSINE_WITH_RESTARTS` | Periodic warm restarts |
| Cosine Hard Restarts | `COSINE_WITH_HARD_RESTARTS` | Abrupt restarts |
| REX | `REX` | Reciprocal exponential |
| Adafactor | `ADAFACTOR` | Only with Adafactor optimizer |
| Custom | `CUSTOM` | User-defined callable |

**Quick guidance:**
- LoRA (short runs): `CONSTANT` or `COSINE`
- Full fine-tune: `COSINE` or `COSINE_WITH_RESTARTS`
- Adaptive optimizers (Prodigy, D-Adapt): `CONSTANT`
- Schedule-free optimizers: `CONSTANT`

---

## 8. Loss Functions

```yaml
mse_strength: 1.0
mae_strength: 0.0
huber_strength: 0.0
huber_delta: 1.0
log_cosh_strength: 0.0
vb_loss_strength: 1.0
```

| Function | Field | When to Use |
|----------|-------|-------------|
| MSE (L2) | `mse_strength` | Default. Stable. |
| MAE (L1) | `mae_strength` | Sharper outputs, less smoothing |
| Huber | `huber_strength` | Noisy datasets, outlier-robust |
| Log-Cosh | `log_cosh_strength` | Like Huber but smoother |
| VB | `vb_loss_strength` | Variational bound |

You can blend them: `mse_strength: 0.8` + `mae_strength: 0.2` gives 80/20 mix.

### Loss Weighting

```yaml
loss_weight_fn: MIN_SNR_GAMMA
loss_weight_strength: 5.0
```

Options: `CONSTANT`, `MIN_SNR_GAMMA`, `P2`, `DEBIASED_ESTIMATION`, `SIGMA`

### Loss Scaling

```yaml
loss_scaler: NONE
```

Options: `NONE`, `BATCH`, `GLOBAL_BATCH`, `GRADIENT_ACCUMULATION`, `BOTH`, `GLOBAL_BOTH`

Use `BOTH` or `GLOBAL_BOTH` with large effective batch sizes to keep gradients stable.

---

## 9. Memory Optimization

### By VRAM Tier

#### 8 GB

```yaml
training_method: lora
lora_rank: 8
batch_size: 1
gradient_accumulation_steps: 4
gradient_checkpointing: on
train_dtype: BFLOAT_16
latent_caching: true

optimizer:
  optimizer: ADAMW_8BIT

transformer:
  weight_dtype: NFLOAT_4

layer_offload_fraction: 0.5
enable_async_offloading: true
```

This is tight. You'll probably need NF4 quantization and layer offloading for anything bigger than SD 1.5.

#### 16 GB

```yaml
training_method: lora
lora_rank: 16
batch_size: 1
gradient_accumulation_steps: 2
gradient_checkpointing: on
train_dtype: BFLOAT_16
latent_caching: true

optimizer:
  optimizer: ADAMW_8BIT
```

Comfortable for most LoRA training. Might need quantization for Flux 2 9B.

#### 24 GB+

```yaml
training_method: lora       # or fine_tune
lora_rank: 32
batch_size: 2
gradient_checkpointing: on
train_dtype: BFLOAT_16
latent_caching: true

optimizer:
  optimizer: ADAMW
```

### Techniques

**Gradient checkpointing** -- recomputes activations during backward pass instead of storing them. Saves 30-50% VRAM, costs ~20% speed.

```yaml
gradient_checkpointing: on
# gradient_checkpointing: cpu_offloaded   # Even more savings, slower
```

**Layer offloading** -- moves inactive layers to CPU. 0.0 = none, 1.0 = all.

```yaml
layer_offload_fraction: 0.5
enable_async_offloading: true
enable_activation_offloading: true
```

**Quantization** -- shrinks frozen base model weights.

```yaml
transformer:
  weight_dtype: NFLOAT_4    # 4-bit, biggest savings
  # weight_dtype: INT_8     # 8-bit
  # weight_dtype: FLOAT_8   # FP8
```

Trainable params (LoRA matrices, embeddings) stay full precision.

**8-bit optimizers** -- halves optimizer state memory. Just swap `ADAMW` for `ADAMW_8BIT`.

**Latent caching** -- caches encoded latents to disk, removes VAE from training loop after first pass.

**Gradient accumulation** -- simulates larger batch without extra VRAM.

```yaml
batch_size: 1
gradient_accumulation_steps: 8   # Effective batch = 8
```

---

## 10. Sampling

Generate sample images/videos during training to see how things are going.

```yaml
sample_after: 10
sample_after_unit: MINUTE       # EPOCH, STEP, MINUTE, HOUR, SECOND, ALWAYS, NEVER
sample_skip_first: 0
sample_image_format: JPG
samples_to_tensorboard: true
non_ema_sampling: true
```

### Sample Prompts

```yaml
samples:
  - prompt: "a photo of my_subject in a garden"
    negative_prompt: "blurry, low quality"
    width: 1024
    height: 1024
    steps: 20
    seed: 42
```

### Noise Schedulers

`DDIM`, `EULER`, `EULER_A`, `DPMPP`, `DPMPP_SDE`, `UNIPC`, `EULER_KARRAS`, `DPMPP_KARRAS`, `DPMPP_SDE_KARRAS`, `UNIPC_KARRAS`

---

## 11. Multi-GPU Training

DDP (DistributedDataParallel) support. Each GPU processes its own batch, gradients are synced.

```yaml
multi_gpu: true
device_indexes: "0,1"
fused_gradient_reduce: true
async_gradient_reduce: true
async_gradient_reduce_buffer: 100
```

Effective batch size = `batch_size * num_gpus * gradient_accumulation_steps`. You may want to scale LR accordingly:

```yaml
loss_scaler: GLOBAL_BATCH
learning_rate_scaler: GLOBAL_BATCH
```

---

## 12. Presets

Pre-built configs for common scenarios live in `serenity/presets/`. Use one as-is or as a starting point:

```bash
python -m serenity.cli.commands train serenity/presets/sdxl_lora_16gb.json
```

To make your own, create a JSON file with whatever subset of `TrainConfig` fields you want. Unspecified fields use defaults.

---

## 13. Troubleshooting

### Out of Memory

Try these in order:
1. `gradient_checkpointing: on`
2. `batch_size: 1` with higher `gradient_accumulation_steps`
3. 8-bit optimizer (`ADAMW_8BIT`)
4. Quantize base model (`transformer: { weight_dtype: NFLOAT_4 }`)
5. Lower `lora_rank`
6. `layer_offload_fraction: 0.5`
7. Lower `resolution`

### Loss Not Decreasing

- LR too low or too high (try 2-5x in either direction)
- Bad captions (verify they match image content)
- Too few images
- Wrong `model_type` for your base model

### NaN/Inf Loss

- Switch to `BFLOAT_16` (more stable than fp16)
- Add gradient clipping: `clip_grad_norm: 1.0`
- Lower LR
- Check for corrupted images

### Slow Training

- Enable `latent_caching: true`
- Increase `dataloader_threads` (default 2, try 4-8)
- Gradient checkpointing adds ~20% overhead -- that's normal
- High `layer_offload_fraction` is slow by design (CPU-GPU transfers)

### Config Errors

- Check spelling on `model_type` (lowercase, use aliases like `sdxl`, `flux_dev`)
- Valid training methods: `lora`, `fine_tune`, `fine_tune_vae`, `embedding`
- Required fields: `model_type`, `training_method`, `transformer_path`, `output_dir`, `concepts`

### Resume Issues

- Set `continue_last_backup: true`
- Make sure `workspace_dir` points to the original workspace

### Multi-GPU Issues

- Check `device_indexes` against `nvidia-smi`
- Rank 0 using more memory than others is normal
- If it hangs during gradient sync, try `async_gradient_reduce: false`

---

*Serenity pre-alpha 0.055*
