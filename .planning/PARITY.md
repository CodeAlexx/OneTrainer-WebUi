# Serenity ↔ OneTrainer Feature Parity Master Checklist

**Generated**: 2026-02-06
**Lead**: Parity Coordinator (Claude Opus 4.6)
**Sources**: PIPELINE_GAPS.md, MODEL_GAPS.md, CONFIG_DATA_GAPS.md

---

## Executive Summary

Serenity has strong foundations in **memory management** (gradient checkpointing, layer offloading, block swap), **adapter support** (LyCORIS with 12 adapter types), **sampling** (diffusers-backed, 18+ model types), and **model loading** (pipeline-based with HF cache discovery). It also has unique features OneTrainer lacks: LTX2 video, Flux 2 Klein 4B/9B, CREPA regularizer, and assistant LoRA loading.

However, Serenity is missing the **entire training pipeline** (training loop, optimizers, LR schedulers), the **entire data pipeline** (image/text loading, bucketing, caching, augmentation), the **configuration system** (180+ config fields → 12), **model saving/export**, and numerous training quality features (loss functions, noise scheduling, EMA integration, mixed precision, multi-GPU).

### Gap Counts by Priority

| Priority | Count | Description |
|----------|-------|-------------|
| CRITICAL | 14 | Cannot train without these |
| HIGH | 22 | Needed for production-quality training |
| MEDIUM | 20 | Feature completeness |
| LOW | 15 | Nice-to-have / legacy |
| PARITY | 8 | Already equivalent or better |

### Areas at Parity (No Action Needed)

- [x] Gradient checkpointing (standard + CPU-offloaded, per-model helpers)
- [x] LyCORIS adapter management (12 adapter types, save/load/merge)
- [x] Sampling / inference (diffusers pipelines, 18+ model types)
- [x] Layer offloading conductor (basic version functional)
- [x] Model type enum (30+ types, superset of OneTrainer)
- [x] Training method enum (lora, fine_tune, fine_tune_vae, embedding)
- [x] OneTrainer config compatibility (CLI bridge translation)
- [x] YAML config loading

### Serenity-Only Features (Preserve)

- LTX2 video model (full training + sampling)
- Flux 2 Klein 4B/9B (most detailed model impl, 1526 lines)
- CREPA regularizer (video frame alignment)
- BlockSwapOffloader (simpler layer offloading)
- BitsAndBytes/Quanto quantization (Qwen INT8/FP8)
- Advanced HF cache discovery (case-insensitive, env vars)
- Assistant LoRA auto-loading (ZImage sampler)
- Pipeline caching in sampler

---

## Implementation Phases

### Phase 1: Training Foundation (CRITICAL)
*Goal: Enable a complete training run end-to-end*

| # | Item | Domain | Gap Report | Status |
|---|------|--------|-----------|--------|
| 1.1 | **Full TrainConfig** - Expand from 12 fields to ~180+ matching OneTrainer | Config | CONFIG_DATA | [x] b07edb44 |
| 1.2 | **Model part configs** - `TrainModelPartConfig` for per-component dtype, train flags, LR | Config | CONFIG_DATA | [x] b07edb44 |
| 1.3 | **Optimizer config** - `TrainOptimizerConfig` with full param support | Config | CONFIG_DATA | [x] b07edb44 |
| 1.4 | **Training loop** - Epoch/step orchestration, gradient accumulation, backward, optimizer step | Pipeline | PIPELINE | [x] dab88752 |
| 1.5 | **Optimizer factory** - Start with AdamW, AdamW-8bit, Prodigy, Adafactor, Lion | Pipeline | PIPELINE | [x] 7fa4da25 |
| 1.6 | **LR scheduler system** - Constant, Linear, Cosine with warmup | Pipeline | PIPELINE | [x] 89653d0e |
| 1.7 | **Data pipeline** - Image loading, text/caption loading (replace stubs) | Config/Data | CONFIG_DATA | [x] 4e037483 |
| 1.8 | **Aspect ratio bucketing** - Real bucketing with quantization (replace stub) | Config/Data | CONFIG_DATA | [x] 965e3f57 |
| 1.9 | **Latent caching** - Disk-based latent + text embedding cache | Config/Data | CONFIG_DATA | [x] 71f02ae6 |
| 1.10 | **Concept config** - Full `ConceptConfig` with image/text sub-configs | Config/Data | CONFIG_DATA | [x] 0f351c5f |
| 1.11 | **Dataset class** - Real `Dataset` with pipeline integration (replace stub) | Config/Data | CONFIG_DATA | [x] 4e037483 |
| 1.12 | **Model saving/export** - LoRA safetensors export, full model diffusers export | Model | MODEL | [x] 29127291 |
| 1.13 | **Caption loading** - Per-image caption files, basic tag support | Config/Data | CONFIG_DATA | [x] afafb048 |
| 1.14 | **Essential enums** - Optimizer (40+), ModelFormat, LossWeight, LossScaler, DataType | Config/Data | CONFIG_DATA | [x] 80ea11d4 |

### Phase 2: Training Quality (HIGH)
*Goal: Training results match OneTrainer quality*

| # | Item | Domain | Gap Report | Status |
|---|------|--------|-----------|--------|
| 2.1 | **Loss function system** - MSE, MAE, Huber, log-cosh, VB loss with type selection | Pipeline | PIPELINE + MODEL | [ ] |
| 2.2 | **Loss weighting** - MIN_SNR_GAMMA, P2, DEBIASED_ESTIMATION, SIGMA, CONSTANT | Pipeline | PIPELINE | [ ] |
| 2.3 | **Loss scaling** - Batch, gradient accumulation, global batch scaling | Pipeline | PIPELINE | [ ] |
| 2.4 | **Masked loss** - Spatial mask support, prior preservation | Pipeline + Model | PIPELINE + MODEL | [ ] |
| 2.5 | **Noise creation** - Offset noise, perturbation noise, generalized offset | Pipeline | PIPELINE | [ ] |
| 2.6 | **Timestep distributions** - Full set (uniform, sigmoid, logit_normal, heavy_tail, cos_map, inverted_parabola) | Pipeline | PIPELINE | [ ] |
| 2.7 | **EMA improvements** - Warmup decay, cross-device CPU/GPU, temp_store/restore for sampling | Pipeline | PIPELINE + MODEL | [ ] |
| 2.8 | **Mixed precision** - Autocast context manager, gradient scaler, fp16/bf16 handling | Pipeline | PIPELINE | [ ] |
| 2.9 | **Multi-GPU / DDP** - Gradient reduction, rank management, sync commands | Pipeline | PIPELINE | [ ] |
| 2.10 | **Diffusion schedule coefficients** - Pre-computed alphas_cumprod etc. for SD1.5/SDXL | Pipeline | PIPELINE | [ ] |
| 2.11 | **Named parameter groups** - Per-component learning rates (text encoder, transformer, etc.) | Model | MODEL | [ ] |
| 2.12 | **Fine-tune setup classes** - Per-model parameter group configuration | Model | MODEL | [ ] |
| 2.13 | **Training resume** - Save/load optimizer state, EMA state, progress to disk | Model + Pipeline | MODEL + PIPELINE | [ ] |
| 2.14 | **Backup system** - Rolling backups with configurable intervals | Config/Data | CONFIG_DATA | [ ] |
| 2.15 | **Augmentation pipeline** - Random flip, crop jitter, scale/crop | Config/Data | CONFIG_DATA | [ ] |
| 2.16 | **Caption handling** - Tag shuffling, tag dropout, prompt source selection | Config/Data | CONFIG_DATA | [ ] |
| 2.17 | **Remaining optimizers** - All 40+ from OneTrainer (CAME, Muon, ADOPT, schedule-free, etc.) | Pipeline | PIPELINE | [ ] |
| 2.18 | **LR scheduler completions** - Cosine with restarts, hard restarts, REX, Adafactor, custom | Pipeline | PIPELINE | [ ] |
| 2.19 | **Config versioning** - Migration pipeline (10 versions) | Config/Data | CONFIG_DATA | [ ] |
| 2.20 | **Model weight dtypes** - `ModelWeightDtypes` for per-component dtype management | Config/Data | CONFIG_DATA | [ ] |
| 2.21 | **Quantization utilities** - General framework (NF4, GGUF, SVD quantized layers) | Model | MODEL | [ ] |
| 2.22 | **Stochastic rounding** - bf16 copy/add/addcdiv with stochastic rounding | Pipeline | PIPELINE | [ ] |

### Phase 3: Feature Completeness (MEDIUM)
*Goal: Full OneTrainer feature coverage*

| # | Item | Domain | Gap Report | Status |
|---|------|--------|-----------|--------|
| 3.1 | **TensorBoard logging** - Loss curves, LR, EMA decay, sample images | Pipeline | PIPELINE | [ ] |
| 3.2 | **Validation loop** - Between-epoch validation with separate dataset | Pipeline | PIPELINE | [ ] |
| 3.3 | **Chroma model** - T5 + ChromaTransformer, LoRA support | Model | MODEL | [ ] |
| 3.4 | **HunyuanVideo model** - Llama+CLIP, 3D Transformer, 3D VAE | Model | MODEL | [ ] |
| 3.5 | **Text encoder training** - LoRA on text encoders, separate LRs | Model | MODEL | [ ] |
| 3.6 | **VAE training** - Train VAE encoder/decoder (SD15, SDXL) | Model | MODEL | [ ] |
| 3.7 | **Embedding/TI training** - Full textual inversion system | Model | MODEL | [ ] |
| 3.8 | **Inpainting training** - Masked loss, conditioning images, prior preservation | Model | MODEL | [ ] |
| 3.9 | **Layer offload improvements** - Static allocators, pinned memory, SyncEvents | Pipeline | PIPELINE | [ ] |
| 3.10 | **Custom gradient scaler** - Fused backward pass support | Pipeline | PIPELINE | [ ] |
| 3.11 | **Video data loading** - Frame extraction, video formats, frame count config | Config/Data | CONFIG_DATA | [ ] |
| 3.12 | **Masking pipeline** - Mask loading, generation, augmentation | Config/Data | CONFIG_DATA | [ ] |
| 3.13 | **Color augmentations** - Brightness, contrast, saturation, hue, rotation | Config/Data | CONFIG_DATA | [ ] |
| 3.14 | **ModelType helper methods** - `is_flux()`, `is_flow_matching()`, etc. | Config/Data | CONFIG_DATA | [ ] |
| 3.15 | **Training callbacks** - `TrainCallbacks` system for start/stop/sample events | Config/Data | CONFIG_DATA | [ ] |
| 3.16 | **Model format/conversion** - Diffusers, safetensors, legacy format, LoRA conversion | Model | MODEL | [ ] |
| 3.17 | **Sample config** - `SampleConfig` with intervals, formats, tensorboard integration | Config/Data | CONFIG_DATA | [ ] |
| 3.18 | **Path utilities** - Supported extensions, safe filenames, atomic write | Config/Data | CONFIG_DATA | [ ] |
| 3.19 | **Preset organization** - Structured presets with VRAM tiers | Config/Data | CONFIG_DATA | [ ] |
| 3.20 | **Token pruning** - Text encoder token pruning (Chroma, Qwen) | Model | MODEL | [ ] |

### Phase 4: Polish & Legacy (LOW)
*Goal: Complete feature parity including legacy models*

| # | Item | Domain | Gap Report | Status |
|---|------|--------|-----------|--------|
| 4.1 | **SD 2.0/2.1 models** - v-prediction, OpenCLIP | Model | MODEL | [ ] |
| 4.2 | **PixArt Alpha/Sigma models** | Model | MODEL | [ ] |
| 4.3 | **Sana model** - Gemma encoder, DC-VAE | Model | MODEL | [ ] |
| 4.4 | **HiDream model** - 4 text encoders | Model | MODEL | [ ] |
| 4.5 | **Wuerstchen/Stable Cascade** | Model | MODEL | [ ] |
| 4.6 | **Torch compile utilities** - sympy Mod patch | Pipeline | PIPELINE | [ ] |
| 4.7 | **Model spec** - Safetensors metadata headers | Model | MODEL | [ ] |
| 4.8 | **LoRA format conversion** - Legacy ↔ OMI format | Model | MODEL | [ ] |
| 4.9 | **Cloud integration** - RunPod, Linux cloud | Config/Data | CONFIG_DATA | [ ] |
| 4.10 | **Captioning CLI** - BLIP, BLIP2, WD14 auto-captioning | Config/Data | CONFIG_DATA | [ ] |
| 4.11 | **Mask generation CLI** - CLIPSEG, REMBG auto-masking | Config/Data | CONFIG_DATA | [ ] |
| 4.12 | **Config packing** - `to_pack_dict()`, settings dict | Config/Data | CONFIG_DATA | [ ] |
| 4.13 | **Secrets management** - HuggingFace token, cloud secrets | Config/Data | CONFIG_DATA | [ ] |
| 4.14 | **Profiling utilities** - Training profiling | Config/Data | CONFIG_DATA | [ ] |
| 4.15 | **Triton 8-bit matmul** - Custom kernel | Pipeline | PIPELINE | [ ] |

---

## Architectural Decisions

### Keep Serenity's Approach (Don't Copy OneTrainer)

1. **LyCORIS for adapters** - Serenity uses external LyCORIS (12 types) vs OneTrainer's custom PEFT (4 types). Keep LyCORIS - more adapter variety, less code to maintain.

2. **Diffusers-based sampling** - Serenity uses diffusers pipelines vs OneTrainer's custom samplers. Keep diffusers - more maintainable, auto-updated with new schedulers.

3. **Per-model `load_pipeline()` methods** - Serenity's approach is simpler than OneTrainer's 75-file mixin loader system. Keep Serenity's approach but add internal checkpoint format support.

4. **HF cache discovery** - Serenity's case-insensitive, env-var-aware discovery is better. Keep it.

### Adopt OneTrainer's Approach

1. **Training loop** - OneTrainer's `GenericTrainer` pattern (epoch/step loop with gradient accumulation, NaN detection, sampling callbacks). Build equivalent in Serenity style.

2. **Config system** - OneTrainer's rich typed config is essential. Build Serenity-style dataclass equivalent with all fields.

3. **Data pipeline** - OneTrainer delegates to MGDS. Serenity should either use MGDS or build equivalent. MGDS dependency is acceptable since OneTrainer already requires it.

4. **Optimizer factory** - OneTrainer's `create.py` pattern. Port to Serenity as `training/optimizers.py`.

5. **Loss system** - OneTrainer's mixin-based loss computation. Port as `training/losses.py` expansion.

6. **Model saving** - OneTrainer's mixin-based savers. Build Serenity-style equivalent for LoRA + full model export.

### Design Decisions Needed

1. **MGDS vs custom data pipeline** - MGDS is already a dependency. Using it saves massive effort but adds coupling.

2. **Internal checkpoint format** - Use OneTrainer's `meta.json` format for interop? Or design Serenity's own?

3. **Config migration** - Build a migration system or just version-bump and require re-creation?

4. **Per-model setup classes** - OneTrainer has ~60 setup files. Serenity could use a more generic approach with config-driven behavior.

---

## File Reference: Key OneTrainer Files to Port

### Phase 1 Priority Files
| OneTrainer File | Lines | Purpose | Serenity Target |
|---|---|---|---|
| `util/create.py` | 1444 | Optimizer/scheduler/EMA factory | `training/optimizers.py` + `training/schedulers.py` |
| `trainer/GenericTrainer.py` | 883 | Main training loop | `core/trainer.py` expansion |
| `util/lr_scheduler_util.py` | 105 | LR lambda functions | `training/schedulers.py` |
| `util/config/TrainConfig.py` | ~500 | Full config | `core/config.py` expansion |

### Phase 2 Priority Files
| OneTrainer File | Lines | Purpose | Serenity Target |
|---|---|---|---|
| `modelSetup/mixin/ModelSetupDiffusionLossMixin.py` | 344 | Loss functions + weighting | `training/losses.py` expansion |
| `modelSetup/mixin/ModelSetupNoiseMixin.py` | 239 | Noise creation + timestep distributions | `training/noise.py` (new) |
| `util/multi_gpu_util.py` | 174 | DDP utilities | `training/distributed.py` (new) |
| `util/dtype_util.py` | 96 | Mixed precision | `training/precision.py` (new) |
| `module/EMAModule.py` | 87 | EMA with warmup | `training/ema.py` expansion |
| `util/bf16_stochastic_rounding.py` | 74 | Stochastic rounding | `training/stochastic_rounding.py` (new) |
| `util/loss/vb_loss.py` | 213 | Variational bound loss | `training/vb_loss.py` (new) |
| `util/loss/masked_loss.py` | 46 | Masked losses | `training/losses.py` expansion |

### Phase 3 Priority Files
| OneTrainer File | Lines | Purpose | Serenity Target |
|---|---|---|---|
| `util/DiffusionScheduleCoefficients.py` | 61 | Schedule coefficients | `training/diffusion.py` (new) |
| `util/CustomGradScaler.py` | 65 | Fused backward scaler | `training/grad_scaler.py` (new) |
| Model saver mixins | ~80 files | Model export | `checkpoint/savers.py` (new) |
| Model loader mixins | ~75 files | Internal format loading | `checkpoint/loaders.py` (new) |

---

## Progress Tracking

### Overall Progress
- Phase 1: 14/14 items complete ✅
- Phase 2: 0/22 items complete
- Phase 3: 0/20 items complete
- Phase 4: 0/15 items complete (OUT OF SCOPE except 4.3, 4.6, 4.7, 4.8, 4.12, 4.13, 4.14)
- **Total: 14/71 items (20%)**

### Next Steps
1. Review this checklist and confirm priorities
2. Start Phase 1 implementation (training foundation)
3. Begin with 1.1 (TrainConfig), 1.4 (training loop), 1.7 (data pipeline) in parallel
4. Each completed item gets checked off and committed

---

*This document is the single source of truth for Serenity parity work. All implementation should reference items by their ID (e.g., "1.4 Training Loop"). Update status as work progresses.*
