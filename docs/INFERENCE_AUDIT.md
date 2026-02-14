# Serenity Inference Engine Audit

**Date**: 2026-02-10
**Auditor**: Claude Opus 4.6 (automated)
**Scope**: All inference-related code across `serenity/inference/`, `/home/alex/serenity-inference/`, TUI, CLI, API
**Method**: Read-only analysis of all source files. No code was executed. All claims backed by direct file contents.

---

## A. Architecture Overview

### Two Separate Inference Codebases

There are **two completely separate inference engines**:

| Property | Trainer-Integrated | Standalone |
|----------|-------------------|------------|
| **Location** | `serenity/inference/` | `/home/alex/serenity-inference/` |
| **Python files** | 64 | 92 |
| **Diffusers-free** | NO (11 import sites in 7 files) | YES (100% pure PyTorch) |
| **Status** | Legacy/trainer-coupled | The REAL inference engine |
| **Git tracked** | Yes (in serenity repo) | No git repo |
| **Used by** | TUI inference tab, Web API | Generation scripts only |

### Standalone Engine Architecture (`/home/alex/serenity-inference/`)

The standalone engine follows a ComfyUI/Forge-style pipeline:

```
detect_from_file(path)          # Read safetensors header only — no weight loading
    → split_checkpoint(sd)      # Separate model/vae/text_encoder keys by prefix
    → create model from arch    # Instantiate correct DiT/UNet from architecture
    → load state dict           # Load weights into model
    → text encoder encode       # CLIP/T5/Qwen3/Mistral encode prompt
    → build_denoise_fn()        # Architecture-aware wrapper with CFG
    → sample()                  # ODE/SDE integration (euler, dpm++, etc.)
    → VAE decode                # Latents → pixels
    → save image
```

**Core pipeline**: `engine/pipeline.py` (716 lines) — orchestrates the full generate flow.

**Key design decisions**:
- Single-file checkpoint loading (`.safetensors`, `.ckpt`, `.pt`, `.bin`, `.gguf`)
- Architecture auto-detection from state dict keys (reads header only for safetensors — no full weight load needed)
- Sequential model offloading: text encoder → GPU → CPU, then DiT → GPU → CPU, then VAE → GPU → decode
- Block-level CPU offloading for large models (forward hooks stream blocks GPU↔CPU one at a time)
- No diffusers dependency. Zero. None.

### Trainer-Integrated Engine (`serenity/inference/`)

This is the older engine embedded in the trainer repo. It has the same directory structure as the standalone but is contaminated with **11 diffusers imports** across 7 files (see Section I for full list). The `DO_NOT_USE_DIFFUSERS.md` guard document exists but is violated by the code itself.

The TUI inference tab (`serenity/ui/tabs/inference.py`) and Web API (`serenity/ui/web/api.py`) both import from this version, meaning they currently depend on diffusers for inference.

### Directory Structure — Standalone Engine

```
serenity-inference/                 92 Python files, NO git
  config.py                        InferenceConfig dataclass
  engine/
    pipeline.py                    Main generate() orchestrator (716 lines)
    loader.py                      load_state_dict(), split_checkpoint()
    denoise.py                     build_denoise_fn() — arch-aware model wrapping + CFG
    conditioning.py                TextEncoderOutput, prepare_model_kwargs() per arch
  models/
    detection.py                   14 architectures auto-detected from state dict keys
    flux_dit.py                    FluxTransformer (Flux 1 Dev/Schnell, Chroma)
    flux2_dit.py                   Flux2KleinTransformer (Klein 4B/9B, Flux 2 Dev)
    unet.py                        UNet (SD 1.5, SDXL)
    sd3_dit.py                     SD3DiT (SD3 Medium/Large)
    zimage_dit.py                  ZImageDiT / NextDiT
    qwen_image_dit.py              QwenImageDiT
    wan_dit.py                     WanDiT (video, t2v/i2v)
    ltx2_dit.py                    LTX2DiT (video)
    clip_vision.py                 CLIP vision (for i2v)
    vae.py                         AutoencoderKL (standard)
    vae_flux2.py                   AutoencoderKLFlux2 (32-channel)
    vae_wan.py                     WanVAE (video, causal 3D conv)
    vae_ltx2.py                    LTX2VAE (video)
    convert.py                     Key conversion utilities
  sampling/
    sampler.py                     10 samplers (4 built-in + k-diffusion fallback)
    schedulers.py                  14 schedulers (includes flux_simple)
    prediction.py                  8 prediction types
    cfg.py                         CFG + RescaleCFG + MaHiRo + CFGNorm + hooks
    conditioning.py                create_noise()
    regions.py                     Regional prompting
  text/
    manager.py                     TextEncoderManager — per-architecture routing
    clip.py                        CLIP-L, CLIP-G
    t5.py                          T5-XXL
    qwen3.py                       Qwen3 (mode=klein: layers [9,18,27], mode=zimage: penultimate)
    mistral.py                     Mistral 3 (layers [10,20,30], dim=15360)
    qwen25vl.py                    Qwen 2.5 VL
    gemma3.py                      Gemma 3
    umt5.py                        UMT5
    tokenizer.py                   Shared tokenizer utils
  attention/
    backends.py                    select_best_backend()
    sage.py, flash.py, sdp.py      SageAttention, Flash, SDP
    xformers_attn.py               xFormers
    token_merging.py               ToMe-style token reduction
  quantization/
    int8.py, fp8.py, bnb.py        INT8, FP8, BNB NF4/FP4
    gguf.py, nunchaku.py            GGUF, Nunchaku
    ops.py                         OperationContext
  memory/
    vram.py                        VRAM detection, budget, accelerator type
    streams.py                     StreamPool, CastBuffer, async transfers
    pinned.py                      PinnedMemoryManager
    offload.py                     OffloadLinear, OffloadConv2d
    manager.py                     ModelManager, LoadedModel
    ram.py                         System RAM monitoring
  lora/
    loader.py                      load_lora(), detect format, normalize keys
    merge.py                       merge/unmerge LoRA into model weights
    online.py                      Runtime LoRA hooks (no weight mod)
    adapters.py                    LoRA, LoHA, LoKR, OFT adapters
```

---

## B. Model Support Matrix

### Architecture Detection (`models/detection.py`)

14 architectures detected from state dict key patterns:

| Architecture | Enum Value | Detection Key | Prediction | Text Encoder(s) |
|-------------|------------|---------------|------------|-----------------|
| SD 1.5 | `sd15` | `input_blocks.0.0.weight` + no ADM | eps | CLIP-L |
| SDXL | `sdxl` | `input_blocks` + `label_emb` + middle_depth>4 | eps | CLIP-L + CLIP-G |
| SDXL Refiner | `sdxl_refiner` | `input_blocks` + `label_emb` + middle_depth≤4 | eps | CLIP-G |
| SD3 | `sd3` | `joint_blocks.0.*` | flow | CLIP-L + CLIP-G + T5-XXL |
| Flux 1 Dev | `flux_dev` | `double_blocks` + `guidance_in` + no flux2 mod | flow | CLIP-L + T5-XXL |
| Flux 1 Schnell | `flux_schnell` | `double_blocks` + no `guidance_in` | flow | CLIP-L + T5-XXL |
| Flux 2 Dev | `flux_2_dev` | `double_stream_modulation_img` + 19d/38s | flow | Mistral 3 24B |
| Flux 2 Klein 4B | `flux_2_klein_4b` | `double_stream_modulation_img` + 8d/16s | flow | Qwen3 (klein) |
| Flux 2 Klein 9B | `flux_2_klein_9b` | `double_stream_modulation_img` + 16d/32s | flow | Qwen3 (klein) |
| Chroma | `chroma` | `distilled_guidance_layer.*` | flow | T5-XXL |
| Wan 2.1 | `wan` | `head.modulation` (detects t2v vs i2v) | flow | T5-XXL (UMT5) |
| Qwen Image | `qwen` | `txt_norm.weight` | flow | Qwen 2.5 VL |
| Lumina 2 | `lumina` | `cap_embedder.1.weight` + no `cap_pad_token` | flow | T5-XXL |
| ZImage | `zimage` | `cap_embedder.1.weight` + `cap_pad_token` | flow | Qwen3 (zimage) |

Detection order: lumina_zimage → wan → flux_nf4 → qwen → flux (also handles chroma) → sd3 → unet_sd.

File-size validation: soft check against expected ranges (e.g., Klein 4B = 5-10 GB, SDXL = 5-8 GB). Warns but never overrides detection.

### Generation Scripts (standalone)

| Script | Model | Resolution | Steps | CFG | Status |
|--------|-------|-----------|-------|-----|--------|
| `generate_klein4b.py` | Klein 4B | 512x512 | 35 | 3.5 | WORKING (98s, 9.3 GB) |
| `generate_flux1dev.py` | Flux 1 Dev | 1024x1024 | 20 | guidance=3.5 | WORKING (80s, 2.75 GB) |
| `generate_klein9b.py` | Klein 9B | 1024x1024 | 35 | 3.5 | WORKING (194s, 3.98 GB) |
| `edit_klein9b.py` | Klein 9B Edit | 1024x1024 | 35 | 3.5 | WORKING (280s, 5.28 GB) |
| `generate_sdxl.py` | SDXL | 1024x1024 | 20 | 4.0 | WORKING (5.0s, 6.70 GB) |
| `generate_sd3_medium.py` | SD3.5 Medium | 1024x1024 | 20 | 4.0 | WORKING (8.4s, 6.97 GB) |
| `generate_zimage.py` | ZImage Turbo | 1024x1024 | 20 | 4.0 | WORKING |
| `generate_sd35_large.py` | SD3.5 Large | — | — | — | UNTESTED (script exists) |
| `generate_qwen_image.py` | Qwen Image | — | — | — | UNTESTED (script exists) |
| `edit_qwen_image.py` | Qwen Image Edit | — | — | — | UNTESTED (script exists) |
| `generate_ltx2.py` | LTX2 | — | — | — | UNTESTED (script exists) |
| `generate_wan22_t2v.py` | Wan 2.1 t2v | — | — | — | UNTESTED (script exists) |
| `generate_wan22_i2v.py` | Wan 2.1 i2v | — | — | — | UNTESTED (script exists) |

**Missing weights** (no checkpoint files available): Chroma, Flux 2 Dev, Mistral3 24B, Lumina 2.
**Missing script**: SD 1.5 (UNet model exists, detection works, no `generate_sd15.py`).

### Trainer-Integrated Engine Model Status

| Model | Detection | Loading | Notes |
|-------|-----------|---------|-------|
| SD 1.5 | YES | YES (diffusers UNet2DConditionModel) | Functional but uses diffusers |
| SDXL | YES | YES (diffusers UNet2DConditionModel) | Functional but uses diffusers |
| SDXL Refiner | YES | YES (diffusers UNet2DConditionModel) | Functional but uses diffusers |
| SD3 | YES | YES (diffusers SD3Transformer2DModel) | Functional but uses diffusers |
| Flux 1 Dev | YES | YES (diffusers FluxTransformer2DModel) | Functional but uses diffusers |
| Flux 1 Schnell | YES | YES (diffusers FluxTransformer2DModel) | Functional but uses diffusers |
| Flux 2 Klein 4B | YES | YES (diffusers FluxTransformer2DModel) | Functional but uses diffusers |
| Flux 2 Klein 9B | YES | YES (diffusers FluxTransformer2DModel) | Functional but uses diffusers |
| Chroma | YES | YES (diffusers FluxTransformer2DModel) | Functional but uses diffusers |
| Wan | YES | YES (diffusers WanTransformer3DModel) | Functional but uses diffusers |
| Lumina 2 | YES (detection only) | NO — `create_model()` raises NotImplementedError | STUB |
| Qwen Image | YES (detection only) | NO — `create_model()` raises NotImplementedError | STUB |
| ZImage | YES (detection only) | NO — `create_model()` raises NotImplementedError | STUB |

10 functional (all use diffusers), 3 stubs.

---

## C. Sampler Inventory

10 sampler types in `SamplerType` enum (identical in both projects):

| # | Sampler | Built-in | k-diffusion | Algorithm |
|---|---------|----------|-------------|-----------|
| 1 | `euler` | YES | also available | 1st-order Euler |
| 2 | `euler_a` | YES | also available | Euler ancestral (stochastic) |
| 3 | `dpm_pp_2m` | YES | also available | DPM++ 2M (2nd-order multistep) |
| 4 | `dpm_pp_2m_sde` | YES | also available | DPM++ 2M SDE (stochastic) |
| 5 | `dpm_2m` | no | YES (required) | DPM 2M |
| 6 | `dpm_2m_sde` | no | YES (required) | DPM 2M SDE |
| 7 | `lcm` | no | YES (required) | Latent Consistency Model |
| 8 | `heun` | no | YES (required) | Heun's method (2nd-order) |
| 9 | `deis` | no | YES (required) | DEIS |
| 10 | `unipc` | no | YES (required) | UniPC |

Strategy: tries k-diffusion first; falls back to built-in for the 4 that have it. Raises `ImportError` for the other 6 if k-diffusion is missing.

---

## D. Scheduler Inventory

| # | Scheduler | Standalone | Trainer | Description |
|---|-----------|-----------|---------|-------------|
| 1 | `normal` | YES | YES | Linear spacing in sigma space |
| 2 | `karras` | YES | YES | Karras et al. noise schedule |
| 3 | `exponential` | YES | YES | Exponential spacing |
| 4 | `sgm_uniform` | YES | YES | SGM uniform timestep spacing |
| 5 | `simple` | YES | YES | Linear in timestep [1.0, 0.0] |
| 6 | `flux_simple` | YES | **NO** | Shifted simple (resolution-aware) — standalone only |
| 7 | `ddim_uniform` | YES | YES | DDIM uniform spacing |
| 8 | `beta` | YES | YES | Beta distribution schedule |
| 9 | `linear_quadratic` | YES | YES | Linear→quadratic transition |
| 10 | `ays` | YES | YES | Align Your Steps |
| 11 | `turbo` | YES | YES | Few-step schedule |
| 12 | `polyexponential` | YES | YES | Polynomial-exponential blend |
| 13 | `kl_optimal` | YES | YES | KL-divergence optimal |
| 14 | `bong_tangent` | YES | YES | Tangent-based schedule |

Standalone: 14 schedulers. Trainer: 13 (missing `flux_simple`).

---

## E. Feature Inventory

### CFG Pipeline (`sampling/cfg.py`)

- **Standard CFG**: `cond + cfg_scale * (cond - uncond)`
- **RescaleCFG**: arXiv:2305.08891 — prevents color shift at high CFG
- **MaHiRo**: Post-CFG correction for detail preservation
- **CFGNorm**: Normalize CFG output magnitude (standalone only)
- **Epsilon Scaling**: Scale the epsilon prediction
- **Distilled CFG Scale**: For distilled models
- **Hook System**: `CFGHookRegistry` with PRE_CFG / CFG / POST_CFG injection points

### Prediction Types (`sampling/prediction.py`)

8 prediction type classes (identical in both projects):

| Class | Models | Behavior |
|-------|--------|----------|
| `EpsPrediction` | SD 1.5, SDXL | c_in scaling, discrete sigma→timestep |
| `VPrediction` | SD 2.x | v-prediction parameterization |
| `FlowPrediction` | Wan, Qwen, Lumina, ZImage | sigma→timestep = sigma directly |
| `FluxPrediction` | Flux 1/2 (all) | mu-based sigma shifting for resolution |
| `EDMPrediction` | EDM models | EDM parameterization |
| `ContinuousEDMPrediction` | Continuous EDM | Continuous sigma |
| `ContinuousVPrediction` | Continuous VP | Continuous VP |
| `DiscreteFlowPrediction` | Discrete flow | Discrete flow matching |

### Conditioning (`engine/conditioning.py`)

Per-architecture model kwargs dispatch covering 8 architectures:

| Architecture | Forward Signature | Notes |
|-------------|------------------|-------|
| SD 1.5 | `(x, timesteps, context)` | UNet-style |
| SDXL | `(x, timesteps, context, y)` | ADM = pooled(1280) + freq_time_ids(1536) = 2816 |
| SD3 | `(hidden_states, encoder_hidden_states, pooled_projections, timestep)` | DiT-style |
| Flux 1 | `(img, txt, timesteps, img_ids, txt_ids, vector, guidance)` | Packed + 3D IDs |
| Klein/Flux 2 | `(img, txt, timesteps, img_ids, txt_ids)` | Patchified + packed + 4D IDs |
| Chroma | `(img, txt, timesteps, img_ids, txt_ids, vector)` | Flux 1 minus guidance |
| ZImage | `(x, timestep, cap_feats, cap_mask)` | NextDiT-style |
| Qwen Image | `(x, timestep, context, attention_mask)` | 5D latents (B,C,T,H,W) |

Latent utilities: `patchify_latents`, `pack_flux_latents`, `pack_klein_latents`, `unpack_*` for each.

### Memory Management (`memory/`)

| Module | Key Classes | Function |
|--------|------------|----------|
| `vram.py` | VRAMBudget, VRAMState | VRAM detection, budget calc, CUDA/XPU detection |
| `streams.py` | StreamPool, CastBuffer, GatheredTransfer | CUDA stream async transfers, dtype casting |
| `pinned.py` | PinnedMemoryManager | Pin weights to page-locked memory |
| `offload.py` | OffloadLinear, OffloadConv2d, OffloadMixin | Layer-level CPU offload |
| `manager.py` | ModelManager, LoadedModel | Model lifecycle, smart eviction |
| `ram.py` | get_ram_usage_ratio, is_ram_pressure_high | System RAM monitoring |

VRAM modes: `auto` / `high` / `normal` / `low` / `no_vram`

### Attention Backends (`attention/`)

| Backend | File | Description |
|---------|------|-------------|
| Auto | `backends.py` | `select_best_backend()` |
| SageAttention | `sage.py` | Quantized attention (fastest) |
| Flash Attention 2 | `flash.py` | Memory-efficient fused |
| xFormers | `xformers_attn.py` | Memory-efficient |
| Scaled Dot Product | `sdp.py` | PyTorch native |
| einsum | `sdp.py` | Fallback |
| Token Merging | `token_merging.py` | ToMe-style reduction |

### Quantization (`quantization/`)

| Method | File | Description |
|--------|------|-------------|
| INT8 | `int8.py` | 8-bit integer |
| FP8 | `fp8.py` | 8-bit float |
| BNB NF4 | `bnb.py` | bitsandbytes 4-bit |
| BNB FP4 | `bnb.py` | bitsandbytes 4-bit float |
| GGUF | `gguf.py` | GGUF format |
| Nunchaku | `nunchaku.py` | Nunchaku backend |

### LoRA System (`lora/`)

| File | Function | Description |
|------|----------|-------------|
| `loader.py` | `load_lora()`, `detect_lora_type()`, `normalize_lora_keys()` | Load + detect standard/kohya/diffusers format |
| `merge.py` | `merge_lora_into_model()`, `unmerge_lora_from_model()` | Permanent weight merging |
| `online.py` | `apply_online_lora()`, `remove_online_lora()` | Runtime hooks (no weight mod) |
| `adapters.py` | LoRAAdapter, LoHAAdapter, LoKRAdapter, OFTAdapter | 4 adapter architectures |

### Klein 9B Native Image Editing

Multi-reference editing via sequence concatenation (`edit_klein9b.py`):
1. VAE encode source → extract mean (32ch) → patchify (128ch) → pack to sequence
2. Reference position IDs with T-coordinate offset = 10.0
3. Concatenate noise + reference in sequence dimension → `(B, 2*H*W, 128)`
4. Model processes via joint attention (RoPE distinguishes noise vs reference via T-coordinate)
5. Slice output to first `noise_seq_len` tokens only
6. Unpack → unpatchify → VAE decode

denoise_strength=1.0 works best for color/style edits. Multi-reference: N images with T-offsets 10, 20, 30...

### Feature Comparison

| Feature | Standalone | Trainer-Integrated |
|---------|-----------|-------------------|
| txt2img | YES | YES |
| img2img | NO | NO |
| Inpainting | NO | NO |
| Image editing | YES (Klein 9B, Qwen) | NO |
| Video generation | YES (LTX2, Wan t2v/i2v) | PARTIAL (Wan only) |
| LoRA | YES (4 adapter types) | YES (same) |
| Quantization | YES (6 modes) | YES (same) |
| Attention backends | YES (5 + token merging) | YES (same) |
| VRAM management | YES (ModelManager + streams) | Basic (manual offload) |
| Block-level offload | Per-script hooks | OffloadLinear/Conv2d |
| Stage cache | YES (LRU, 1 GiB) | NO |
| VAE tiling | YES (multi-pass + temporal) | NO |
| Regional prompting | YES | YES |
| CFGNorm | NO | YES |
| RescaleCFG | YES | YES |
| MaHiRo | YES | YES |
| Seeds | YES | YES |
| Batch generation | YES | NO (single-image scripts) |
| OOM recovery | YES | NO |

---

## F. TUI State

**Framework**: DearPyGui (NOT Textual)
**File**: `serenity/ui/tabs/inference.py` (783 lines)
**Entry point**: `serenity/ui/app.py` → `SerenityApp`
**Tabs**: general, model, training, sampling, lora, embedding, concepts, data, backup, **inference**

### Inference Tab Layout

**Left column**:
- Model selection via `ModelScanner` (scans dirs for `.safetensors`/`.ckpt`/`.pt`/`.bin`/`.gguf`)
- Prompt / negative prompt text areas
- Sampling params: seed, steps, CFG scale, sampler dropdown, scheduler dropdown
- Generate / Cancel buttons with progress bar

**Right column**:
- Resolution presets
- LoRA selection (path + weight)
- Advanced: VRAM mode, quantization, attention backend, RescaleCFG, MaHiRo, batch count
- Output directory, generation info, history

**Execution**: Runs generation in background thread using `InferenceEngine` directly.

### Critical Issue

The TUI inference tab imports from `serenity.inference.engine.InferenceEngine` — the trainer-integrated version with diffusers contamination. It does NOT use the standalone engine. **The TUI inference tab currently requires diffusers to function.**

---

## G. CLI State

**File**: `serenity/cli/commands.py` (1030 lines)
**Available command**: `train` only

**There is NO `generate` command.** Users must:
1. Run generation scripts directly (`python generate_klein4b.py`)
2. Use the TUI inference tab
3. Use the web API

---

## H. API State

**Files**: `serenity/ui/web/api.py`, `server.py`, `websocket.py`, `scanner.py`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/generate` | Submit generation job (returns job_id) |
| GET | `/api/job/{id}` | Check job status / retrieve result |
| GET | `/api/models` | List all detected models |
| POST | `/api/models/refresh` | Re-scan model directories |
| GET | `/api/status` | Engine status (loaded model, VRAM) |
| GET | `/api/samplers` | List available samplers |
| GET | `/api/schedulers` | List available schedulers |
| POST | `/api/interrupt` | Cancel current generation |
| WS | `/ws` | WebSocket real-time progress |

`ModelScanner` scans configured directories. CORS configured for localhost. Background task execution. WebSocket progress streaming.

### Note

The web API imports from `serenity.inference` (trainer-integrated), not standalone. Same diffusers contamination issue as TUI.

---

## I. Config / Settings

### InferenceConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | str | required | Path to model checkpoint |
| `model_dtype` | str/dtype | float16 | Model weight dtype |
| `vae_path` | str/None | None | Separate VAE path |
| `vram_mode` | VRAMMode | auto | Memory strategy |
| `attention_backend` | AttentionBackend | auto | Attention implementation |
| `quantization` | QuantizationMode | none | Weight quantization |
| `lora_paths` | list[str] | [] | LoRA file paths |
| `lora_weights` | list[float] | [] | LoRA strengths |
| `sampler` | str | euler | Sampler algorithm |
| `scheduler` | str | normal | Noise schedule |
| `steps` | int | 20 | Denoising steps |
| `cfg_scale` | float | 7.0 | CFG guidance scale |
| `width` | int | 512 | Image width |
| `height` | int | 512 | Image height |
| `rescale_cfg` | float | 0.0 | RescaleCFG strength |
| `mahiro` | bool | False | MaHiRo correction |
| `clip_skip` | int | 0 | CLIP skip layers |
| `vae_tiling` | bool | False | VAE tiling |

### Enums

**VRAMMode**: `auto`, `high`, `normal`, `low`, `no_vram`
**AttentionBackend**: `auto`, `sage`, `flash`, `xformers`, `sdp`, `einsum`
**QuantizationMode**: `none`, `int8`, `fp8`, `bnb_nf4`, `bnb_fp4`, `gguf`

### Diffusers Contamination Sites (Trainer-Integrated)

11 import sites in 7 files:

| File | Line | Import |
|------|------|--------|
| `engine.py` | 563 | `from diffusers.models import AutoencoderKL` |
| `models/loader.py` | 234 | `from diffusers.models import AutoencoderKL` |
| `models/sd15.py` | 77 | `from diffusers.models import UNet2DConditionModel` |
| `models/sdxl.py` | 137 | `from diffusers.models import UNet2DConditionModel` |
| `models/sdxl.py` | 253 | `from diffusers.models import UNet2DConditionModel` |
| `models/sd3.py` | 69 | `from diffusers.models import SD3Transformer2DModel` |
| `models/flux.py` | 133 | `from diffusers.models import FluxTransformer2DModel` |
| `models/flux.py` | 324 | `from diffusers.models import FluxTransformer2DModel` |
| `models/flux.py` | 380 | `from diffusers.models import FluxTransformer2DModel` |
| `models/chroma.py` | 71 | `from diffusers.models import FluxTransformer2DModel` |
| `models/wan.py` | 55 | `from diffusers.models import WanTransformer3DModel` |

---

## Summary Assessment

### What Works Well

1. **Standalone engine is solid** — 92 Python files, pure PyTorch, clean separation of concerns. Correct architecture-specific conditioning for 8+ model types.

2. **Architecture detection is excellent** — 14 architectures from state dict keys without loading weights. Covers SD 1.5 through Flux 2, Chroma, Wan, Qwen, ZImage, Lumina.

3. **7 generation scripts confirmed working** — Klein 4B/9B, Flux 1 Dev, SDXL, SD3.5 Medium, ZImage, Klein 9B Edit. Measured performance and VRAM.

4. **Memory management is sophisticated** — Block-level offloading, CUDA streams, pinned memory, 5 VRAM modes, RAM pressure monitoring.

5. **Sampler/scheduler coverage is comprehensive** — 10 samplers (4 built-in), 14 schedulers, all implemented.

6. **Text encoder mappings are correct** — CLIP, T5, Qwen3 (2 modes), Mistral, Qwen 2.5 VL, Gemma3 with proper per-model layer extraction.

7. **LoRA handles all major formats** — Standard, Kohya, diffusers keys. 4 adapter types.

8. **Klein 9B native editing works** — Multi-reference sequence concatenation, verified.

### What's Broken

1. **Trainer `serenity/inference/` has 11 diffusers imports** despite `DO_NOT_USE_DIFFUSERS.md` existing in the same directory.

2. **TUI and Web API import from the wrong engine** — both use the diffusers-contaminated trainer version, not the standalone.

3. **No CLI generate command** — only `train` exists.

4. **Standalone engine has no git tracking** — 92 untracked Python files.

5. **5 generation scripts untested** — SD3.5 Large, Qwen Image, LTX2, Wan t2v/i2v.

6. **Missing model weights** — Chroma, Flux 2 Dev, Mistral3 24B, Lumina 2.

7. **No SD 1.5 generation script** — detection and UNet exist but no script.

8. **Two codebases duplicating logic** — detection, loading, conditioning, etc. exist in both with the standalone being correct.

9. **No img2img or inpainting** in either codebase.

### Completeness vs Forge/ComfyUI

~35-40%. The infrastructure (samplers, schedulers, attention, quantization, memory, TUI, API) is solid. But:
- Model loading requires diffusers (trainer) or has no UI (standalone)
- No img2img, inpainting, or ControlNet
- No unified CLI
- Two divergent codebases
- The standalone native models need to replace diffusers imports in the integrated engine

### Recommended Actions

1. **Replace `serenity/inference/models/*.py` adapters** with standalone engine's native `nn.Module` implementations — eliminates all 11 diffusers imports
2. **Port standalone engine back into `serenity/inference/`** — or make it a proper subpackage
3. **Add `generate` CLI command** — expose inference via `serenity generate`
4. **Git-track the standalone engine** — 92 untracked files is a liability
5. **Sync scheduler delta** — add `flux_simple` to trainer engine
6. **Test remaining scripts** — SD3.5 Large, Qwen Image, LTX2, Wan 2.1
7. **Write `generate_sd15.py`** — trivial given existing infrastructure
