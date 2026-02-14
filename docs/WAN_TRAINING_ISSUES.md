# WAN 2.2 Training Issues Tracker

## Status: All Issues Resolved

WAN 2.2 14B dual-stage (high/low noise) training with Stagehand block-swapping on a 24 GB GPU.

---

## Issue 1: OOM during first training step (RESOLVED)

**Status**: RESOLVED — no longer reproduces after fixes to issues 2-7

**Original symptom**: After caching completes, first forward pass OOMed with 20.94 GiB allocated.

**Resolution**: The OOM was caused by a combination of issues 2-7 (stale GPU processes, VAE/TE not freed after caching, dispatched module hooks blocking stagehand, etc.). After all those fixes, a 2-step training run completed successfully:
- GPU after offload: 8 MB (clean)
- GPU after non-block move: 453 MB (only small submodules)
- Forward + backward through all 40 blocks: no OOM
- Step 2 loss: 1.968654
- LoRA saved successfully

**Diagnostic**: `_audit_gpu_memory()` available in `native_diffusion.py` — enable with `"debug_memory": true` in config.

---

## Issue 2: Latent normalization shape bug (FIXED)

**Status**: Fixed in `wan.py` lines 228-236

**Root cause**: `torch.tensor([16 values]).unsqueeze(-1)` loop produced shape `(16, 1, 1, 1, 1)` instead of `(1, 16, 1, 1, 1)`.

**Fix**: Explicit `mean.view(*shape)` with `shape = [1, C, 1, 1, 1]`.

**Impact**: Latents were 6D `(1, 16, 16, 5, 48, 48)` instead of 5D `(1, 16, 5, 48, 48)`. Crashed transformer forward with `ValueError: too many values to unpack`.

---

## Issue 3: hf_device_map blocking stagehand (FIXED)

**Status**: Fixed in `wan.py` lines 129-131

**Root cause**: `WanTransformer3DModel.from_single_file()` sets `hf_device_map = {'': device(type='cpu')}`. `_is_dispatched_module()` checks this attribute and returns True, causing stagehand to be skipped.

**Fix**: `del transformer.hf_device_map` after loading.

**Impact**: Without this fix, the dispatched-transformer code path runs instead of stagehand. The model stays on CPU, forward pass gets device mismatch errors (CUDA input vs CPU weights).

---

## Issue 4: Slab size too small for WAN blocks (FIXED)

**Status**: Fixed in config files (`pinned_slab_mb: 1024`)

**Root cause**: Each WAN block is ~674 MB. With `pinned_slab_mb: 512`, `pool.acquire()` returns a list of slabs. `transfer.submit_h2d()` expects a single slab and crashes with `AttributeError: 'list' object has no attribute 'buffer'`.

**Fix**: Increased `pinned_slab_mb` from 512 to 1024 in both configs. 8192/1024 = 8 slabs exactly.

**Note**: Multi-slab H2D transfer is not implemented in the transfer engine. Block must fit in one slab.

---

## Issue 5: UMT5 text encoder download hang (FIXED)

**Status**: Fixed in `wan.py` lines 164-173

**Root cause**: `UMT5EncoderModel.from_pretrained("google/umt5-xxl", local_files_only=False)` tried to download the full ~50 GB model just to get config. Hung at "Fetching 6 files: 0%".

**Fix**: Use `AutoConfig.from_pretrained()` for config + `UMT5EncoderModel(config_obj)` + `load_state_dict()` from local safetensors.

---

## Issue 6: VAE/text encoder VRAM not freed after caching (FIXED)

**Status**: Fixed in `native_diffusion.py` lines 1477-1486

**Root cause**: After `_cache_training_data()`, the VAE and text encoder stayed on GPU, consuming VRAM needed for training.

**Fix**: Explicit `pipeline.vae.to("cpu")`, `native_model.offload_text_encoders(pipeline)`, `torch_gc()`.

---

## Issue 7: Stale GPU processes consuming VRAM (FIXED)

**Status**: Manually fixed by killing processes

**Root cause**: Two stale pytest processes (PIDs 378496, 378748) each held 284 MiB GPU memory, reducing available VRAM from 23.51 GiB to ~22.96 GiB.

---

## Architecture Notes

### WAN 2.2 Transformer Structure (14B)
```
rope:                  WanRotaryPosEmbed     (~0.0 MB)
patch_embedding:       Conv3d                (~0.6 MB)
condition_embedder:    WanTimeTextImageEmbedding (~442.6 MB)
blocks:                ModuleList × 40       (~26,809 MB, ~670 MB each)
norm_out:              FP32LayerNorm         (~0.0 MB)
proj_out:              Linear                (~0.6 MB)
Total:                                       ~27,253 MB (~26.6 GiB)
```

### Stagehand Config
```json
{
    "pinned_pool_mb": 8192,
    "pinned_slab_mb": 1024,
    "vram_high_watermark_mb": 20000,
    "vram_low_watermark_mb": 16000,
    "prefetch_window_blocks": 2
}
```

### Expected VRAM Budget
- Non-block submodules: ~445 MB
- 1-3 blocks on GPU at a time: 674-2022 MB
- Activations + optimizer state: ~1-2 GB
- **Expected peak: 3-5 GiB** (well within 24 GiB)

### Training Config
- Resolution: 384×384, Frames: 17, Batch: 1
- LoRA rank 16, alpha 16
- AdamW, constant LR 1e-4
- Gradient accumulation: 4
- Gradient checkpointing: on
