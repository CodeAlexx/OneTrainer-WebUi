# Stagehand Guide (Serenity)

This document is the persistent runbook for Stagehand in Serenity. Use it in future sessions for setup, validation, tuning, and failure analysis.

## 1. What Stagehand Is

Stagehand is Serenity's block-swapping runtime for large transformer training.

Core behavior:
- Keeps transformer blocks off GPU by default.
- Stages block params through a fixed pinned host pool.
- Prefetches lookahead blocks and evicts old blocks via VRAM watermarks.
- Integrates with training via forward/backward hooks.

Primary implementation:
- `serenity/stagehand/__init__.py`
- `serenity/stagehand/scheduler.py`
- `serenity/stagehand/registry.py`
- `serenity/memory/stagehand_strategy.py`
- `serenity/cli/native_diffusion.py`

## 2. Activation Path

Single-stage activation path (native diffusion):
1. `native_diffusion.py` sees `memory.strategy == "stagehand"` or stagehand flags.
2. Builds `StagehandStrategyConfig`.
3. Calls `StagehandStrategy.setup()`.
4. Strategy builds `StagehandRuntime`.
5. Runtime builds registry, pool, scheduler, telemetry, transfer engine.
6. Training loop wraps each step in `strategy.forward_context()`.

Expected log signatures:
- `[native/diffusion] stagehand strategy active (pool=..., vram=...-...MB)`
- If enabled: `[native/diffusion] stagehand file-backed mode enabled (..., converted_params=...)`

Dual WAN mode activation path:
- Each stage (`high` / `low`) creates its own Stagehand runtime.
- Pool is reused across swaps via `shutdown_keep_pool()` and `setup_with_pool()`.
- Stage swap logs appear as:
  - `[native/diffusion/dual] === SWAPPING high → low at step ... ===`
  - `[native/diffusion/dual] swap complete in ...s`

## 3. Config Surface (What Actually Matters)

Primary config is under `memory.stagehand` in training JSON.

Important keys:
- `pinned_pool_mb`: total pinned host pool.
- `pinned_slab_mb`: slab size (must divide pool).
- `vram_high_watermark_mb`: eviction trigger.
- `vram_low_watermark_mb`: eviction stop target.
- `prefetch_window_blocks`: lookahead depth.
- `max_inflight_transfers`: concurrent async copies.
- `telemetry_enabled`: writes Stagehand stats.
- `telemetry_file`: default `stagehand_telemetry.jsonl`.
- `file_backed_weights`: attempt safetensors-backed frozen params.
- `checkpoint_path`: safetensors source for file-backed conversion.

Family defaults:
- File-backed defaults to enabled for WAN in `native_diffusion.py`.
- If checkpoint path is not `.safetensors`, file-backed is disabled.

## 4. File-Backed Behavior (Current Reality)

Current implementation is hybrid file-backed:
- Registry is built from real module structure first.
- `convert_to_file_backed()` maps matching frozen params to safetensors offsets.
- Converted params in module are replaced with empty CPU tensors.
- Mutable params (e.g. trainable adapters) stay module-backed.

Operational consequence:
- Eviction with save-back only clones mutable params to CPU for file-backed blocks.
- Frozen base params reload from mmap-backed safetensors slices.

Validation signal:
- `converted_params > 0` means file-backed conversion succeeded for at least some params.
- `converted_params=0` means effectively module-backed behavior remains.

## 5. Runtime Model: Prefetch + Evict

Per step lifecycle:
1. `begin_step(step)`
2. For each block pre-forward:
   - ensure block is GPU-ready (stall if needed)
   - maybe evict if above high watermark
   - prefetch lookahead window
3. Post-block:
   - decrement refcount
   - numeric guard check (`nan_count`, `inf_count`)
   - eviction pass if needed
4. `end_step()` records pool + VRAM stats and archives telemetry

State machine (`residency.py`):
- `unloaded -> host_staged -> prefetching -> gpu_ready`
- `gpu_ready -> evicting -> unloaded` (or host_staged in legacy save-back flow)
- `gpu_ready -> gpu_freeing -> unloaded` (no save-back)

## 6. Telemetry: How To Read It

Telemetry JSONL fields (`StepMetrics`):
- transfer: `h2d_bytes`, `d2h_bytes`
- stalls: `stall_time_ms`, `stall_count`
- scheduler: `evictions`, `prefetch_hits`, `prefetch_misses`
- memory: `vram_used_mb`, `vram_reserved_mb`, `host_pool_free`, `host_pool_in_use`
- numerics: `nan_count`, `inf_count`

Useful interpretation:
- high `prefetch_misses` + high stall time: lookahead/pool/inflight likely too small or IO bottleneck.
- `nan_count` or `inf_count` > 0: numeric instability in forward outputs.
- low `host_pool_free` for long periods: pool pressure.
- `vram_used_mb` hugging high watermark: constant eviction pressure.

## 7. Minimal Working Config (Stagehand)

```json
{
  "memory": {
    "strategy": "stagehand",
    "stagehand": {
      "pinned_pool_mb": 3072,
      "pinned_slab_mb": 512,
      "vram_high_watermark_mb": 20000,
      "vram_low_watermark_mb": 16000,
      "prefetch_window_blocks": 2,
      "max_inflight_transfers": 1,
      "telemetry_enabled": true,
      "telemetry_file": "stagehand_telemetry.jsonl",
      "file_backed_weights": true
    }
  }
}
```

Lower-memory fallback (64 GB host RAM pressure):
- `pinned_pool_mb: 2048`
- `prefetch_window_blocks: 1-2`
- `max_inflight_transfers: 1`

## 8. Verification Checklist (Always Run)

1. Process is active:
```bash
ps -eo pid,etime,rss,cmd | rg 'python.*-m serenity train'
```

2. Stagehand activation log exists:
```bash
rg -n "stagehand strategy active|stagehand file-backed" <train-log>
```

3. Recent training metrics exist:
```bash
rg -n "\[native/diffusion\] step|\[native/diffusion/dual\] step" <train-log> | tail -n 5
```

4. Telemetry is updating:
```bash
tail -n 3 stagehand_telemetry.jsonl
```

5. Memory health snapshot:
```bash
free -h
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

## 9. Symptom -> Diagnosis -> Action

### A) Host OOM / process killed
Likely causes:
- pinned pool too large for host RAM headroom.
- module-backed save-back cloning pressure (especially when file-backed conversion fails).

Actions:
- reduce `pinned_pool_mb` to 2048-3072.
- verify `converted_params > 0`.
- keep desktop/system load lean during training.

### B) `converted_params=0`
Likely causes:
- checkpoint path mismatch.
- non-safetensors source.
- key/shape mismatch for many params.

Actions:
- ensure `.safetensors` checkpoint path is correct.
- confirm model variant aligns with loaded transformer.
- inspect warnings and continue as module-backed if unavoidable.

### C) High loss / exploding behavior
Likely causes:
- optimizer/LR/clip settings, not Stagehand design itself.
- numerical instability in model/data settings.

Actions:
- check gradient clipping and LR.
- check telemetry `nan_count`/`inf_count`.
- if non-finite appears, reduce LR and verify dtype/precision path.

### D) Speed too low / high ETA
Likely causes:
- high stall time from prefetch misses.
- excessive eviction churn near watermark.

Actions:
- tune `prefetch_window_blocks` (usually 2).
- tune `max_inflight_transfers` (1-2).
- slightly raise `vram_high_watermark_mb` only if safe.

### E) "Stagehand not working?"
Proof criteria:
- activation log present.
- stagehand telemetry file actively appending.
- stagehand metrics fields change across steps.

If not met:
- check config path actually used by run.
- check fallback warning logs (Stagehand setup failed).

## 10. Assistant Playbook For Future Sessions

When user asks quick ops questions (`speed?`, `loss?`, `mem usage?`, `eta?`, `stagehand works?`):
1. Report latest step/loss/speed/eta from log.
2. Report trainer PID + RSS + host available RAM.
3. Report latest telemetry line with nan/inf and prefetch stats.
4. State whether Stagehand is active with explicit evidence.
5. Provide one concrete next action.

When user reports crash:
1. Determine if process exited.
2. Check OOM evidence and RAM pressure first.
3. Distinguish module-backed vs file-backed via conversion logs.
4. Recommend smallest config change first (pool/window/inflight).

Use exact paths and values in responses; avoid generic advice.

## 11. One-Command Status (Helper Script)

```bash
python ~/.codex/skills/serenity-stagehand-monitor/scripts/stagehand_status.py \
  --project-root /home/alex/serenity
```

Optional:
- `--log /path/to/train.log`
- `--telemetry /path/to/stagehand_telemetry.jsonl`
- `--pid <trainer_pid>`
- `--json`
