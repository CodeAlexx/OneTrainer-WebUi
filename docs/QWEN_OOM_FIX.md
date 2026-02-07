# Qwen Image Edit OOM Fix Documentation

## Overview

This document describes the OOM (Out of Memory) bugs discovered and fixed during Qwen Image Edit model training with layer offloading enabled.

## Problem Statement

Training the 20.4B parameter Qwen Image Edit model with 50% layer offloading was failing with OOM despite the expected memory usage being ~10 GB:

```
Expected: ~10 GB with 50% layer offload
Actual: 22.62 GB → OOM on 24 GB GPU
```

## Root Causes

### Bug 1: Step 0 Sampling Triggered Incorrectly

**Location:** `serenity/sampling/sampler.py`

**Issue:** The `should_sample()` method incorrectly triggered sampling at step 0 even when `sample_at_start=False` and `interval=25`.

**Root Cause:** The modulo operation `0 % 25 == 0` returns `True`, causing step 0 to trigger sampling.

**Before (buggy):**
```python
def should_sample(self, progress: TrainProgress) -> bool:
    if not self.config.enabled:
        return False

    # Check sample_at_start
    if self.config.sample_at_start and progress.global_step == 0:
        return True

    # BUG: 0 % interval == 0 for any interval!
    unit = TimeUnit(self.config.interval_unit)
    return self.tracker.repeating_action_needed(
        "sample", self.config.interval, unit, progress,
        start_at_zero=True,  # This causes 0 % 25 == 0 to return True
    )
```

**After (fixed):**
```python
def should_sample(self, progress: TrainProgress) -> bool:
    if not self.config.enabled:
        return False

    # Check sample_at_start
    if self.config.sample_at_start and progress.global_step == 0:
        return True

    # Skip step 0 if sample_at_start is False
    # (0 % interval == 0 for any interval, so we must explicitly skip)
    if progress.global_step == 0:
        return False

    # Check interval trigger for steps 25, 50, 75...
    unit = TimeUnit(self.config.interval_unit)
    return self.tracker.repeating_action_needed(
        "sample", self.config.interval, unit, progress,
        start_at_zero=True,
    )
```

### Bug 2: Sampling Bypassed Layer Offload Conductor

**Location:** `serenity/training/trainer.py` (two locations)

**Issue:** After sampling completed, the code reloaded the transformer using raw `module.to(device)` which bypassed the layer offload conductor, causing ALL layers to load onto GPU.

**Root Cause:** The sampler offloads the transformer to CPU for sampling. When reloading for training, the code used:
```python
module = getattr(self.model, "transformer", self.model)
module.to(self.device)  # Bypasses conductor - loads ALL 20B params to GPU!
```

This ignored the `transformer_offload_conductor` which manages partial layer offloading.

**Before (buggy):**
```python
# After sampling, reload transformer
module = getattr(self.model, "transformer", self.model)
module.to(self.device)  # BUG: Bypasses layer offload conductor!
```

**After (fixed):**
```python
# After sampling, reload transformer via conductor for proper layer offloading
if hasattr(self.model, 'transformer_to'):
    self.model.transformer_to(self.device)
elif hasattr(self.model, 'transformer_offload_conductor') and self.model.transformer_offload_conductor is not None:
    self.model.transformer_offload_conductor.to(self.device)
else:
    # Fallback for models without conductor
    module = getattr(self.model, "transformer", self.model)
    module.to(self.device)
```

## Memory Results

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| GPU Memory Used | 22.62 GB (OOM) | 10.34-10.91 GB |
| Layer Offloading | Bypassed | Working (50%) |
| Training Step 0 | Failed | Completed |

## Technical Details

### Layer Offload Conductor

The `LayerOffloadConductor` manages which transformer layers stay on GPU vs CPU:

1. **Initial Load:** All layers move to GPU temporarily
2. **Offload:** Based on `layer_offload_fraction`, some layers move back to CPU
3. **During Training:** Layers swap between GPU/CPU as needed

When `model.transformer_to(device)` is called, it delegates to `conductor.to(device)` which maintains the partial offload state. Direct `module.to(device)` bypasses this and loads everything.

### Qwen Image Edit Model Sizes

- **Transformer:** 20.43B params (40.86 GB at bf16, 20.43 GB at int8)
- **Text Encoder:** 8.29B params (16.58 GB at bf16)
- **VAE:** 127M params

With 50% layer offload and int8 quantization:
- Expected GPU usage: ~10.22 GB
- Actual GPU usage: 10.34 GB (within expected range)

## Affected Files

1. `serenity/sampling/sampler.py` - Fixed step 0 sampling logic
2. `serenity/training/trainer.py` - Fixed transformer reload after sampling (2 locations)

## Testing

To verify the fix, run:

```bash
python -m serenity train serenity_qwen_image_edit_25step.yaml
```

Expected output:
- Memory stays at ~10-11 GB throughout training
- Step 0 sampling skipped (unless `sample_at_start: true`)
- Sampling at step 25, 50, etc. works correctly with layer offloading preserved

## Related Issues

The NaN loss at step 1 observed during testing is a separate model/forward pass issue unrelated to memory optimization. This should be investigated separately.
