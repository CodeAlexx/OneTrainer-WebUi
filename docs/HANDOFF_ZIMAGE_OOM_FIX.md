# Handoff: Z-Image BF16 OOM Fix

**Date**: 2026-01-25
**Status**: IN PROGRESS
**Priority**: HIGH

---

## Problem Summary

Z-Image LoRA training with BF16 (no quantization) causes OOM in Serenity but works in OneTrainer.

- **Serenity**: OOM at 22GB allocated on 24GB GPU
- **OneTrainer**: Runs fine with ~8GB peak VRAM

User explicitly stated: **"we DO NOT QUANTIZE ZIMAGE!!!!!!!!!!!!!!!!!!! BP16 !"**

---

## Root Cause

Standard `from_pretrained()` loading doubles memory during the load phase:

```python
# BAD - What Serenity was doing:
self._transformer = ZImageTransformer2DModel.from_pretrained(
    model_path,
    subfolder="transformer",
    torch_dtype=dtype,
    local_files_only=True,
)
# Creates full model tensors (~12GB), then loads weights (~12GB) = 24GB peak
```

OneTrainer uses `accelerate.init_empty_weights()` to avoid this:

```python
# GOOD - What OneTrainer does:
with accelerate.init_empty_weights():
    sub_module = module_type.from_config(config)
# Creates meta tensors (0 bytes), then loads weights directly (~12GB) = 12GB peak
```

Reference: `/home/alex/OneTrainer/modules/modelLoader/mixin/HFModelLoaderMixin.py:220-221`

---

## Model Sizes

- **Z-Image transformer**: ~23GB on disk, ~12GB in BF16
- **T5 text encoder**: ~3GB in BF16
- **VAE**: ~300MB

---

## Partial Fix Applied

File: `serenity/models/zimage.py`

### Added Imports
```python
import os
import accelerate
from safetensors.torch import load_file
```

### Added Helper Function
```python
def _load_sharded_safetensors(path: str, subfolder: str) -> dict:
    """Load state dict from sharded or single safetensors files."""
    subpath = os.path.join(path, subfolder)
    index_file = os.path.join(subpath, "diffusion_pytorch_model.safetensors.index.json")
    single_file = os.path.join(subpath, "diffusion_pytorch_model.safetensors")

    state_dict = {}
    if os.path.isfile(index_file):
        import json
        with open(index_file, 'r') as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard in shard_files:
            shard_path = os.path.join(subpath, shard)
            state_dict.update(load_file(shard_path))
    elif os.path.isfile(single_file):
        state_dict = load_file(single_file)
    else:
        raise FileNotFoundError(f"No safetensors found in {subpath}")
    return state_dict
```

### Updated Transformer Loading
```python
# Load config only
transformer_config = ZImageTransformer2DModel.load_config(
    model_path,
    subfolder="transformer",
    local_files_only=True,
)

# Create empty model (meta tensors - no memory)
with accelerate.init_empty_weights():
    self._transformer = ZImageTransformer2DModel.from_config(transformer_config)

# Load weights directly into the empty model
state_dict = _load_sharded_safetensors(model_path, "transformer")
for key, value in state_dict.items():
    parts = key.split(".")
    module = self._transformer
    for part in parts[:-1]:
        module = getattr(module, part)
    param_name = parts[-1]
    if param_name in module._parameters:
        module._parameters[param_name] = nn.Parameter(
            value.to(dtype=dtype), requires_grad=False
        )
    elif param_name in module._buffers:
        module._buffers[param_name] = value.to(dtype=dtype)
del state_dict
```

---

## Status: FIX VERIFIED WORKING

The memory-efficient loading code IS fully implemented in `serenity/models/zimage.py`:
- Lines 51-108: Helper functions for loading sharded safetensors
- Lines 197-222: Transformer loading with `accelerate.init_empty_weights()`

## Test Results (2026-01-25)

**Z-Image BF16 Training - SUCCESS**
- 118 steps in 2:52 (~1.46s/step)
- **VRAM: 11.8GB** (previously OOM at 22GB)
- Final smooth loss: 0.5178

The fix works. Memory usage is now stable and training completes without OOM.

## Remaining Task

**Compare loss with OneTrainer** - Original goal was loss parity testing

---

## Test Config

File: `serenity_zimage_test.yaml`

```yaml
model_type: zimage
model:
  path: /home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/013496ad041be2e9ded1c291c00d8cc4054a6422
  dtype: bfloat16
adapter:
  type: lora
  rank: 16
  alpha: 16.0
memory:
  strategy: noop
  gradient_checkpointing: "on"
  quantization: none  # MUST BE NONE FOR BF16
data:
  concepts:
    - /home/alex/eri2
  cache_dir: /home/alex/OneTrainer/workspace/zimage_test_cache
  resolution: 512
  batch_size: 1
learning_rate: 3.0e-4
epochs: 1
```

---

## Related Fix

Also fixed dtype mismatch in `serenity/utils/quantization.py:230-233`:

```python
# Fixed dequantized weight dtype for sampling
w = dequantize(self.weight.detach(), self.scale).to(x.dtype)
bias = self.bias.to(x.dtype) if self.bias is not None else None
```

---

## Notes

- User deleted the GitHub repo - may need to re-initialize
- INT8 Z-Image training works but user explicitly doesn't want quantization
- The change for Klein (Flux 2) broke Z-Image because Klein uses the standard loading pattern which was copied over

---

## Command to Test

```bash
python -m serenity --config serenity_zimage_test.yaml
```
