# Kandinsky 5 (K5) Integration Status

**Last Updated**: 2026-01-04
**Status**: BLOCKED - OOM crashes during model loading

---

## Model Specifications

- **Model**: Kandinsky 5 PRO (20B parameters)
- **Architecture**: DiT (Diffusion Transformer) with Flow Matching
- **Components**:
  - `transformer`: K5 DiT model (~40GB in float32, ~20GB in bf16)
  - `text_encoder_qwen`: Qwen2.5-VL (~30GB in float32, ~15GB in bf16)
  - `vae`: VAE encoder/decoder
- **Block Types**:
  - `TransformerEncoderBlock` - forward params: `x`
  - `TransformerDecoderBlock` - forward params: `visual_embed`, `text_embed`

## Hardware Constraints

- **System RAM**: 62GB
- **GPU VRAM**: Varies (typically 24GB for 4090)
- **Issue**: K5 PRO requires ~75GB+ RAM to load in float32

---

## Current State (2026-01-04)

### Files Status

| File | Status | Notes |
|------|--------|-------|
| `modules/modelLoader/Kandinsky5ModelLoader.py` | REVERTED | Using original OT loader |
| `modules/modelSetup/Kandinsky5LoRASetup.py` | REVERTED | Uses `enable_checkpointing()` |
| `modules/model/Kandinsky5Model.py` | REVERTED | No custom block swap |
| `modules/modelSampler/Kandinsky5Sampler.py` | REVERTED | Original sampler (still has corruption bug) |
| `modules/util/checkpointing_util.py` | MODIFIED | Added K5 block registration |
| `models/kandinsky-5-code/kandinsky/models/dit.py` | MODIFIED | `@torch.compile` commented out |

### Test Configuration

```json
// configs/k5_test.json
{
  "text_encoder": {
    "train": false,
    "weight_dtype": "BFLOAT_16"
  },
  "transformer": {
    "train": true,
    "weight_dtype": "BFLOAT_16"
  }
}
```

---

## Issues Encountered

### 1. OOM Crashes During Model Loading (BLOCKING)

**Symptoms**:
- System reboots/crashes during K5 model loading
- Occurs before training starts
- Crash point: "Loading state dict into transformer..."

**Root Cause**:
- K5 PRO transformer: ~40GB in float32
- Qwen text encoder: ~30GB in float32
- Total: ~70-75GB needed just to load
- System only has 62GB RAM

**Crash Sequence**:
1. Transformer created on CPU (float32) - ~40GB
2. Transformer weights loaded - OK
3. Qwen text encoder loading starts (float32) - +30GB
4. Total exceeds RAM → System crash

### 2. torch.compile Freezes (FIXED)

**Symptoms**: Training freezes indefinitely after model loads

**Root Cause**: `@torch.compile()` decorator in `dit.py` line 170

**Fix**: Commented out the decorator

### 3. Sampler Corruption (UNRESOLVED)

**Symptoms**: Inference produces corrupted/garbled images

**Status**: Not yet diagnosed - blocked by loading issues

---

## Attempted Solutions (All Failed)

### 1. Meta Device Loading
```python
# Attempted: Load model on meta device first, then materialize
transformer = K5Transformer(..., device='meta')
transformer = transformer.to_empty(device='cpu')
```
**Result**: Incompatible with block swapping, still crashed

### 2. Direct bf16 Loading
```python
# Attempted: Create model in bf16 directly
transformer = K5Transformer(...).to(torch.bfloat16)
```
**Result**: Still crashed - Qwen loads in float32 regardless

### 3. Custom Block Swapping
```python
# Attempted: Manual GPU↔CPU block transfer during forward/backward
class BlockSwapTransformer:
    def forward(self, ...):
        for block in self.blocks:
            block.to('cuda')
            output = block(...)
            block.to('cpu')
```
**Result**: Memory conflicts with meta tensors, didn't solve loading OOM

### 4. OneTrainer Standard Path (Current Test)
```python
# Using OT's enable_checkpointing() with LayerOffloadConductor
model.transformer_offload_conductor = enable_checkpointing(
    model.transformer, config, False, [
        (TransformerEncoderBlock, ["x"]),
        (TransformerDecoderBlock, ["visual_embed", "text_embed"]),
    ]
)
```
**Result**: Still crashing - doesn't help with LOADING, only training

---

## What Needs to Be Done

### Priority 1: Fix Model Loading OOM

**Option A: INT8 Quantization for Qwen** (Recommended)
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model.text_encoder_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_path,
    quantization_config=bnb_config,
    device_map='cpu',
    low_cpu_mem_usage=True,
)
```
- Reduces Qwen from ~30GB to ~8GB
- Reference: `/home/alex/diffusion-pipe/models/kandinsky_5.py` uses this

**Option B: Sequential Loading with Offload**
```python
# Load transformer, convert to bf16, offload to disk
# Then load Qwen
# Requires custom safetensors streaming
```

**Option C: mmap Loading**
```python
# Memory-map the model files instead of loading into RAM
# Requires model to be in specific format
```

### Priority 2: Fix Sampler Corruption

Once model loads successfully:
1. Debug K5 sampler inference path
2. Check noise scheduling
3. Verify latent/image conversion

### Priority 3: LoRA Training Verification

1. Confirm LoRA injection works with OT's `create_peft_wrapper()`
2. Test gradient flow through checkpointed blocks
3. Verify LayerOffloadConductor GPU transfers

---

## Key Files Reference

### OneTrainer Checkpointing System
- `modules/util/checkpointing_util.py` - `enable_checkpointing()` function
- `modules/util/LayerOffloadConductor.py` - GPU↔CPU transfer manager

### K5 Model Code
- `models/kandinsky-5-code/kandinsky/models/dit.py` - Main DiT implementation
- `models/kandinsky-5-code/kandinsky/pipelines/text2img_pipeline.py` - Inference

### Reference Implementation
- `/home/alex/diffusion-pipe/models/kandinsky_5.py` - Working K5 with INT8

---

## OneTrainer Offloading Architecture

```
LayerOffloadConductor
├── Manages GPU↔CPU transfers during training
├── Coordinates with gradient checkpointing
└── Blocks registered with forward parameter names

enable_checkpointing(model, config, compile, block_specs)
├── Wraps blocks with gradient checkpointing
├── Creates LayerOffloadConductor if offload enabled
└── Returns conductor for manual control
```

**Block Registration Format**:
```python
[
    (BlockClass, ["param1", "param2"]),  # Forward function params
]
```

---

## Debug Logging Locations

Crash-proof logging was added to track failure points:
- `/home/alex/OneTrainer/k5_load_debug.log` - Persistent file logging
- Logs: "Loading state dict into transformer...", "About to load Qwen..."

---

## Summary

**Current Blocker**: Cannot load K5 PRO model - exceeds 62GB RAM limit

**Most Promising Solution**: INT8 quantization for Qwen text encoder

**Next Step**: Implement BitsAndBytesConfig INT8 loading for Qwen to reduce memory from 30GB to ~8GB during model loading

**Key Insight**: The issue is LOADING, not TRAINING. OneTrainer's offloading helps during training but doesn't help get the model loaded initially.
