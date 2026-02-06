# OmniGen2 Integration into OneTrainer - Ultrathink Implementation Plan

## Executive Summary

This document outlines the comprehensive plan for integrating OmniGen2 (VectorSpaceLab/OmniGen2) into OneTrainer's training pipeline. OmniGen2 is a multimodal generation model with dual decoding pathways, built on Qwen2.5-VL for visual understanding.

---

## 1. Architecture Analysis

### 1.1 OmniGen2 Core Components

| Component | Class/Source | Description |
|-----------|--------------|-------------|
| **Transformer** | `OmniGen2Transformer2DModel` | 32-layer DiT with triple refinement streams |
| **Text Encoder** | `Qwen2_5_VLModel` (transformers) | Vision-Language encoder from Qwen2.5-VL-3B-Instruct |
| **VAE** | `AutoencoderKL` (diffusers) | FLUX.1-dev VAE with 16 latent channels |
| **Tokenizer** | `AutoTokenizer` | Qwen2.5 tokenizer with chat template |
| **Scheduler** | `FlowMatchEulerDiscreteScheduler` | Flow matching scheduler for inference |
| **Transport** | `create_transport()` | Velocity-based flow matching for training |

### 1.2 Transformer Architecture Details

```
OmniGen2Transformer2DModel
├── patch_embed: Linear(16*2*2 → 2520)           # Patch embedding
├── timestep_embed: Lumina2CombinedTimestepCaptionEmbedding
├── noise_refiner_layers: ModuleList[2]          # Noise refinement (modulated)
├── ref_img_refiner_layers: ModuleList[2]        # Reference image refinement (modulated)
├── context_refiner_layers: ModuleList[2]        # Context refinement (non-modulated)
├── layers: ModuleList[32]                       # Main transformer blocks (modulated)
├── norm_out: LuminaLayerNormContinuous          # Output normalization
└── linear_out: Linear(2520 → 16*2*2)            # Output projection
```

### 1.3 Training Pipeline (Flow Matching)

```python
# Simplified training forward pass
1. text_embeds = text_encoder(tokens)           # Qwen2.5-VL encoding
2. latents = vae.encode(image).latent_dist.sample()
3. latents = (latents - shift_factor) * scaling_factor
4. t = transport.sample(latents)                 # Lognormal timestep sampling
5. noise = torch.randn_like(latents)
6. noisy_latents = t * noise + (1 - t) * latents  # Linear interpolation
7. velocity_pred = transformer(noisy_latents, t, text_embeds, ref_images)
8. velocity_target = noise - latents
9. loss = F.mse_loss(velocity_pred, velocity_target)
```

### 1.4 Key Differences from Existing Models

| Feature | OmniGen2 | FLUX | SD3 |
|---------|----------|------|-----|
| Text Encoder | Qwen2.5-VL (Vision-Language) | CLIP + T5 | CLIP + T5 + CLIP |
| Latent Channels | 16 | 16 | 16 |
| Training Method | Velocity Flow Matching | Flow Matching | Flow Matching |
| Reference Images | Yes (in-context generation) | No | No |
| Dual Guidance | Yes (text + image) | CFG only | CFG only |
| Triple Refiners | Yes (noise, ref, context) | No | No |

---

## 2. Files to Create

### 2.1 Model Class

**File**: `modules/model/OmniGen2Model.py`

```python
class OmniGen2Model(BaseModel):
    # Components
    tokenizer: AutoTokenizer | None
    text_encoder: Qwen2_5_VLModel | None  # Frozen
    vae: AutoencoderKL | None              # Frozen
    transformer: OmniGen2Transformer2DModel | None  # Trainable
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None

    # LoRA
    transformer_lora: LoRAModuleWrapper | None

    # Offloading
    transformer_offload_conductor: LayerOffloadConductor | None
    text_encoder_offload_conductor: LayerOffloadConductor | None

    # Block swap
    musubi_manager: MusubiBlockSwapManager | None
```

### 2.2 Model Loader

**File**: `modules/modelLoader/OmniGen2ModelLoader.py`
**File**: `modules/modelLoader/OmniGen2ModelLoaderLoRA.py`

Load from HuggingFace:
- Transformer: `OmniGen2/OmniGen2` subfolder `transformer`
- Text Encoder: `Qwen/Qwen2.5-VL-3B-Instruct`
- VAE: `black-forest-labs/FLUX.1-dev` subfolder `vae`

### 2.3 Model Setup

**File**: `modules/modelSetup/BaseOmniGen2Setup.py`
**File**: `modules/modelSetup/OmniGen2LoRASetup.py`
**File**: `modules/modelSetup/OmniGen2FineTuneSetup.py`

Key methods:
- `setup_optimizations()`: Gradient checkpointing, Musubi block swap
- `predict()`: Forward pass with velocity prediction
- `calculate_loss()`: Flow matching loss (MSE of velocity)

### 2.4 Model Sampler

**File**: `modules/modelSampler/OmniGen2Sampler.py`

Use diffusers pipeline pattern:
- FlowMatchEulerDiscreteScheduler for denoising
- Support for reference images
- Dual guidance (text + image)

### 2.5 Model Saver

**File**: `modules/modelSaver/OmniGen2ModelSaver.py`
**File**: `modules/modelSaver/OmniGen2ModelSaverLoRA.py`

Save LoRA weights in diffusers-compatible format.

### 2.6 Data Loader (Optional - if needed)

**File**: `modules/dataLoader/OmniGen2BaseDataLoader.py`

Handle reference images in training data.

---

## 3. Enum and Factory Updates

### 3.1 ModelType Enum

**File**: `modules/util/enum/ModelType.py`

```python
OMNIGEN2 = 'OMNIGEN2'

def is_omnigen2(self):
    return self == ModelType.OMNIGEN2

# Add to is_flow_matching():
or self.is_omnigen2()
```

### 3.2 Factory Registration

**File**: `modules/util/create.py`

Register all new classes:
- `create_model_loader()`: Add OmniGen2 cases
- `create_model_setup()`: Add OmniGen2 cases
- `create_model_saver()`: Add OmniGen2 cases
- `create_model_sampler()`: Add OmniGen2 cases

---

## 4. Implementation Phases

### Phase 1: Core Model Infrastructure (Priority: Critical)

1. **Create OmniGen2Model.py**
   - Define all component attributes
   - Implement `adapters()`, `to()`, `eval()`
   - Implement `encode_text()` with Qwen2.5-VL
   - Implement `scale_latents()`, `unscale_latents()`

2. **Create OmniGen2ModelLoader.py**
   - Load transformer from HuggingFace
   - Load text encoder (Qwen2.5-VL)
   - Load VAE (FLUX VAE)
   - Handle weight dtypes

3. **Update ModelType enum**
   - Add OMNIGEN2 value
   - Add `is_omnigen2()` helper
   - Update `is_flow_matching()` to include OmniGen2

### Phase 2: Training Setup (Priority: Critical)

4. **Create BaseOmniGen2Setup.py**
   - Implement `setup_optimizations()`:
     - Gradient checkpointing on transformer.layers
     - Musubi block swap support
     - Autocast context creation
   - Implement `predict()`:
     - Encode text with Qwen2.5-VL
     - Sample timesteps (lognormal distribution)
     - Forward through transformer
     - Return velocity prediction and target
   - Implement `calculate_loss()`:
     - Flow matching velocity loss (MSE)

5. **Create OmniGen2LoRASetup.py**
   - Create LoRA wrapper for transformer
   - Target modules: `["to_k", "to_q", "to_v", "to_out.0"]`
   - Implement `create_parameters()`
   - Implement `setup_model()`

### Phase 3: Inference & Saving (Priority: High)

6. **Create OmniGen2Sampler.py**
   - Create inference pipeline
   - Implement `sample()` method
   - Support FlowMatchEulerDiscreteScheduler
   - Handle reference images if present

7. **Create OmniGen2ModelSaverLoRA.py**
   - Save LoRA weights in diffusers format
   - Create model index file

### Phase 4: Factory Registration (Priority: High)

8. **Update create.py**
   - Register all OmniGen2 classes
   - Add imports
   - Add match cases for (OMNIGEN2, TrainingMethod) combinations

### Phase 5: Testing (Priority: Critical)

9. **Create unit tests**
   - Test model loading
   - Test forward pass
   - Test LoRA creation
   - Test sampling

10. **Integration testing**
    - Train a small LoRA
    - Generate samples
    - Verify loss decreases

---

## 5. LoRA Configuration

### 5.1 Target Modules

Based on OmniGen2's native training code:

```python
target_modules = [
    "to_k",      # Key projection in attention
    "to_q",      # Query projection in attention
    "to_v",      # Value projection in attention
    "to_out.0",  # Output projection in attention
]
```

### 5.2 Block Filter

For selective layer training:
- Main transformer: `layers.{0-31}`
- Noise refiner: `noise_refiner_layers.{0-1}`
- Ref image refiner: `ref_img_refiner_layers.{0-1}`
- Context refiner: `context_refiner_layers.{0-1}`

---

## 6. Block Swapping Strategy

### 6.1 Training Block Swap (Musubi)

```python
# In BaseOmniGen2Setup.setup_optimizations():
blocks_to_swap = config.musubi_blocks_to_swap
if blocks_to_swap == 0 and config.gradient_checkpointing.enabled():
    # Auto: swap ~half the main transformer blocks
    num_blocks = len(model.transformer.layers)  # 32
    blocks_to_swap = num_blocks // 2  # 16

if blocks_to_swap > 0:
    model.musubi_manager = MusubiBlockSwapManager.build(
        depth=len(model.transformer.layers),
        blocks_to_swap=blocks_to_swap,
        swap_device="cpu",
    )
    model.musubi_manager.activate_with_forward_hooks(
        model.transformer.layers,
        self.train_device,
        grad_enabled=True
    )
```

---

## 7. Dependencies

### 7.1 Required Packages

```
transformers>=4.51.3  # For Qwen2.5-VL
diffusers>=0.30.0     # For AutoencoderKL, schedulers
torch>=2.0.0
einops
flash-attn>=2.7.0 (optional, for faster attention)
```

### 7.2 Model Weights

Download from HuggingFace:
- `OmniGen2/OmniGen2` - Main model (transformer)
- `Qwen/Qwen2.5-VL-3B-Instruct` - Text encoder
- `black-forest-labs/FLUX.1-dev` - VAE (already used by Flux)

---

## 8. Memory Estimates

### 8.1 Full Precision (FP32)

| Component | Parameters | Memory |
|-----------|------------|--------|
| Transformer | ~3B | ~12 GB |
| Text Encoder | ~3B | ~12 GB |
| VAE | ~84M | ~0.3 GB |
| **Total** | ~6B | ~24 GB |

### 8.2 Mixed Precision (BF16)

| Component | Parameters | Memory |
|-----------|------------|--------|
| Transformer | ~3B | ~6 GB |
| Text Encoder | ~3B | ~6 GB |
| VAE | ~84M | ~0.2 GB |
| **Total** | ~6B | ~12 GB |

### 8.3 LoRA Training (BF16 + 8-bit quantized base)

| Component | Memory |
|-----------|--------|
| Quantized Transformer | ~3 GB |
| Text Encoder (frozen, offloaded) | ~0 GB (CPU) |
| VAE (frozen, offloaded) | ~0 GB (CPU) |
| LoRA weights | ~0.1-0.5 GB |
| Optimizer states | ~0.2-1 GB |
| Activations + gradients | ~4-8 GB |
| **Total** | ~8-12 GB |

---

## 9. Execution Order

```
1. Clone OmniGen2 repo ✓ (done)
2. Read and understand architecture ✓ (done)
3. Create OmniGen2Model.py
4. Create OmniGen2ModelLoader.py
5. Update ModelType enum
6. Create BaseOmniGen2Setup.py
7. Create OmniGen2LoRASetup.py
8. Create OmniGen2Sampler.py
9. Create OmniGen2ModelSaverLoRA.py
10. Update create.py
11. Download model weights
12. Run unit tests
13. Run integration training test
14. Generate samples to verify
```

---

## 10. Risk Assessment

### High Risk

1. **Qwen2.5-VL Integration**: Unlike CLIP/T5, this is a vision-language model. May need special handling for text-only encoding.
   - **Mitigation**: Study how OmniGen2 uses it for text-only (without images)

2. **Reference Image Support**: OneTrainer may not have infrastructure for optional reference images.
   - **Mitigation**: Start with text-to-image only, add reference later

### Medium Risk

1. **Memory Usage**: ~6B parameters may require careful memory management.
   - **Mitigation**: Use quantization, Musubi block swap, CPU offloading

2. **Flow Matching Transport**: Custom transport module may not align with OneTrainer's mixins.
   - **Mitigation**: Implement custom timestep sampling matching OmniGen2's approach

### Low Risk

1. **LoRA Compatibility**: Standard attention module targets should work.
2. **VAE Compatibility**: Uses same FLUX VAE already supported.

---

## 11. Success Criteria

1. **Model Loading**: Load all components without errors
2. **Forward Pass**: Training forward pass completes without OOM
3. **Loss Computation**: Loss is finite and decreases during training
4. **Sampling**: Generate coherent images during training
5. **LoRA Saving**: Save and reload LoRA weights successfully
6. **Memory**: Fit training on 24GB VRAM with LoRA + Musubi

---

## 12. Files Summary

| File | Status | Priority |
|------|--------|----------|
| `modules/model/OmniGen2Model.py` | TODO | Critical |
| `modules/modelLoader/OmniGen2ModelLoader.py` | TODO | Critical |
| `modules/modelLoader/OmniGen2ModelLoaderLoRA.py` | TODO | Critical |
| `modules/modelSetup/BaseOmniGen2Setup.py` | TODO | Critical |
| `modules/modelSetup/OmniGen2LoRASetup.py` | TODO | Critical |
| `modules/modelSetup/OmniGen2FineTuneSetup.py` | TODO | Medium |
| `modules/modelSampler/OmniGen2Sampler.py` | TODO | High |
| `modules/modelSaver/OmniGen2ModelSaverLoRA.py` | TODO | High |
| `modules/util/enum/ModelType.py` | TODO | Critical |
| `modules/util/create.py` | TODO | High |
| `tests/test_omnigen2.py` | TODO | High |
