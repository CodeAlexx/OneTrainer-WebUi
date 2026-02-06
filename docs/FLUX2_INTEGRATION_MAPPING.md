# FLUX.2 Integration Mapping: SimpleTuner → EriTrainer

## Executive Summary

This document maps SimpleTuner's FLUX.2 training implementation to EriTrainer's architecture, providing a complete integration plan for FLUX.2 Klein 4B/9B support.

**Key Decision**: Create new `Flux2KleinModel` class rather than extending existing `FluxModel` because:
1. Different text encoders (Qwen3 vs CLIP+T5)
2. Different embedding stacking (layers 9,18,27 vs last_hidden_state)
3. No guidance embeddings in Klein
4. Different VAE processing (batch norm + pixel shuffle)

---

## Architecture Comparison

### FLUX.1 (Current EriTrainer)
```
Text Encoders: CLIP (768d) + T5 (4096d)
Embedding: T5 last_hidden_state → [B, 512, 4096]
Guidance: Yes (Dev), No (Schnell)
VAE: 16 channels, scaling_factor
Blocks: 19 double + 38 single = 57 total
```

### FLUX.2 Klein (Target)
```
Text Encoder: Qwen3 bundled (single encoder)
Embedding: Stack layers [9, 18, 27] → [B, 512, 3*d] = [B, 512, 12288] (9B) or [B, 512, 7680] (4B)
Guidance: No (Klein models ignore guidance)
VAE: 32 channels, batch norm, pixel shuffle to 128 channels
Blocks:
  - klein-4b: 5 double + 20 single = 25 total
  - klein-9b: 8 double + 24 single = 32 total
```

---

## Component Mapping

| SimpleTuner Component | Location | EriTrainer Target | Notes |
|----------------------|----------|-------------------|-------|
| `Flux2TransformerBlock` | `helpers/models/flux2/transformer.py:528` | Use diffusers native | No custom implementation needed |
| `AutoencoderKLFlux2` | `helpers/models/flux2/autoencoder.py` | Use diffusers native | Handle batch norm separately |
| Qwen3 text encoder | `helpers/models/flux2/model.py:341-401` | `models/flux2_klein.py` | Custom stacked embedding |
| Flow matching loss | `helpers/models/flux2/model.py:894-898` | `training/loss.py` | Already exists, reuse |
| Latent packing | `helpers/models/flux2/__init__.py:22-91` | `models/flux2_klein.py` | Similar to Flux.1, 4D position IDs |
| Block swapping | Uses Musubi | `training/block_swap.py` | Already exists, reuse |
| TREAD routing | `helpers/training/tread.py` | Optional Phase 2 | Token efficiency optimization |

---

## Implementation Files

### Phase 1: Core Implementation

#### 1. Enums (`eritrainer/core/enums.py`)

**ADD** to `ModelType`:
```python
# FLUX.2 Klein
FLUX_2_KLEIN_4B = "flux_2_klein_4b"
FLUX_2_KLEIN_9B = "flux_2_klein_9b"
FLUX_2_KLEIN_4B_BASE = "flux_2_klein_4b_base"  # Undistilled
FLUX_2_KLEIN_9B_BASE = "flux_2_klein_9b_base"  # Undistilled
```

**ADD** helper method:
```python
def is_flux_2_klein(self) -> bool:
    """Returns True for FLUX.2 Klein models."""
    return self in {
        ModelType.FLUX_2_KLEIN_4B,
        ModelType.FLUX_2_KLEIN_9B,
        ModelType.FLUX_2_KLEIN_4B_BASE,
        ModelType.FLUX_2_KLEIN_9B_BASE,
    }
```

**UPDATE** `is_flow_matching()` to include FLUX.2 Klein.

---

#### 2. Model Wrapper (`eritrainer/models/flux2_klein.py`)

**NEW FILE** - ~400 lines

```python
class Flux2KleinModel(ShiftScaleLatentScaler, VAELoaderMixin, TextEncoderLoaderMixin, SafeLoaderMixin, BaseModel):
    """
    FLUX.2 Klein model wrapper (4B and 9B variants).

    Architecture:
    - DiT-based transformer (smaller than FLUX.1)
    - Single text encoder: Qwen3 (bundled with model)
    - 32-channel VAE with batch normalization and pixel shuffle
    - Flow matching training (no guidance for Klein)

    Variants:
    - klein-4b: 25 blocks (5 double + 20 single), 7680 embedding dim
    - klein-9b: 32 blocks (8 double + 24 single), 12288 embedding dim
    - klein-*-base: Undistilled variants for better training
    """

    # Key differences from FluxModel:
    # 1. Single text encoder (Qwen3) vs dual (CLIP + T5)
    # 2. Stacked layer embeddings [9, 18, 27] vs last_hidden_state
    # 3. No guidance embeddings
    # 4. VAE batch normalization + pixel shuffle (32 → 128 channels)
    # 5. Different block counts
```

**Key Methods**:
- `encode_prompt_qwen3()`: Stack layers [9, 18, 27] into embeddings
- `patchify_latents()`: 32ch → 128ch via pixel shuffle
- `normalize_latents()`: Apply VAE batch norm statistics
- `pack_latents()` / `unpack_latents()`: 4D position IDs (T, H, W, L)

---

#### 3. Model Loader (`eritrainer/components/model_loader.py`)

**ADD** new loader class:
```python
class Flux2KleinModelLoader(BaseModelLoader):
    """Loader for FLUX.2 Klein models."""

    def load(self, model_path: str, dtype=torch.bfloat16, device="cpu", **kwargs):
        # Load Qwen3 text encoder from model subfolder
        # Load FLUX.2 transformer
        # Load VAE with batch norm
        # Return Flux2KleinModel instance
```

**UPDATE** factory:
```python
def create_model_loader(model_type):
    # ... existing ...
    elif model_type.is_flux_2_klein():
        return Flux2KleinModelLoader()
```

---

#### 4. Forward Pass (`eritrainer/training/predict.py`)

**ADD** predictor for FLUX.2:
```python
def predict_flux2_klein(model, batch, config):
    """Forward pass for FLUX.2 Klein models."""
    latents = batch["latents"]

    # 1. Patchify if needed (32 → 128 channels)
    if latents.shape[1] == 32:
        latents = model.patchify_latents(latents)

    # 2. Normalize with batch norm stats
    latents = model.normalize_latents(latents)

    # 3. Pack for transformer
    packed, img_ids = model.pack_latents(latents)
    txt, txt_ids = model.pack_text(batch["prompt_embeds"])

    # 4. Sample timesteps [0, 1]
    timesteps = torch.rand(batch_size, device=device, dtype=dtype)

    # 5. Add noise (flow matching)
    noisy = (1 - t) * latents + t * noise

    # 6. Forward pass (NO guidance for Klein)
    pred = model.transformer(
        hidden_states=noisy,
        encoder_hidden_states=txt,
        timestep=timesteps,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=None,  # Klein has no guidance
    ).sample

    # 7. Target is velocity: noise - latents
    target = noise - latents

    return pred, target
```

---

#### 5. Model Setup (`eritrainer/components/model_setup.py`)

**ADD** setup for FLUX.2:
```python
def setup_flux2_klein(model, config):
    """Setup FLUX.2 Klein for training."""
    # Freeze VAE
    model.vae.requires_grad_(False)
    model.vae.eval()

    # Freeze text encoder (Qwen3)
    model.text_encoder.requires_grad_(False)
    model.text_encoder.eval()

    if config.training_method == TrainingMethod.LORA:
        apply_lora(model.transformer, config)
    else:
        model.transformer.requires_grad_(True)
        model.transformer.train()

    # Enable block swap if configured
    if config.blocks_to_swap > 0:
        max_blocks = 24 if model.is_klein_4b() else 31
        model.enable_block_swap(min(config.blocks_to_swap, max_blocks))
```

---

### Phase 2: Memory Optimization

#### Block Swapping Configuration

| Model | Total Blocks | Max Swappable | Recommended (24GB) |
|-------|--------------|---------------|-------------------|
| klein-4b | 25 | 24 | 12 |
| klein-9b | 32 | 31 | 18 |

---

### Phase 3: Text Encoder Integration

#### Qwen3 Embedding Stacking

From SimpleTuner (`model.py:527-579`):
```python
# Klein uses layers [9, 18, 27]
OUTPUT_LAYERS = (9, 18, 27)

def encode_prompt_qwen3(prompt, tokenizer, text_encoder, max_length=512):
    tokens = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )

    # Stack layers [9, 18, 27]
    stacked = torch.stack([
        output.hidden_states[9],
        output.hidden_states[18],
        output.hidden_states[27],
    ], dim=1)  # [B, 3, L, D]

    # Reshape to [B, L, 3*D]
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
        stacked.shape[0], stacked.shape[2], -1
    )  # [B, L, 3*D]

    return prompt_embeds  # [B, 512, 12288] for 9B or [B, 512, 7680] for 4B
```

---

### Phase 4: VAE Processing

#### Patchify Latents (Pixel Shuffle)

From SimpleTuner (`model.py:282-293`):
```python
def patchify_latents(latents: Tensor) -> Tensor:
    """
    Pixel-shuffle latents: (B, 32, H, W) → (B, 128, H/2, W/2)

    This matches FLUX.2's VAE output format where 32 channels
    are shuffled into 128 channels with 2x2 spatial reduction.
    """
    b, c, h, w = latents.shape
    assert c == 32, f"Expected 32 channels, got {c}"

    # Reshape: (B, 32, H, W) → (B, 32, H/2, 2, W/2, 2)
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)

    # Permute: (B, 32, H/2, 2, W/2, 2) → (B, 32, 2, 2, H/2, W/2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)

    # Flatten channels: (B, 32*2*2, H/2, W/2) = (B, 128, H/2, W/2)
    latents = latents.reshape(b, c * 4, h // 2, w // 2)

    return latents
```

#### Latent Normalization (Batch Norm Stats)

From SimpleTuner (`model.py:295-303`):
```python
def normalize_latents(latents: Tensor, vae: VAE) -> Tensor:
    """Apply VAE batch normalization statistics."""
    if not hasattr(vae, "bn"):
        return latents

    bn = vae.bn  # BatchNorm2d with running stats
    mean = bn.running_mean.view(1, -1, 1, 1)
    var = bn.running_var.view(1, -1, 1, 1)
    eps = bn.eps

    return (latents - mean) / torch.sqrt(var + eps)
```

---

## LoRA Configuration

### Target Modules (from SimpleTuner)

**Default targets** (`model.py:95-122`):
```python
FLUX2_LORA_TARGETS = [
    # Double stream attention
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    # Single stream (fused QKV + MLP)
    "attn.to_qkv_mlp_proj",
]
```

**LyCORIS targets**:
```python
FLUX2_LYCORIS_TARGETS = [
    "Flux2TransformerBlock",
    "Flux2SingleTransformerBlock",
]
```

---

## Configuration Examples

### LoRA Training (Klein 4B)
```yaml
model_type: flux_2_klein_4b
base_model: /path/to/flux2-klein-4b
training_method: lora

lora:
  rank: 16
  alpha: 16
  target_modules:
    - "attn.to_q"
    - "attn.to_k"
    - "attn.to_v"
    - "attn.to_out.0"
    - "attn.to_qkv_mlp_proj"

training:
  batch_size: 1
  gradient_accumulation: 4
  learning_rate: 1e-4
  epochs: 10

memory:
  gradient_checkpointing: on
  blocks_to_swap: 12
```

### Full Fine-tune (Klein 4B Base)
```yaml
model_type: flux_2_klein_4b_base
base_model: /path/to/flux2-klein-base-4b
training_method: fine_tune

training:
  batch_size: 1
  gradient_accumulation: 8
  learning_rate: 1e-5
  epochs: 50

memory:
  gradient_checkpointing: on
  blocks_to_swap: 18
  quantization: int8
```

---

## VRAM Requirements

| Model | Training Type | Min VRAM | Recommended | With Optimizations |
|-------|---------------|----------|-------------|-------------------|
| klein-4b | LoRA | ~13GB | 16GB | 12GB (block swap) |
| klein-4b | Full | ~18GB | 24GB | 16GB (quantize + swap) |
| klein-9b | LoRA | ~22GB | 24GB | 18GB (block swap) |
| klein-9b | Full | ~35GB | 48GB | 24GB (quantize + swap) |

---

## Implementation Checklist

### Phase 1: Core (Required)
- [ ] Add `FLUX_2_KLEIN_*` to `ModelType` enum
- [ ] Create `Flux2KleinModel` class
- [ ] Create `Flux2KleinModelLoader` class
- [ ] Add predictor for FLUX.2 Klein
- [ ] Add setup for FLUX.2 Klein
- [ ] Update `create_model_loader()` factory

### Phase 2: Data Pipeline
- [ ] Qwen3 text encoder caching
- [ ] VAE latent caching with patchification
- [ ] Aspect ratio bucket alignment (16x)

### Phase 3: Memory Optimization
- [ ] Block swap configuration for Klein
- [ ] Gradient checkpointing support
- [ ] INT8 quantization support

### Phase 4: Testing
- [ ] Model loading test
- [ ] Forward pass test
- [ ] LoRA injection test
- [ ] End-to-end training test

### Phase 5: Optional Enhancements
- [ ] TREAD token routing
- [ ] Reference image conditioning
- [ ] Distilled → Base training pathway

---

## References

- SimpleTuner FLUX.2: `/home/alex/SimpleTuner/simpletuner/helpers/models/flux2/`
- SimpleTuner Docs: `/home/alex/SimpleTuner/documentation/quickstart/FLUX2.md`
- EriTrainer Flux: `/home/alex/OneTrainer/eritrainer/models/flux.py`
- EriTrainer Enums: `/home/alex/OneTrainer/eritrainer/core/enums.py`
