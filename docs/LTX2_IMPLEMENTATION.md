# LTX-2 Video Model Training Implementation for OneTrainer

## Overview

This document describes the implementation of LTX-2 (Lightricks Text-to-Video 2) training support in OneTrainer. LTX-2 is a 19B parameter video generation model using flow matching.

## Model Specifications

| Component | Specification |
|-----------|---------------|
| **Transformer** | 19B parameters, 48 layers, 32 attention heads |
| **Attention Head Dim** | 128 |
| **Cross Attention Dim** | 4096 |
| **VAE Spatial Compression** | 32x |
| **VAE Temporal Compression** | 8x |
| **Text Encoder** | Gemma 3 (optional, 12B recommended) |
| **Training Method** | Flow Matching (velocity target) |
| **Supported Training** | LoRA (full fine-tune not recommended) |

## Original Model Sources

### LTX-2 Core Package
- **Repository**: https://github.com/Lightricks/LTX-2
- **Package**: `ltx-core` (installed from `packages/ltx-core`)
- **License**: LTX-2 Community License

### Model Weights
- **FP8 Checkpoint**: `ltx-2-19b-dev-fp8.safetensors`
- **Download**: Available from Lightricks or Hugging Face
- **Size**: ~19GB (FP8 quantized)

### Text Encoder (Optional)
- **Model**: Gemma 3 12B Instruct
- **Path**: Local directory with Gemma model files
- **Note**: Text encoder is optional; can train without it

## Installation

### 1. Install ltx-core Package

```bash
cd /path/to/LTX-2/packages/ltx-core
pip install -e .
```

Or install to OneTrainer venv:
```bash
/path/to/OneTrainer/venv/bin/pip install -e /path/to/LTX-2/packages/ltx-core
```

### 2. Verify Installation

```python
from ltx_core.model.transformer import LTXModel, LTXModelConfigurator
from ltx_core.model.video_vae import VideoEncoder, VideoDecoder
print("ltx-core installed successfully")
```

## Architecture Overview

### File Structure

```
OneTrainer/
├── modules/
│   ├── model/
│   │   └── LTX2Model.py              # Model container class
│   ├── modelLoader/
│   │   └── LTX2ModelLoader.py        # Model loading with meta device
│   ├── modelSetup/
│   │   ├── BaseLTX2Setup.py          # Base training setup
│   │   └── LTX2LoRASetup.py          # LoRA-specific setup
│   ├── modelSaver/
│   │   └── LTX2FineTuneModelSaver.py # Checkpoint saving
│   ├── modelSampler/
│   │   └── LTX2Sampler.py            # Inference sampling
│   └── dataLoader/
│       └── LTX2BaseDataLoader.py     # Video data loading
├── training_presets/
│   └── ltx2 LoRA.json                # Training preset
└── model_specs/
    └── ltx2-lora.json                # Model specification
```

### Key Components

#### 1. LTX2Model (modules/model/LTX2Model.py)

Container class holding all model components:

```python
class LTX2Model(BaseModel):
    # Core components
    transformer: LTXModel           # 19B transformer from ltx-core
    vae_encoder: VideoEncoder       # Video VAE encoder (wrapped)
    vae_decoder: VideoDecoder       # Video VAE decoder (wrapped)
    text_encoder: AVGemmaTextEncoderModel  # Optional Gemma encoder
    noise_scheduler: LTX2Scheduler  # Flow matching scheduler
    patchifier: VideoLatentPatchifier

    # Training components
    transformer_lora: LoRAModuleWrapper  # LoRA adapter
    transformer_offload_conductor: LayerOffloadConductor  # Layer offloading
```

#### 2. LTX2ModelLoader (modules/modelLoader/LTX2ModelLoader.py)

Loads model using meta device trick for memory efficiency:

```python
def _load_transformer(self, checkpoint_path, dtype, quantize_int8):
    # Load config from safetensors metadata
    with safe_open(checkpoint_path, framework='pt') as f:
        metadata = f.metadata()
    config = json.loads(metadata['config'])

    # Create model on meta device (instant, no memory)
    with torch.device("meta"):
        transformer = LTXModelConfigurator.from_config(config)

    # Load state dict with prefix stripping
    full_state_dict = load_file(checkpoint_path)
    transformer_state_dict = {
        k[len("model.diffusion_model."):]: v
        for k, v in full_state_dict.items()
        if k.startswith("model.diffusion_model.")
    }

    # Load with assign=True for meta->real tensor transfer
    transformer.load_state_dict(transformer_state_dict, strict=False, assign=True)
    return transformer
```

#### 3. VAE Wrappers

The ltx-core VAE uses `forward()` directly, but OneTrainer's pipeline expects `encode()`/`decode()` methods returning distribution objects:

```python
class LTX2VAEEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self._encoder = encoder

    def encode(self, x):
        latent = self._encoder(x)
        return LTX2LatentDistribution(latent)

class LTX2LatentDistribution:
    """Mimics diffusers DiagonalGaussianDistribution."""

    def __init__(self, latent_tensor):
        self._latent_tensor = latent_tensor
        self.latent_dist = self  # For EncodeVAE compatibility

    def sample(self):
        return self._latent_tensor

    def mean(self):
        return self._latent_tensor

    def mode(self):
        return self._latent_tensor
```

#### 4. Layer Offloading (modules/util/checkpointing_util.py)

Enables training 19B model on 24GB GPU:

```python
def enable_checkpointing_for_ltx2_transformer(
    model: nn.Module,
    config: TrainConfig,
) -> LayerOffloadConductor:
    """Enable gradient checkpointing with layer offload for LTX-2."""
    return enable_checkpointing(model, config, config.compile, [
        (model.transformer_blocks, ["hidden_states", "encoder_hidden_states"]),
    ])
```

## Training Configuration

### Preset File: `training_presets/ltx2 LoRA.json`

```json
{
    "__version": 5,
    "base_model_name": "/path/to/ltx-2-19b-dev-fp8.safetensors",
    "text_encoder_model": "/path/to/gemma-3-12b-it",
    "model_type": "LTX_2",
    "training_method": "LORA",
    "output_model_destination": "/path/to/output/ltx2-lora.safetensors",
    "output_model_format": "SAFETENSORS",

    "workspace_dir": "/path/to/workspace",
    "cache_dir": "/path/to/cache",
    "tensorboard_log_dir": "/path/to/tensorboard",

    "batch_size": 1,
    "gradient_checkpointing": "CPU_OFFLOADED",
    "layer_offload_fraction": 0.85,
    "dataloader_threads": 1,

    "learning_rate": 0.0001,
    "resolution": "512",
    "timestep_distribution": "LOGIT_NORMAL",
    "train_dtype": "BFLOAT_16",

    "peft_type": "LORA",
    "lora_rank": 32,
    "lora_alpha": 32,

    "epochs": 1,
    "backup_after": 100,
    "backup_after_unit": "STEP",
    "save_after": 500,
    "save_after_unit": "STEP",

    "sample_after": 0,
    "sample_after_unit": "NEVER",

    "transformer": {
        "train": true,
        "weight_dtype": "FLOAT_8"
    },
    "text_encoder": {
        "train": false,
        "weight_dtype": "FLOAT_8"
    },

    "concepts": [
        {
            "name": "my_dataset",
            "enabled": true,
            "path": "/path/to/videos",
            "prompt_path": "/path/to/dataset.json",
            "include_subdirectories": false
        }
    ]
}
```

### Key Configuration Notes

| Setting | Recommended | Notes |
|---------|-------------|-------|
| `__version` | 5 | Required to skip migration that breaks CPU_OFFLOADED |
| `gradient_checkpointing` | "CPU_OFFLOADED" | Essential for 24GB GPU |
| `layer_offload_fraction` | 0.85 | Offload 85% of layers to CPU |
| `batch_size` | 1 | Memory constrained |
| `lora_rank` | 32 | Balance quality/memory |
| `weight_dtype` | "FLOAT_8" | Model is FP8 quantized |

## Using OneTrainer Web UI

### 1. Launch Web UI

```bash
cd /path/to/OneTrainer
./venv/bin/python scripts/run_ui.py
```

Access at: http://localhost:3000

### 2. Configure Training

1. **Model Selection**
   - Model Type: `LTX_2`
   - Base Model: Path to `ltx-2-19b-dev-fp8.safetensors`
   - Text Encoder (optional): Path to Gemma model directory

2. **Training Method**
   - Method: `LORA`
   - LoRA Rank: 32
   - LoRA Alpha: 32

3. **Memory Optimization** (Critical for 24GB GPU)
   - Gradient Checkpointing: `CPU_OFFLOADED`
   - Layer Offload Fraction: 0.85
   - Batch Size: 1

4. **Dataset**
   - Add concept with video directory
   - Create `dataset.json` with prompts:
   ```json
   [
     {"file_path": "video1.mp4", "prompt": "description of video 1"},
     {"file_path": "video2.mp4", "prompt": "description of video 2"}
   ]
   ```

5. **Output**
   - Output Format: SAFETENSORS
   - Output Path: Where to save LoRA

### 3. Video Requirements

- **Format**: MP4, MOV, or image sequences
- **Frame Count**: Must be 1 + 8*k (e.g., 9, 17, 25, 33 frames)
- **Resolution**: 512x512 recommended for training
- **Duration**: Short clips (1-4 seconds)

## Code Port Details

### From ltx-core to OneTrainer

| ltx-core Component | OneTrainer Adaptation |
|-------------------|----------------------|
| `LTXModelConfigurator.from_config()` | Used with `torch.device("meta")` for fast loading |
| `VideoEncoderConfigurator.from_config()` | Wrapped in `LTX2VAEEncoderWrapper` |
| `VideoDecoderConfigurator.from_config()` | Wrapped in `LTX2VAEDecoderWrapper` |
| `LTX2Scheduler` | Used directly for flow matching |
| `VideoLatentPatchifier` | Used for latent space operations |

### State Dict Key Mapping

The safetensors file uses prefixed keys:
```
model.diffusion_model.* -> transformer.*
vae.encoder.* -> encoder.*
vae.decoder.* -> decoder.*
vae.per_channel_statistics.* -> per_channel_statistics.*
```

### Config from Metadata

The safetensors file contains model config in metadata:
```python
with safe_open(checkpoint_path, framework='pt') as f:
    metadata = f.metadata()
    config = json.loads(metadata['config'])
    # config contains: transformer, vae, scheduler, audio_vae, vocoder
```

## Memory Requirements

| Configuration | VRAM Required | Notes |
|--------------|---------------|-------|
| Full model (no offload) | ~40GB | Not practical |
| 85% layer offload | ~6GB | Recommended |
| 50% layer offload | ~12GB | Faster but needs more VRAM |
| With text encoder | +4-8GB | Gemma 12B adds memory |

### Estimated RAM Usage
- System RAM: 64GB recommended
- With 85% offload: ~50GB system RAM used

## Troubleshooting

### Common Issues

1. **CUDA OOM**
   - Increase `layer_offload_fraction` (max 0.95)
   - Reduce batch size to 1
   - Don't load text encoder if not needed

2. **"Invalid number of frames"**
   - Video must have 1 + 8*k frames
   - Use ffmpeg to trim: `ffmpeg -i input.mp4 -frames:v 17 output.mp4`

3. **Config migration breaks CPU_OFFLOADED**
   - Add `"__version": 5` to preset JSON

4. **VAE encode error**
   - Ensure using wrapped encoder (`LTX2VAEEncoderWrapper`)
   - Check `latent_dist` attribute exists

5. **Slow loading**
   - Meta device trick should make loading fast (~30 seconds)
   - If slow, check disk I/O

## Example Training Command

```bash
cd /path/to/OneTrainer
./venv/bin/python scripts/train.py \
    --config-path "training_presets/ltx2 LoRA.json"
```

## Output LoRA Usage

The trained LoRA can be used with:
- ComfyUI (with LTX-2 nodes)
- Original LTX-2 inference code
- Any system supporting safetensors LoRA format

```python
# Example loading trained LoRA
from safetensors.torch import load_file
lora_state_dict = load_file("ltx2-my-lora.safetensors")
# Apply to model using standard LoRA merging
```

## References

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [OneTrainer GitHub](https://github.com/Nerogar/OneTrainer)
- [Flow Matching Paper](https://arxiv.org/abs/2210.02747)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## Version History

- **v1.0** (2025-01): Initial implementation
  - LoRA training support
  - Meta device loading
  - Layer offloading for 24GB GPUs
  - VAE wrappers for pipeline compatibility
