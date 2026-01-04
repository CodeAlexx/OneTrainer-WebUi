# OneTrainer Web UI

A modern web-based interface for OneTrainer with real-time training monitoring, inference, and enhanced model support.

## New Features

### Web Interface
- **Real-time Dashboard** - Live loss charts (raw + smoothed), GPU monitoring (temperature, VRAM, power), training console, ETA tracking
- **Training Queue** - Queue and manage multiple training jobs
- **Preset Card System** - Visual preset selector with model type filtering and tags
- **Concepts Manager** - Card-based training concepts with enable/disable toggles
- **Sample Browser** - Tree view sample organization with image preview
- **Inference UI** - Full-featured generation interface with LoRA stacking, ControlNet, FreeU, init image, refine/upscale

### Utility Tools
- **Qwen VL Captioner** - Automatic captioning with Qwen2-VL-7B, custom prompts, summary/one-sentence modes, token control
- **Model Conversion** - Convert between model formats
- **Mask Generation** - Create training masks
- **Dataset Tools** - Video preparation, frame extraction
- **Image Tools** - Image processing utilities
- **Video Editor** - Edit and process video clips for training

### Training Enhancements
- **Block Swapping** - Memory-efficient training via layer offloading for large models
- **Partial RAM/Torch Offload** - Hybrid CPU/GPU memory management
- **Local WandB Integration** - Offline experiment tracking
- **Diffusion-4K Wavelet Loss** - High-frequency detail preservation (arXiv:2503.18352)
- **4K Resolution Presets** - Quick selection for 1024/2048/4096 training
- **Advanced Bucketing** - Aspect ratio presets, oversampling, bucket balancing with config preview

### Model Support
- **Kandinsky 5** - Text-to-image and text-to-video (Lite/Pro variants)
- **Qwen Image Edit** - Image editing model support
- **LyCORIS 3.4** - Latest LyCORIS training methods
- **DoRA** - Weight decomposition for LoRA adapters
- **Custom Target Layers** - Regex-based layer targeting for adapters

### Inference Features
- **Multi-model Support** - FLUX, SDXL, Qwen, Kandinsky, and more
- **Video Generation** - Video model inference with frame settings
- **LoRA Stacking** - Load multiple LoRAs with individual weights
- **ControlNet** - Guided generation support
- **Variation Seed** - Seed interpolation for variations
- **Resolution Presets** - Quick resolution selection per model type

## Supported Models

| Model | Training | Inference | Video |
|-------|----------|-----------|-------|
| FLUX.1 | ✅ | ✅ | - |
| SDXL | ✅ | ✅ | - |
| SD 1.5/2.x | ✅ | ✅ | - |
| Qwen Image | ✅ | ✅ | - |
| Qwen Image Edit | ✅ | ✅ | - |
| Kandinsky 5 | ✅ | ✅ | ✅ |
| Hunyuan Video | ✅ | ✅ | ✅ |
| Z-Image | ✅ | ✅ | - |
| Chroma | ✅ | ✅ | - |
| PixArt | ✅ | ✅ | - |
| Sana | ✅ | ✅ | - |

## Credits

Based on [OneTrainer](https://github.com/Nerogar/OneTrainer) by Nerogar.
