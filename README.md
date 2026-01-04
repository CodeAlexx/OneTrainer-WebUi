# OneTrainer Web UI

A modern web-based interface for OneTrainer with real-time training monitoring, inference, and enhanced model support.

## Screenshots

### Dashboard
![Dashboard](docs/screenshots/dashboard.png)
Real-time training dashboard with live loss charts (raw + smoothed), GPU monitoring showing temperature, VRAM usage, and power draw. Training console displays step-by-step progress with loss values, speed metrics, and ETA countdown.

### Inference UI
![Inference](docs/screenshots/inference.png)
Full-featured image generation interface with core parameters (steps, CFG, seed), resolution presets, sampler selection, init image support, refine/upscale options, ControlNet integration, video settings, FreeU, and LoRA stacking.

### Training Concepts
![Concepts](docs/screenshots/concepts.png)
Card-based concept manager for organizing training data. Each concept card shows a preview with enable/disable toggles, allowing quick activation of specific training subjects.

### Sample Browser
![Samples](docs/screenshots/samples.png)
Tree view sample organization with expandable folders. Select any sample to preview images with their captions. Supports multiple dataset workspaces.

### LoRA / Adapters Configuration
![LoRA Adapters](docs/screenshots/lora-adapters.png)
Configure PEFT adapters with LoRA, DoRA, and LyCORIS support. Set rank, alpha, dropout, and target layers with custom regex patterns for precise layer targeting.

### Diffusion 4K & Resolution Presets
![Diffusion 4K](docs/screenshots/diffusion4k.png)
Enable Diffusion-4K Wavelet Loss for high-frequency detail preservation (arXiv:2503.18352). Quick resolution presets for 1024, 2048, and 4096 training.

### Aspect Ratio Bucketing
![Buckets](docs/screenshots/buckets.png)
Advanced bucketing configuration with aspect ratio presets, bucket balancing modes (oversample/undersample), quantization settings, and real-time config summary preview.

### Qwen VL Captioner
![Captioner](docs/screenshots/captioner.png)
Automatic image captioning using Qwen2-VL-7B. Customize prompts, enable summary or one-sentence modes, control max tokens, and batch process entire folders.

### Video Editor
![Video Editor](docs/screenshots/video-editor.png)
Timeline-based video editor with multi-track support, playback controls, In/Out point selection, clip properties, effects panel, and video export.

### Preset Card Selector
![Preset Selector](docs/screenshots/preset-selector.png)
Visual preset browser with model type filtering (Kandinsky, Qwen, Flux, SDXL, Chroma, etc.), search, tags for LoRA/Finetune/VRAM requirements, and favorites.

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
