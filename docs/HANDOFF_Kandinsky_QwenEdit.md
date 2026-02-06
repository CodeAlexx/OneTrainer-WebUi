# Kandinsky 5 & Qwen-Edit Integration Handoff

## Current Status

### Qwen-Edit Training - WORKING
- LoRA training works with `qwen_image_edit_lora.json` preset
- Loss decreasing, samples generating
- Key fix: Musubi block swap disabled when gradient checkpointing is ON (incompatible)

### Kandinsky 5 Training - WORKING
- LoRA training works with `kandinsky5_lite_lora.json` preset
- Uses `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s` from HuggingFace

### Kandinsky 5 Inference - FIXED ✅
- Fixed: Patched `kandinsky/utils.py` to allow `dit_path` override when `conf_path` is also provided
- `app.py` now passes both `conf_path` (for Pro architecture) AND `dit_path` (for custom weights)

---

## Inference App Issues

### File: `/home/alex/OneTrainer/inference_app/backend/app.py`

#### Problem 1: Pro model loads with Lite config
The `_load_kandinsky5_single_file` function passes `dit_path` but no `conf_path`, causing kandinsky-5-code to default to Lite architecture.

**Current code** (line ~1846):
```python
pipe = get_T2V_pipeline(
    device_map=device_map,
    cache_dir=hf_cache,
    dit_path=dit_path,
    offload=True,
)
```

**Fix needed**: Pass correct config for Pro models:
```python
# For Pro models, pass the Pro config
if is_pro:
    config_file = kandinsky_repo / "configs" / "k5_pro_t2v_5s_sft_sd.yaml"
    pipe = get_T2V_pipeline(
        device_map=device_map,
        cache_dir=hf_cache,
        dit_path=dit_path,
        conf_path=str(config_file),
        offload=True,
    )
```

**BUT** this has another problem: the config files have relative paths like `./weights/model/...` that don't exist.

#### Problem 2: Config file paths are relative
The yaml configs in kandinsky-5-code have paths like:
```yaml
model:
  checkpoint_path: ./weights/model/kandinsky5pro_t2v_sft_5s.safetensors
  vae:
    checkpoint_path: ./weights/...
```

**Proper fix**: Need to either:
1. Modify the kandinsky utils to accept both `conf_path` AND `dit_path` override
2. Or create modified configs with absolute paths
3. Or patch the config after loading with OmegaConf

### Available Configs
```
/home/alex/OneTrainer/models/kandinsky-5-code/configs/
├── k5_lite_t2i_sft_hd.yaml      # Lite T2I, 1024px
├── k5_lite_t2v_5s_sft_sd.yaml   # Lite T2V, 512px, 5s
├── k5_pro_t2v_5s_sft_sd.yaml    # Pro T2V, 512px, 5s  <-- Need this for Pro
├── k5_pro_t2v_5s_sft_hd.yaml    # Pro T2V, 1024px, 5s
└── ...
```

### Model Files
- **Pro weights**: `/home/alex/OneTrainer/models/kandinsky-5-video-pro/model/kandinsky5pro_t2v_sft_5s.safetensors`
- **Lite weights**: Downloaded from HuggingFace to `~/.cache/huggingface/hub/`

---

## Kandinsky Pipeline Parameters

### T2V Pipeline (`__call__`):
```python
def __call__(
    self,
    text: str,                    # Prompt
    time_length: int = 5,         # Seconds (0 = image)
    width: int = 768,
    height: int = 512,
    seed: int = None,
    num_steps: int = None,        # Default from config
    guidance_weight: float = None,# Default from config
    scheduler_scale: float = 10.0,
    negative_caption: str = "...",
    expand_prompts: bool = True,  # Uses Qwen to enhance
    save_path: str = None,
    progress: bool = True,
)
```

### T2I Pipeline (`__call__`):
```python
# Same but no time_length, different defaults:
scheduler_scale: float = 3.0,
width: int = 1024,
height: int = 1024,
```

### Resolution Constraints
```python
RESOLUTIONS = {
    512: [(512, 512), (512, 768), (768, 512)],
    1024: [(1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)],
}
```

---

## Files Modified

### Training (DONE)
- `modules/modelSetup/BaseQwenSetup.py` - Disable Musubi when checkpointing ON
- `modules/model/QwenImageEditModel.py` - Add processor to pipeline
- `modules/modelLoader/QwenImageEditModelLoader.py` - Load processor
- `modules/dataLoader/mixin/DataLoaderText2ImageMixin.py` - Video format for cond images
- `training_presets/qwen_image_edit_lora.json` - Preset
- `training_presets/kandinsky5_lite_lora.json` - Preset
- `docs/QwenImageEdit.md` - Training guide

### Web UI (DONE)
- `web_ui/frontend/src/model_constants.ts` - Added Kandinsky, Qwen-Edit, Wan to dropdowns
- `web_ui/frontend/src/components/views/NewJobView.tsx` - Uses shared constants
- `web_ui/frontend/src/components/views/TrainingView.tsx` - Added block swap controls
- `web_ui/frontend/src/components/PresetCardSelector.tsx` - Filter tabs for K5, Qwen-Edit

### Inference App (PARTIAL - needs fix)
- `inference_app/backend/app.py`:
  - Added `KANDINSKY_5_VIDEO_PRO` model type
  - Added `_build_pipeline_kwargs` for Kandinsky-specific params
  - Added resolution auto-adjustment
  - Fixed output handling for Kandinsky
  - **STILL BROKEN**: `_load_kandinsky5_single_file` needs Pro config support

- `inference_app/frontend/src/App.tsx`:
  - Added Kandinsky 5 T2I, T2V Lite, T2V Pro options
  - Added Qwen-Edit option

---

## To Complete Kandinsky Inference

### Option A: Patch kandinsky utils
Modify `/home/alex/OneTrainer/models/kandinsky-5-code/kandinsky/utils.py` to accept both `conf_path` AND override individual paths like `dit_path`.

### Option B: Create runtime config
In `_load_kandinsky5_single_file`, create a modified config on-the-fly:
```python
from omegaconf import OmegaConf

conf = OmegaConf.load(str(config_file))
# Override paths
conf.model.checkpoint_path = model_path
conf.model.vae.checkpoint_path = hf_cache + "/vae"
# etc...
# Then call pipeline with patched conf object instead of conf_path
```

### Option C: Use default pipeline with strict=False
If the kandinsky code supports `strict=False` for weight loading, could work around mismatches.

---

## Testing Commands

```bash
# Test Kandinsky load via API
curl -X POST http://localhost:7860/api/model/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/home/alex/OneTrainer/models/kandinsky-5-video-pro/model/kandinsky5pro_t2v_sft_5s.safetensors", "model_type": "kandinsky_5_video_pro"}'

# Check logs
tail -100 /tmp/inference.log
```
