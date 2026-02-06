# OneTrainer TODO / Possible Features

## Memory Optimization

### RamTorch Integration
- **URL**: https://github.com/lodestone-rock/RamTorch
- **Description**: CPU-GPU hybrid training that stores parameters in CPU RAM and transfers to GPU on-demand
- **Key Features**:
  - Memory-Efficient Linear Layers: Parameters stored on CPU with on-demand GPU transfer
  - Asynchronous CUDA Streams: Overlaps computation with data transfer for minimal latency
  - ZeRO-1: Optimizer state sharding across multiple GPUs
  - ZeRO-2: Gradient sharding with automatic reduction
  - Shared CPU Memory: Multi-GPU workers share the same CPU tensor storage
  - Drop-in Replacement: Compatible with existing PyTorch code
- **Installation**: `pip install ramtorch`
- **Potential Use Case**: Training large models (e.g., Qwen 6B, Z-Image) on limited VRAM GPUs
- **Priority**: Medium
- **Added**: 2026-01-02

#### SimpleTuner Implementation Reference
SimpleTuner uses RamTorch via `--ramtorch` flag. Key integration points:

1. **Replace Linear Layers**:
```python
from ramtorch.helpers import replace_linear_with_ramtorch
replace_linear_with_ramtorch(model, device="cuda")
# Or use SimpleTuner's wrapper:
from simpletuner.helpers.utils import ramtorch as ramtorch_utils
ramtorch_utils.replace_linear_layers_with_ramtorch(module, device=device, target_patterns=patterns)
```

2. **Mark RamTorch Parameters** (for DDP ignore):
```python
setattr(param, "is_ramtorch", True)
```

3. **Move Non-RamTorch Layers to GPU** (Embeddings, LayerNorm need GPU):
```python
ramtorch_utils.move_embeddings_to_device(module, device)
```

4. **PEFT/LoRA Integration** - Custom wrapper ensures LoRA weights on GPU during forward:
```python
class RamTorchPeftLinear(PeftLinear):
    def forward(self, x):
        self._ensure_lora_on_device(x.device)
        return super().forward(x)
```

5. **DDP Integration**:
```python
ramtorch_utils.mark_ddp_ignore_params(module)  # Exclude RamTorch params from DDP sync
```

6. **ZeRO Utils**:
```python
from ramtorch.zero1 import broadcast_zero_params, create_zero_param_groups
from ramtorch.zero2 import setup_grad_sharding_hooks
```

#### Implementation Notes for OneTrainer
- NOT compatible with group_offload (choose one or the other)
- Requires moving embedding/normalization layers to GPU separately
- Need to patch PEFT for LoRA compatibility with RamTorch Linear
- Consider adding `--ramtorch` flag to TrainConfig
- Target models: Z-Image (6B), Qwen-based models, any large transformer

---

## Musubi Block Swap Implementation (COMPLETED)

### Completed
1. ✅ Created `modules/util/musubi_block_swap.py` - MusubiBlockSwapManager helper with forward hooks support
2. ✅ Updated K5 `models/kandinsky-5-code/kandinsky/models/dit.py` - Added `musubi_manager` attribute and training block swap in forward()
3. ✅ Updated `modules/modelSetup/Kandinsky5LoRASetup.py` - Auto-enables Musubi with ~half visual blocks swapped
4. ✅ Updated ZImage:
   - `modules/model/ZImageModel.py` - Added `musubi_manager` attribute
   - `modules/modelSetup/BaseZImageSetup.py` - Added Musubi setup using forward hooks
5. ✅ Updated Qwen Image:
   - `modules/model/QwenModel.py` - Added `musubi_manager` attribute
   - `modules/model/QwenImageEditModel.py` - Added `musubi_manager` attribute
   - `modules/modelSetup/BaseQwenSetup.py` - Added Musubi setup using forward hooks
6. ✅ Added UI:
   - `modules/util/config/TrainConfig.py` - Added `musubi_blocks_to_swap` config option
   - `modules/ui/OffloadingWindow.py` - Added "Block Swap (Training)" entry field

### How it works
- **Forward hooks**: For ZImage/Qwen (diffusers models), uses `register_forward_pre_hook` and `register_forward_hook` to stream blocks in/out
- **Backward hooks**: Uses `register_full_backward_pre_hook` and `register_full_backward_hook` for gradient computation
- **K5 manual swap**: K5 has control over forward loop, so uses explicit `stream_in()`/`stream_out()` calls
- **Auto-enable**: When `musubi_blocks_to_swap=0` and gradient checkpointing is on, auto-swaps ~half the blocks

### Config
- `musubi_blocks_to_swap`: 0 = auto (half blocks), or specify exact count
- Access via UI: Training tab → Gradient checkpointing advanced button → Block Swap (Training)

### Git State
```
HEAD:   9c417c8 Add Kandinsky 5 LoRA training with block swap inference
Stash:  stash@{0} WIP: UI and other changes before ZImage block swap
```

### Rollback if needed
```bash
git checkout 9c417c8 -- modules/modelSetup/*ZImage* modules/modelSetup/*Qwen* modules/model/QwenImageEditModel.py modules/model/QwenModel.py modules/model/ZImageModel.py
```
