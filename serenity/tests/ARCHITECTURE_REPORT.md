# Serenity Architecture Report

## MODEL LAYER (known from FLUX.2 implementation)

| File | Classes/Functions | Purpose |
|------|-------------------|---------|
| `serenity/models/flux2_klein.py` | `Flux2KleinModel`, `Flux2KleinModelLoader` | FLUX.2 Klein model wrapper and loader |
| `serenity/models/base.py` | `BaseModel` (ABC) | Abstract base for all models |
| `serenity/components/model_setup.py` | `Flux2KleinModelSetup`, `create_model_setup()` | Model initialization for training |
| `serenity/training/predict.py` | `Flux2KleinPredictor`, `get_predictor()` | Forward pass implementations |
| `serenity/components/model_loader.py` | `create_model_loader()` | Model loader factory |
| `serenity/core/enums.py` | `ModelType.FLUX_2_KLEIN_*`, `is_flux_2_klein()` | Model type enums and helpers |

## MEMORY LAYER (discovered)

### Core Memory Management (`serenity/memory/`)

| File | Classes/Functions | Purpose |
|------|-------------------|---------|
| `manager.py` | `MemoryManager`, `create_memory_manager()` | **High-level API** - unified memory management |
| `conductor.py` | `LayerOffloadConductor`, `LayerOffloadStrategy` | Layer-by-layer GPU/CPU offloading |
| `allocators.py` | `StaticTensorAllocator`, `StaticLayerAllocator`, `StaticActivationAllocator` | Pre-allocated memory pools |
| `checkpointing.py` | `CheckpointLayer`, `OffloadCheckpointLayer`, `enable_checkpointing_for_transformer()` | Gradient checkpointing |
| `sync.py` | `SyncEvent`, `torch_gc()`, `tensors_to_device_()` | Device sync utilities |

### Block Swapping (`serenity/training/block_swap.py`)

| Class | Purpose |
|-------|---------|
| `BlockSwapOffloader` | Swap transformer blocks between GPU/CPU during forward/backward |
| `ModelOffloader` | High-level wrapper that wraps block forward methods |
| `setup_block_swapping()` | Factory function |

### Other Memory Components

| File | Classes/Functions |
|------|-------------------|
| `training/gradient.py` | `setup_gradient_checkpointing()`, gradient utilities |
| `training/offload.py` | `enable_activation_offloading_for_transformer()` |

## PIPELINE LAYER (discovered)

### Main Trainer (`serenity/core/trainer.py`)

| Class | Key Methods | Purpose |
|-------|-------------|---------|
| `Trainer` | `start()`, `train()`, `_training_step()` | Main training orchestrator |

Key attributes:
- `self.model: BaseModel` - the model wrapper
- `self.model_setup: ModelSetup` - model initialization
- `self.memory_manager: MemoryManager` - unified memory management (line 824)
- `self.activation_conductor` - CPU activation offloading
- `self.offload_context` - for forward context

### Data Pipeline (`serenity/pipeline/`)

| File | Classes/Functions | Purpose |
|------|-------------------|---------|
| `dataset.py` | `EriDataset`, `create_training_dataset()` | PyTorch Dataset with cache integration |
| `cache.py` | `CacheManager`, `LatentCacher`, `TextCacher` | Disk-based caching |
| `staged_loader.py` | `run_staged_caching()` | VRAM-efficient staged caching |
| `buckets.py` | `BucketManager`, `create_buckets()` | Aspect ratio bucketing |
| `concepts.py` | `ConceptConfig`, `TrainingItem`, `scan_concepts()` | Concept loading |

## INTERFACES/PROTOCOLS

### BaseModel (`models/base.py:153`)
```python
class BaseModel(ABC):
    transformer: Optional[nn.Module]
    vae: nn.Module
    text_encoder: nn.Module
    layer_offload_conductor: Optional[LayerOffloadConductor]  # Line 196
```

### MemoryManager (`memory/manager.py:31`)
```python
class MemoryManager:
    def setup_optimizations(self)  # Create conductors, setup checkpointing
    def move_to_train_device(self)  # Move with offloading strategy
    def forward_context(self)  # Context manager for forward/backward
```

### LayerOffloadConductor (`memory/conductor.py:146`)
```python
class LayerOffloadConductor:
    def add_layer(layer, param_indices)  # Register layer
    def to(device)  # Move with strategy
    def start_forward(keep_graph)  # Start of forward pass
    def before_layer(layer_index, call_index, activations)  # Pre-layer hook
    def after_layer(layer_index, call_index, activations)  # Post-layer hook
```

## HOW THEY CONNECT

### Model → Memory
- `BaseModel` stores optional `layer_offload_conductor` reference (`base.py:196`)
- `MemoryManager` creates conductor and attaches to model (`manager.py:111-112`):
  ```python
  if hasattr(self.model, 'transformer_offload_conductor'):
      self.model.transformer_offload_conductor = self.transformer_conductor
  ```

### Memory → Pipeline
- `Trainer` holds `memory_manager` (`trainer.py:824`):
  ```python
  self.memory_manager = None  # Unified memory management (new architecture)
  ```
- Training step uses `offload_context.forward_context()` (`trainer.py:1987-1991`):
  ```python
  if self.offload_context:
      with self.offload_context.forward_context():
          loss = self.model_setup.forward_pass(self.model, batch)
  ```

### Pipeline → Model
- `Trainer.start()` calls `create_model_setup()` and `create_model_loader()`
- `Trainer._training_step()` calls `model_setup.forward_pass(model, batch)`
- `model_setup.forward_pass()` calls model-specific predict logic

## BOUNDARY CONCERNS

### Potential Coupling Issues

1. **trainer.py:1987-1991** - Trainer directly accesses `offload_context.forward_context()`
   - This is appropriate - trainer orchestrates memory context

2. **Flux2KleinModelSetup.forward_pass()** (`model_setup.py`)
   - Calls model methods like `patchify_latents()`, `normalize_latents()`
   - Direct model access is expected in setup class

3. **BaseModel.layer_offload_conductor** (`base.py:196`)
   - Model has reference to memory conductor
   - This is a controlled interface point

### Clean Separations Observed

- Memory layer (`memory/`) does NOT import specific model classes
- Model layer does NOT call `torch.cuda.empty_cache()` directly (uses `torch_gc()`)
- Pipeline uses setup/predictor interfaces, not model internals

## KEY FINDINGS

### Two Memory Offloading Systems

1. **LayerOffloadConductor** (`memory/conductor.py`)
   - Async CUDA stream transfers
   - Static memory allocation
   - Activation offloading during gradient checkpointing
   - Used via `MemoryManager`

2. **BlockSwapOffloader** (`training/block_swap.py`)
   - Simpler block-level swapping
   - Thread pool for async transfers
   - Forward hooks for automatic offloading
   - Used directly by model setup classes

### Flux2KleinModel Block Swap

From `flux2_klein.py` and `model_setup.py`:
```python
# In Flux2KleinModelSetup.setup():
if config.blocks_to_swap > 0:
    variant = "klein-9b" if model.is_klein_9b() else "klein-4b"
    max_blocks = self.MAX_SWAPPABLE_BLOCKS.get(variant, 24)
    model.enable_block_swap(min(config.blocks_to_swap, max_blocks))
```

Block swap limits:
- Klein 4B: 24 max (25 total - 1)
- Klein 9B: 31 max (32 total - 1)

### FLUX.2 Klein Memory Path

1. `Trainer.start()` → `_load_transformer_only()` → `Flux2KleinModelLoader.load()`
2. `create_model_setup()` → `Flux2KleinModelSetup`
3. `Flux2KleinModelSetup.setup()` → enables block swap, grad checkpointing
4. `Trainer._training_step()` → `Flux2KleinModelSetup.forward_pass()`
5. Forward pass uses flow matching with patchification/normalization

## FILES TO TEST

### Model Layer
- `serenity/models/flux2_klein.py`
- `serenity/components/model_setup.py` (Flux2KleinModelSetup)
- `serenity/training/predict.py` (Flux2KleinPredictor)

### Memory Layer
- `serenity/memory/manager.py`
- `serenity/memory/conductor.py`
- `serenity/training/block_swap.py`

### Pipeline Layer
- `serenity/core/trainer.py`
- `serenity/pipeline/dataset.py`
- `serenity/pipeline/cache.py`
