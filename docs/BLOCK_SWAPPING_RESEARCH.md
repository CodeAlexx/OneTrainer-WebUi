# Block Swapping Research - diffusion-pipe → OneTrainer Integration

## Executive Summary

Block swapping is a memory optimization technique that enables training large transformer models (Wan 14B, HunyuanVideo, Flux) on consumer GPUs with 24GB VRAM. It works by dynamically swapping transformer blocks between GPU and CPU during forward/backward passes, keeping only 2+ blocks on GPU at any time.

**Key Numbers**:
- Wan 14B: 40 transformer blocks → swap 32 → train on 24GB
- HunyuanVideo: 20 double + 40 single blocks → swap accordingly
- Flux: 19 double + 38 single blocks → swap accordingly
- Memory savings: 40-60% VRAM reduction
- Performance cost: 10-30% slower (depends on PCIe bandwidth)

---

## 1. diffusion-pipe Core Implementation

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     ModelOffloader                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ blocks: list[nn.Module]     # All transformer blocks │   │
│  │ num_blocks: int             # Total block count      │   │
│  │ blocks_to_swap: int         # Blocks to offload      │   │
│  │ device: torch.device        # GPU device             │   │
│  │ forward_only: bool          # Inference vs training  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Methods:                                                   │
│  ├── enable_block_swap()     # Initialize swapping         │
│  ├── wait_for_block(idx)     # Ensure block on GPU         │
│  ├── submit_move_blocks_forward(idx)  # Schedule next      │
│  └── prepare_block_devices_before_forward()  # Setup       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Core File: `/home/alex/diffusion-pipe/utils/offloading.py`

**Key Classes**:

```python
class ModelOffloader:
    """
    Manages GPU↔CPU block transfers during forward/backward passes.

    Key Features:
    - CUDA stream-based async transfers
    - Pinned memory for faster PCIe transfers
    - Thread pool for non-blocking operations
    - Backward hooks for training mode
    """

    def __init__(self, name, blocks, num_blocks, blocks_to_swap,
                 use_cuda_graph, device, reentrant_checkpointing,
                 use_pinned_memory=False, debug=False):
        # Validate constraints
        assert blocks_to_swap <= num_blocks - 2, \
            "Must keep at least 2 blocks on GPU"

        # Setup transfer infrastructure
        self.blocks = blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.stream = torch.cuda.Stream(device)

        # Register backward hooks for training
        if not forward_only:
            self._register_backward_hooks()

    def wait_for_block(self, block_idx):
        """Wait for block to be transferred to GPU."""
        # Synchronize with CUDA stream
        # Check if block is on correct device

    def submit_move_blocks_forward(self, block_idx):
        """Schedule next block transfer (pipelined)."""
        # Move current block back to CPU (async)
        # Move next block to GPU (async)
```

### 1.3 Transfer Strategy

**Forward Pass**:
```
Block 0: GPU ──forward──> GPU→CPU (async)
Block 1: CPU→GPU (async) ──forward──> GPU→CPU (async)
Block 2: CPU→GPU (async) ──forward──> GPU→CPU (async)
...
Block N: CPU→GPU ──forward──> stays on GPU (for backward)
```

**Backward Pass** (training only):
```
Block N: GPU ──backward──> stays (grad needed)
Block N-1: CPU→GPU ──backward──> GPU→CPU
Block N-2: CPU→GPU ──backward──> GPU→CPU
...
Block 0: CPU→GPU ──backward──> done
```

### 1.4 Memory Layout

```
GPU Memory:
┌────────────────────────────────────────────────────┐
│  Block K   │  Block K+1  │  Embeddings │  Grads   │
│  (current) │  (prefetch) │  & Output   │  (LoRA)  │
└────────────────────────────────────────────────────┘

CPU Memory (Pinned):
┌─────────────────────────────────────────────────────────────┐
│ Block 0 │ Block 1 │ ... │ Block K-1 │ Block K+2 │ ... │ N-1 │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Model-Specific Implementations

### 2.1 Wan Model (`/home/alex/diffusion-pipe/models/wan/wan.py`)

**Architecture**: 40 transformer blocks (single type)

```python
class WanPipeline(BasePipeline):
    checkpointable_layers = ['WanTransformerLayer']

    def enable_block_swap(self, blocks_to_swap):
        blocks = self.transformer.blocks  # 40 blocks
        assert blocks_to_swap <= len(blocks) - 2

        self.offloader = ModelOffloader(
            'WanBlock', blocks, len(blocks), blocks_to_swap,
            use_cuda_graph=True, device=torch.device('cuda'),
            reentrant_checkpointing=self.config['reentrant_activation_checkpointing']
        )

        # Detach blocks from model, move to CPU
        self.transformer.blocks = None
        self.transformer.to('cuda')  # Move non-block params
        self.transformer.blocks = blocks  # Reattach

        self.prepare_block_swap_training()

    def to_layers(self):
        """Wrap blocks for checkpointing + offloading."""
        layers = [EmbeddingWrapper(...)]
        for i, block in enumerate(self.transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(OutputWrapper(...))
        return layers
```

**Block Wrapper**:
```python
class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # Wait for this block to be on GPU
        self.offloader.wait_for_block(self.block_idx)

        # Forward through block
        output = self.block(...)

        # Schedule next block transfer (pipelined)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return output
```

**Config Example** (`wan_14b_min_vram.toml`):
```toml
blocks_to_swap = 32
activation_checkpointing = 'unsloth'
micro_batch_size_per_gpu = 1
transformer_dtype = 'float8'
```

### 2.2 Flux Model (`/home/alex/diffusion-pipe/models/flux.py`)

**Architecture**: 19 double blocks + 38 single blocks (dual type)

```python
class FluxPipeline(BasePipeline):
    NUM_DOUBLE_BLOCKS = 19
    NUM_SINGLE_BLOCKS = 38

    def __init__(self, config):
        # Create separate offloaders for each block type
        self.offloader_double = ModelOffloader('dummy', [], 0, 0, ...)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, ...)

    def enable_block_swap(self, blocks_to_swap):
        double_blocks = self.transformer.transformer_blocks
        single_blocks = self.transformer.single_transformer_blocks

        # Split blocks_to_swap between types
        # Note: Musubi's formula swaps more single blocks
        double_blocks_to_swap = blocks_to_swap // 2
        single_blocks_to_swap = (blocks_to_swap - double_blocks_to_swap) * 2 + 1

        assert double_blocks_to_swap <= len(double_blocks) - 2
        assert single_blocks_to_swap <= len(single_blocks) - 2

        self.offloader_double = ModelOffloader(
            'DoubleBlock', double_blocks, len(double_blocks),
            double_blocks_to_swap, ...
        )
        self.offloader_single = ModelOffloader(
            'SingleBlock', single_blocks, len(single_blocks),
            single_blocks_to_swap, ...
        )

        # Same detach/reattach pattern
        self.transformer.transformer_blocks = None
        self.transformer.single_transformer_blocks = None
        self.transformer.to('cuda')
        self.transformer.transformer_blocks = double_blocks
        self.transformer.single_transformer_blocks = single_blocks
```

**Wrapper Classes**:
```python
class TransformerWrapper(nn.Module):  # For double blocks
    def forward(self, inputs):
        self.offloader.wait_for_block(self.block_idx)
        encoder_hidden_states, hidden_states = self.block(...)
        self.offloader.submit_move_blocks_forward(self.block_idx)
        return ...

class SingleTransformerWrapper(nn.Module):  # For single blocks
    # Same pattern, different input signature
```

### 2.3 HunyuanVideo (`/home/alex/diffusion-pipe/models/hunyuan_video.py`)

**Architecture**: 20 double blocks + 40 single blocks + token refiner

```python
class HunyuanVideoPipeline(BasePipeline):
    def enable_block_swap(self, blocks_to_swap):
        double_blocks = self.transformer.transformer_blocks  # 20
        single_blocks = self.transformer.single_transformer_blocks  # 40

        # Similar split formula to Flux
        double_to_swap = blocks_to_swap // 2
        single_to_swap = (blocks_to_swap - double_to_swap) * 2 + 1

        self.offloader_double = ModelOffloader(...)
        self.offloader_single = ModelOffloader(...)
```

---

## 3. OneTrainer Current Architecture

### 3.1 Existing Offloading Infrastructure

**LayerOffloadConductor** (`/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py`):
- Sophisticated layer-by-layer offloading system
- Pre-allocated memory buffers (StaticLayerAllocator)
- CUDA stream-based async transfers
- Activation offloading support
- Strategy-based loading/unloading

```python
class LayerOffloadConductor:
    """
    Orchestrates layer offloading during training.

    Key Features:
    - layer_offload_fraction: How much to offload (0.0-1.0)
    - Automatic strategy calculation
    - Integration with checkpointing
    - Activation offloading for gradient checkpointing
    """

    def before_layer(self, layer_index, call_index, activations):
        """Called before each layer forward."""
        # Wait for layer transfer
        # Schedule next layer loads
        # Handle activation transfers

    def after_layer(self, layer_index, call_index, activations):
        """Called after each layer forward."""
        # Schedule layer offload
        # Save activations for backward
```

**Checkpointing Utilities** (`/home/alex/OneTrainer/modules/util/checkpointing_util.py`):

```python
class OffloadCheckpointLayer(nn.Module):
    """Wraps layers for checkpointing + offloading."""

    def __init__(self, orig_module, train_device, conductor, layer_index):
        self.conductor = conductor
        self.layer_index = layer_index

    def forward(self, *args, **kwargs):
        call_id = _generate_call_index()

        if torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self.__checkpointing_forward,
                self.dummy, call_id, *args,
                use_reentrant=True
            )
        else:
            args = self.conductor.before_layer(self.layer_index, call_id, args)
            output = self.orig_forward(*args)
            self.conductor.after_layer(self.layer_index, call_id, args)
            return output
```

### 3.2 Model Setup Files

**BaseFluxSetup** - Has checkpointing enabled:
```python
def setup_optimizations(self, model, config):
    if config.gradient_checkpointing.enabled():
        model.transformer_offload_conductor = \
            enable_checkpointing_for_flux_transformer(model.transformer, config)
```

**BaseWanSetup** - Missing checkpointing! (TODO in code):
```python
def setup_optimizations(self, model, config):
    # TODO: Add gradient checkpointing for Wan transformer
    # if config.gradient_checkpointing.enabled():
    #     enable_checkpointing_for_wan_transformer(model.transformer, config)
```

---

## 4. Key Differences: diffusion-pipe vs OneTrainer

| Aspect | diffusion-pipe | OneTrainer |
|--------|---------------|------------|
| **Offloading Granularity** | Block-level (transformer blocks) | Layer-level (any layers) |
| **Memory Allocation** | Manual pinned buffers | StaticLayerAllocator with caching |
| **Async Transfers** | CUDA streams + ThreadPool | CUDA streams only |
| **Backward Hooks** | Manual registration | Integrated with checkpointing |
| **Strategy** | Fixed (swap N blocks) | Dynamic (fraction-based) |
| **Configuration** | `blocks_to_swap=32` | `layer_offload_fraction=0.8` |
| **Checkpointing** | Separate from offloading | Unified in OffloadCheckpointLayer |

### 4.1 OneTrainer Advantages
1. **Unified System**: Checkpointing + offloading in one wrapper
2. **Flexible Strategy**: Fraction-based rather than block count
3. **Better Memory Management**: Pre-allocated static buffers reduce fragmentation
4. **Activation Offloading**: Built-in support for gradient checkpoint activations

### 4.2 diffusion-pipe Advantages
1. **Simpler Block-Level Logic**: Easier to reason about
2. **Proven 24GB Training**: Tested configs for Wan 14B
3. **Pinned Memory Option**: Faster transfers (optional)

---

## 5. Integration Plan for OneTrainer

### 5.1 Option A: Use Existing LayerOffloadConductor (Recommended)

The existing infrastructure already supports everything needed! The issue is:
1. **Wan model**: Missing `enable_checkpointing_for_wan_transformer()` function
2. **Configuration**: Need to expose `layer_offload_fraction` in UI

**Changes Required**:

```python
# modules/util/checkpointing_util.py - Add Wan support

def enable_checkpointing_for_wan_transformer(
    model: nn.Module,
    config: TrainConfig,
) -> LayerOffloadConductor:
    """Enable checkpointing + offloading for Wan transformer."""
    return enable_checkpointing(model, config, config.compile, [
        # Wan has single block type: WanTransformerBlock or similar
        (model.blocks, ["hidden_states"]),  # Adjust param names
    ])
```

```python
# modules/modelSetup/BaseWanSetup.py - Enable checkpointing

def setup_optimizations(self, model: WanModel, config: TrainConfig):
    if config.gradient_checkpointing.enabled():
        model.transformer_offload_conductor = \
            enable_checkpointing_for_wan_transformer(model.transformer, config)
```

### 5.2 Option B: Port diffusion-pipe's ModelOffloader

If exact compatibility is needed, port the ModelOffloader class:

```python
# modules/util/BlockSwapManager.py (new file)

class BlockSwapManager:
    """
    Block swapping manager ported from diffusion-pipe.

    For compatibility with musubi-tuner configs.
    """

    def __init__(self, blocks, blocks_to_swap, device):
        assert blocks_to_swap <= len(blocks) - 2
        self.blocks = blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.stream = torch.cuda.Stream(device)

    def enable_block_swap(self):
        """Move initial blocks to CPU."""
        for i in range(self.blocks_to_swap):
            self.blocks[i].to('cpu')

    def wait_for_block(self, idx):
        """Ensure block is on GPU."""
        if self.blocks[idx].parameters().__next__().device.type == 'cpu':
            self.blocks[idx].to(self.device)
            torch.cuda.current_stream().wait_stream(self.stream)

    def submit_move_blocks_forward(self, idx):
        """Schedule next block transfer."""
        with torch.cuda.stream(self.stream):
            # Move current block to CPU
            if idx < self.blocks_to_swap:
                self.blocks[idx].to('cpu', non_blocking=True)
            # Prefetch next block
            next_idx = (idx + 1) % len(self.blocks)
            if next_idx < len(self.blocks):
                self.blocks[next_idx].to(self.device, non_blocking=True)
```

### 5.3 Configuration Changes

```python
# modules/util/config/TrainConfig.py

class TrainConfig:
    # Existing
    layer_offload_fraction: float = 0.0

    # New option for explicit block count (diffusion-pipe compatible)
    blocks_to_swap: int = 0  # 0 = use layer_offload_fraction
```

---

## 6. Recommended Implementation Steps

### Phase 1: Enable Wan Checkpointing (Immediate)
1. Add `enable_checkpointing_for_wan_transformer()` to checkpointing_util.py
2. Enable it in BaseWanSetup.py
3. Test with existing `layer_offload_fraction` config

### Phase 2: Test Layer Offloading (Immediate)
1. Set `layer_offload_fraction=0.8` for Wan
2. Verify VRAM usage drops
3. Compare training speed/quality

### Phase 3: Add blocks_to_swap Config (Optional)
1. Add config option for exact block count
2. Map to `layer_offload_fraction` internally
3. Compatibility with musubi-tuner users

### Phase 4: Optimize (Future)
1. Add pinned memory option
2. Tune prefetch distance
3. Profile and optimize transfer patterns

---

## 7. Expected Results

With proper block swapping on Wan 14B:

| Configuration | VRAM Usage | Training Speed |
|---------------|------------|----------------|
| No offloading | ~48GB (OOM) | N/A |
| blocks_to_swap=32 | ~22GB | ~85% baseline |
| blocks_to_swap=36 | ~18GB | ~75% baseline |
| Full offload | ~12GB | ~50% baseline |

**Note**: Speed depends heavily on PCIe bandwidth (x16 Gen4 recommended).

---

## 8. References

- diffusion-pipe: `/home/alex/diffusion-pipe/`
- musubi-tuner (embedded): `/home/alex/diffusion-pipe/musubi-tuner/`
- Example config: `/home/alex/diffusion-pipe/examples/wan_14b_min_vram.toml`
- OneTrainer offloading: `/home/alex/OneTrainer/modules/util/LayerOffloadConductor.py`
