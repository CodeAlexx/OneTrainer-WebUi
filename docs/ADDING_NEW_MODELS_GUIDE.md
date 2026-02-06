# Adding New Models to OneTrainer - Complete Guide

A comprehensive blueprint for adding new diffusion models to OneTrainer with training, inference, block swapping, and UI integration.

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture Understanding](#model-architecture-understanding)
3. [Required Files](#required-files)
4. [Training Implementation](#training-implementation)
5. [Inference/Sampling Implementation](#inferencesampling-implementation)
6. [Block Swapping for Memory Efficiency](#block-swapping-for-memory-efficiency)
7. [UI Integration](#ui-integration)
8. [Testing](#testing)
9. [Examples](#examples)

---

## Overview

### What You Need to Add a New Model

1. **Model Class** (`modules/model/YourModel.py`) - Data container for model components
2. **Model Loader** (`modules/modelLoader/YourModelLoader.py`) - Load/save model from disk
3. **Model Setup** (`modules/modelSetup/BaseYourSetup.py`, `YourLoRASetup.py`, etc.) - Configure training
4. **Model Sampler** (`modules/modelSampler/YourSampler.py`) - Generate samples during training
5. **Enum Updates** - Add to `ModelType`, register in factory methods
6. **UI Updates** - Add training tab configuration

### Directory Structure
```
modules/
├── model/
│   └── YourModel.py              # Model data container
├── modelLoader/
│   ├── YourModelLoader.py        # Full model loader
│   └── YourModelLoaderLoRA.py    # LoRA variant loader
├── modelSetup/
│   ├── BaseYourSetup.py          # Shared training logic
│   ├── YourFineTuneSetup.py      # Full fine-tune setup
│   └── YourLoRASetup.py          # LoRA training setup
└── modelSampler/
    └── YourSampler.py            # Inference/sampling
```

---

## Model Architecture Understanding

Before implementing, understand your model's architecture:

### Key Components to Identify
1. **Transformer/UNet** - Main trainable component
   - Block structure (nn.ModuleList of layers)
   - Block names (e.g., `model.layers`, `model.transformer_blocks`, `model.visual_transformer_blocks`)

2. **Text Encoder(s)** - Text conditioning
   - Single or multiple (e.g., CLIP + T5 for SDXL)
   - Tokenizer requirements

3. **VAE** - Image encoding/decoding
   - Latent dimensions
   - Scaling factors

4. **Noise Scheduler** - Diffusion process
   - Flow matching vs DDPM
   - Timestep handling

### Example: Kandinsky 5 Architecture
```python
# From kandinsky/models/dit.py
class DiffusionTransformer3D(nn.Module):
    self.text_transformer_blocks  # nn.ModuleList of TransformerEncoderBlock
    self.visual_transformer_blocks  # nn.ModuleList of TransformerDecoderBlock (32 blocks)
    self.num_visual_blocks = 32
```

### Example: ZImage Architecture
```python
# From diffusers ZImageTransformer2DModel
transformer.layers  # nn.ModuleList of transformer blocks (28 typical)
transformer.noise_refiner
transformer.context_refiner
```

---

## Required Files

### 1. Model Class (`modules/model/YourModel.py`)

```python
"""
YourModel - Model class for Your Model
"""

from contextlib import nullcontext
from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor
from modules.util.musubi_block_swap import MusubiBlockSwapManager

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKL,
    YourTransformerModel,
    YourScheduler,
    YourPipeline,
)
from transformers import YourTokenizer, YourTextEncoder


class YourModel(BaseModel):
    # Base model components
    tokenizer: YourTokenizer | None
    noise_scheduler: YourScheduler | None
    text_encoder: YourTextEncoder | None
    vae: AutoencoderKL | None
    transformer: YourTransformerModel | None

    # Autocast context
    autocast_context: torch.autocast | nullcontext
    text_encoder_autocast_context: torch.autocast | nullcontext

    # Training dtype
    train_dtype: DataType
    text_encoder_train_dtype: DataType

    # Offloading conductors
    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # Musubi block swap manager for training
    musubi_manager: MusubiBlockSwapManager | None

    # LoRA training data
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)

        # Initialize all to None
        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None

        self.autocast_context = nullcontext()
        self.text_encoder_autocast_context = nullcontext()

        self.train_dtype = DataType.FLOAT_32
        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.musubi_manager = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        """Return list of LoRA adapters for saving."""
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
        ] if a is not None]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)
        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)
        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self):
        """Create pipeline for inference."""
        return YourPipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )

    def encode_text(self, train_device, batch_size=1, text=None, tokens=None, ...):
        """Encode text to embeddings."""
        # Implementation depends on your model
        pass

    def scale_latents(self, latents: Tensor) -> Tensor:
        """Scale latents using VAE config."""
        return latents * self.vae.config.scaling_factor

    def unscale_latents(self, latents: Tensor) -> Tensor:
        """Unscale latents for VAE decoding."""
        return latents / self.vae.config.scaling_factor
```

### 2. Model Loader (`modules/modelLoader/YourModelLoader.py`)

```python
"""
YourModelLoader - Load Your Model from disk
"""

from modules.model.YourModel import YourModel
from modules.util.enum.ModelType import ModelType
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch
from diffusers import AutoencoderKL, YourTransformerModel, YourScheduler
from transformers import YourTokenizer, YourTextEncoder


class YourModelLoader:
    def __init__(self):
        pass

    def load(
            self,
            model_type: ModelType,
            model_path: str,
            weight_dtypes: ModelWeightDtypes,
    ) -> YourModel:
        model = YourModel(model_type=model_type)

        # Load tokenizer
        model.tokenizer = YourTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
        )

        # Load text encoder
        model.text_encoder = YourTextEncoder.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtypes.text_encoder.torch_dtype(),
        )

        # Load VAE
        model.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=weight_dtypes.vae.torch_dtype(),
        )

        # Load transformer
        model.transformer = YourTransformerModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=weight_dtypes.transformer.torch_dtype(),
        )

        # Load scheduler
        model.noise_scheduler = YourScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
        )

        return model
```

---

## Training Implementation

### 3. Base Setup (`modules/modelSetup/BaseYourSetup.py`)

```python
"""
BaseYourSetup - Shared training logic for Your Model
"""

from abc import ABCMeta
from random import Random

from modules.model.YourModel import YourModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import enable_checkpointing
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context
from modules.util.musubi_block_swap import MusubiBlockSwapManager
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseYourSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    metaclass=ABCMeta
):

    def setup_optimizations(
            self,
            model: YourModel,
            config: TrainConfig,
    ):
        # 1. Gradient checkpointing
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = enable_checkpointing(
                model.transformer, config, config.compile, [
                    # (block_list, [tensor_arg_names])
                    (model.transformer.layers, ["hidden_states"]),
                ]
            )

        # 2. Autocast context
        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, [
                config.weight_dtypes().transformer,
                config.weight_dtypes().text_encoder,
                config.weight_dtypes().vae,
            ], config.enable_autocast_cache)

        # 3. Quantization
        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

        # 4. Musubi block swap for training (see Block Swapping section below)
        blocks_to_swap = config.musubi_blocks_to_swap
        if blocks_to_swap == 0 and config.gradient_checkpointing.enabled():
            # Auto-enable: swap ~half the blocks
            num_blocks = len(model.transformer.layers)
            blocks_to_swap = num_blocks // 2

        if blocks_to_swap > 0 and hasattr(model.transformer, 'layers'):
            model.musubi_manager = MusubiBlockSwapManager.build(
                depth=len(model.transformer.layers),
                blocks_to_swap=blocks_to_swap,
                swap_device="cpu",
            )
            if model.musubi_manager is not None:
                # For diffusers models: use forward hooks
                model.musubi_manager.activate_with_forward_hooks(
                    model.transformer.layers,
                    self.train_device,
                    grad_enabled=True
                )
                print(f"Musubi block swap enabled: {blocks_to_swap} blocks swapped to CPU")

    def predict(
            self,
            model: YourModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(train_progress.global_step if not deterministic else 0)

            # 1. Get text embeddings
            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
            )

            # 2. Get latent image
            latent_image = batch['latent_image']
            scaled_latent_image = model.scale_latents(latent_image)

            # 3. Create noise
            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            # 4. Sample timestep
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic, generator,
                scaled_latent_image.shape[0], config,
            )

            # 5. Add noise to latents
            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image, latent_noise, timestep,
                model.noise_scheduler.timesteps,
            )

            # 6. Forward pass through transformer
            predicted_flow = model.transformer(
                hidden_states=scaled_noisy_latent_image,
                timestep=timestep,
                encoder_hidden_states=text_encoder_output,
                return_dict=True,
            ).sample

            # 7. Calculate target (for flow matching)
            flow = latent_noise - scaled_latent_image

            return {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow,
            }

    def calculate_loss(
            self,
            model: YourModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()
```

### 4. LoRA Setup (`modules/modelSetup/YourLoRASetup.py`)

```python
"""
YourLoRASetup - LoRA training setup
"""

from modules.model.YourModel import YourModel
from modules.modelSetup.BaseYourSetup import BaseYourSetup
from modules.module.LoRAModule import create_peft_wrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class YourLoRASetup(BaseYourSetup):
    def __init__(self, train_device, temp_device, debug_mode):
        super().__init__(train_device=train_device, temp_device=temp_device, debug_mode=debug_mode)

    def create_parameters(self, model: YourModel, config: TrainConfig) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        self._create_model_part_parameters(
            parameter_group_collection, "transformer_lora",
            model.transformer_lora, config.transformer
        )
        return parameter_group_collection

    def __setup_requires_grad(self, model: YourModel, config: TrainConfig):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
        self._setup_model_part_requires_grad(
            "transformer_lora", model.transformer_lora,
            config.transformer, model.train_progress
        )

    def setup_model(self, model: YourModel, config: TrainConfig):
        # Create LoRA wrapper
        model.transformer_lora = create_peft_wrapper(
            model.transformer, "transformer", config, config.layer_filter.split(",")
        )

        # Load existing LoRA state if available
        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        self.__setup_requires_grad(model, config)
        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(self, model: YourModel, config: TrainConfig):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        model.text_encoder.eval()
        model.vae.eval()
        model.transformer.train() if config.transformer.train else model.transformer.eval()

    def after_optimizer_step(self, model, config, train_progress):
        self.__setup_requires_grad(model, config)
```

---

## Inference/Sampling Implementation

### 5. Model Sampler (`modules/modelSampler/YourSampler.py`)

```python
"""
YourSampler - Generate samples during training
"""

from modules.model.YourModel import YourModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType

import torch
from PIL import Image


class YourSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: YourModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)
        self.model = model
        self.model_type = model_type
        self.pipeline = None

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat,
            text_encoder_layer_skip: int,
            force_last_timestep: bool = False,
            on_sample: callable = None,
            on_update_progress: callable = None,
    ) -> list[Image.Image]:
        # Move models to train device for sampling
        self.model.vae_to(self.train_device)
        self.model.text_encoder_to(self.train_device)
        self.model.transformer_to(self.train_device)

        # Create pipeline if needed
        if self.pipeline is None:
            self.pipeline = self.model.create_pipeline()
            self.pipeline.to(self.train_device)

        # Generate with inference block swap if configured
        # (See Block Swapping section for inference block swap)
        with torch.no_grad():
            generator = torch.Generator(device=self.train_device)
            generator.manual_seed(sample_config.seed)

            output = self.pipeline(
                prompt=sample_config.prompt,
                negative_prompt=sample_config.negative_prompt,
                num_inference_steps=sample_config.diffusion_steps,
                guidance_scale=sample_config.cfg_scale,
                height=sample_config.height,
                width=sample_config.width,
                generator=generator,
            )

        images = output.images

        # Save images
        for i, image in enumerate(images):
            if on_sample:
                on_sample(image)
            self._save_image(image, destination, image_format, i)

        return images
```

---

## Block Swapping for Memory Efficiency

### Overview

Block swapping streams transformer blocks between CPU and GPU to reduce VRAM usage:

1. **Training Block Swap (Musubi)**: Uses forward/backward hooks to move blocks during training
2. **Inference Block Swap**: Moves blocks on-demand during generation

### MusubiBlockSwapManager

Located in `modules/util/musubi_block_swap.py`:

```python
from modules.util.musubi_block_swap import MusubiBlockSwapManager

# Build manager: swap last N blocks
manager = MusubiBlockSwapManager.build(
    depth=32,           # Total number of blocks
    blocks_to_swap=16,  # Swap last 16 blocks
    swap_device="cpu",
)

# For diffusers models (can't modify forward loop): use forward hooks
manager.activate_with_forward_hooks(
    blocks=model.transformer.layers,
    compute_device=torch.device('cuda'),
    grad_enabled=True  # Training mode
)

# For custom models (can modify forward loop): use manual swap
manager.activate(
    blocks=model.transformer.visual_transformer_blocks,
    compute_device=torch.device('cuda'),
    grad_enabled=True
)

# In custom forward loop:
for i, block in enumerate(self.blocks):
    if manager.is_managed_block(i):
        manager.stream_in(block, device)

    output = block(input)

    if manager.is_managed_block(i):
        manager.stream_out(block)
```

### Two Approaches

#### 1. Forward Hooks (for diffusers models)

Use when you can't modify the transformer's forward method:

```python
# In setup_optimizations:
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

This registers:
- `register_forward_pre_hook`: Moves block to GPU before forward
- `register_forward_hook`: Moves block to CPU after forward
- `register_full_backward_pre_hook`: Moves block to GPU before backward
- `register_full_backward_hook`: Moves block to CPU after backward

#### 2. Manual Swap (for custom models like K5)

Use when you control the forward loop:

```python
# In transformer forward method:
elif self.musubi_manager is not None and self.training:
    device = visual_embed.device
    for i, block in enumerate(self.visual_transformer_blocks):
        # Stream in before forward
        if self.musubi_manager.is_managed_block(i):
            self.musubi_manager.stream_in(block, device)

        # Forward through block
        visual_embed = block(visual_embed, ...)

        # Stream out after forward (backward hooks handle the rest)
        if self.musubi_manager.is_managed_block(i):
            self.musubi_manager.stream_out(block)
```

### Inference Block Swap

For memory-efficient inference (sampling), implement in the transformer:

```python
class YourTransformer(nn.Module):
    def __init__(self, ..., block_swap_enabled=False, blocks_in_memory=6):
        self.block_swap_enabled = block_swap_enabled
        self.blocks_in_memory = blocks_in_memory
        self.swap_stream = torch.cuda.Stream() if block_swap_enabled else None
        self.currently_loaded_blocks = []
        self.prefetch_started = set()

    def forward(self, x, ...):
        if self.block_swap_enabled and not self.training:
            for i, block in enumerate(self.blocks):
                # Offload old blocks
                while len(self.currently_loaded_blocks) >= self.blocks_in_memory:
                    block_to_offload = self.currently_loaded_blocks.pop(0)
                    self.blocks[block_to_offload].to('cpu')

                # Load current block
                if i not in self.currently_loaded_blocks:
                    block.to(x.device)
                    self.currently_loaded_blocks.append(i)

                x = block(x)

                # Prefetch next block asynchronously
                next_idx = i + 1
                if next_idx < len(self.blocks) and self.swap_stream:
                    with torch.cuda.stream(self.swap_stream):
                        self.blocks[next_idx].to(x.device, non_blocking=True)
                    self.prefetch_started.add(next_idx)
        else:
            # Normal forward
            for block in self.blocks:
                x = block(x)
        return x
```

---

## UI Integration

### 1. Add to TrainConfig (`modules/util/config/TrainConfig.py`)

```python
# In class TrainConfig:
musubi_blocks_to_swap: int  # 0 = auto, or specify exact count

# In default_values():
data.append(("musubi_blocks_to_swap", 0, int, False))
```

### 2. Add to OffloadingWindow (`modules/ui/OffloadingWindow.py`)

```python
# In __content_frame:
components.label(frame, row, 0, "Block Swap (Training)",
                 tooltip="Musubi block swap: streams blocks CPU↔GPU. 0 = auto, or specify count.")
components.entry(frame, row, 1, self.ui_state, "musubi_blocks_to_swap")
```

### 3. Add Training Tab (`modules/ui/TrainingTab.py`)

```python
# In refresh_ui:
elif self.train_config.model_type.is_your_model():
    self.__setup_your_model_ui(column_0, column_1, column_2)

def __setup_your_model_ui(self, column_0, column_1, column_2):
    self.__create_base_frame(column_0, 0)
    self.__create_text_encoder_frame(column_0, 1)

    self.__create_base2_frame(column_1, 0)
    self.__create_transformer_frame(column_1, 1)
    self.__create_noise_frame(column_1, 2)

    self.__create_masked_frame(column_2, 1)
    self.__create_loss_frame(column_2, 2)
    self.__create_layer_frame(column_2, 3)
```

---

## Testing

### Unit Tests

Create `tests/test_your_model.py`:

```python
import torch
import torch.nn as nn
from modules.util.musubi_block_swap import MusubiBlockSwapManager, _module_on_device


def test_block_swap_manager():
    """Test MusubiBlockSwapManager."""
    manager = MusubiBlockSwapManager.build(depth=10, blocks_to_swap=3)
    assert manager is not None
    assert manager.block_indices == {7, 8, 9}
    print("✓ Block swap manager works")


def test_forward_hooks():
    """Test forward hooks with real blocks."""
    if not torch.cuda.is_available():
        print("⚠ Skipping CUDA test")
        return

    manager = MusubiBlockSwapManager.build(depth=5, blocks_to_swap=2)
    blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
    blocks.to('cuda')

    manager.activate_with_forward_hooks(blocks, torch.device('cuda'), grad_enabled=True)

    # Managed blocks should be offloaded
    assert _module_on_device(blocks[3], 'cpu')
    assert _module_on_device(blocks[4], 'cpu')
    assert _module_on_device(blocks[0], 'cuda')

    # Forward pass
    x = torch.randn(2, 10, device='cuda')
    for block in blocks:
        x = block(x)

    # After forward, managed blocks back on CPU
    assert _module_on_device(blocks[3], 'cpu')
    assert _module_on_device(blocks[4], 'cpu')

    manager.cleanup()
    print("✓ Forward hooks work")


if __name__ == "__main__":
    test_block_swap_manager()
    test_forward_hooks()
    print("\n=== All tests passed! ===")
```

### Integration Test

```python
# Quick training test
from modules.modelSetup.YourLoRASetup import YourLoRASetup
from modules.modelLoader.YourModelLoader import YourModelLoader
from modules.util.config.TrainConfig import TrainConfig

# Load model
loader = YourModelLoader()
model = loader.load(ModelType.YOUR_MODEL, "path/to/model", weight_dtypes)

# Setup
config = TrainConfig.default_values()
config.gradient_checkpointing = GradientCheckpointingMethod.ON
config.musubi_blocks_to_swap = 0  # Auto

setup = YourLoRASetup(train_device, temp_device, debug_mode=False)
setup.setup_optimizations(model, config)
setup.setup_model(model, config)
setup.setup_train_device(model, config)

# Check Musubi is enabled
assert model.musubi_manager is not None
print("✓ Integration test passed")
```

---

## Examples

### Complete Example: Kandinsky 5

See these files for a complete implementation:
- `modules/model/Kandinsky5Model.py`
- `modules/modelLoader/Kandinsky5ModelLoader.py`
- `modules/modelSetup/Kandinsky5LoRASetup.py`
- `modules/modelSampler/Kandinsky5Sampler.py`
- `models/kandinsky-5-code/kandinsky/models/dit.py` (manual block swap in forward)

### Complete Example: ZImage

See these files for diffusers model with forward hooks:
- `modules/model/ZImageModel.py`
- `modules/modelSetup/BaseZImageSetup.py` (forward hooks block swap)
- `modules/modelSetup/ZImageLoRASetup.py`

---

## Checklist for New Models

- [ ] Create `YourModel.py` with all components
- [ ] Add `musubi_manager` attribute to model
- [ ] Create `YourModelLoader.py`
- [ ] Create `BaseYourSetup.py` with:
  - [ ] `setup_optimizations()` with checkpointing and Musubi
  - [ ] `predict()` for training forward pass
  - [ ] `calculate_loss()` for loss computation
- [ ] Create `YourLoRASetup.py` and/or `YourFineTuneSetup.py`
- [ ] Create `YourSampler.py` for inference
- [ ] Add `ModelType.YOUR_MODEL` enum
- [ ] Register in factory methods (`create_model_loader`, `create_model_setup`, etc.)
- [ ] Add `musubi_blocks_to_swap` to TrainConfig (already done globally)
- [ ] Add UI tab in TrainingTab.py
- [ ] Test with unit tests
- [ ] Test with integration training run

---

## RamTorch (Future Enhancement)

RamTorch is an alternative to block swap that stores parameters in CPU RAM with on-demand GPU transfer:

```python
from ramtorch.helpers import replace_linear_with_ramtorch

# Replace Linear layers with RamTorch Linear
replace_linear_with_ramtorch(model, device="cuda")

# Key notes:
# - NOT compatible with group_offload (choose one or other)
# - Embeddings/LayerNorm need to stay on GPU
# - Need to patch PEFT for LoRA compatibility
# - Must apply AFTER LoRA injection, or patch LoRA to support RamTorch Linear
```

See `TODO.md` for RamTorch implementation notes.
