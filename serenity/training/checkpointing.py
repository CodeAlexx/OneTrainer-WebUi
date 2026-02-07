"""
Gradient checkpointing utilities for EriTrainer.

Ported from OneTrainer's modules/util/checkpointing_util.py.
Provides layer-specific gradient checkpointing with optional activation offloading.
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
from torch import nn

from serenity.training.layer_offload import LayerOffloadConductor
from serenity.training.torch_util import add_dummy_grad_fn_, has_grad_fn

if TYPE_CHECKING:
    from serenity.core.config import TrainerConfig

logger = logging.getLogger(__name__)


def _kwargs_to_args(fun: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    """Convert kwargs to positional args based on function signature."""
    signature = dict(inspect.signature(fun).parameters)
    parameters = []

    for i, (key, value) in enumerate(signature.items()):
        if i < len(args):
            parameters.append(args[i])
        elif key in kwargs:
            parameters.append(kwargs[key])
        elif value.default is not value.empty:
            parameters.append(value.default)

    return tuple(parameters)


def _get_args_indices(fun: Callable, arg_names: list[str]) -> list[int]:
    """Get indices of named arguments in function signature."""
    signature = dict(inspect.signature(fun).parameters)
    indices = []

    for i, key in enumerate(signature.keys()):
        if key in arg_names:
            indices.append(i)

    return indices


__current_call_index = 0


def _generate_call_index() -> int:
    """Generate unique call index for tracking checkpoint calls."""
    global __current_call_index
    __current_call_index += 1
    return __current_call_index


class BaseCheckpointLayer(torch.nn.Module):
    """Base class for checkpoint layers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CheckpointLayer(BaseCheckpointLayer):
    """
    Standard gradient checkpointing layer.

    Wraps a module to use torch.utils.checkpoint.checkpoint for
    reduced memory usage during training.
    """

    def __init__(self, orig_module: nn.Module, orig_forward, train_device: torch.device, dtype: torch.dtype | None = None):
        super().__init__()

        assert (orig_module is None or orig_forward is None) and not (orig_module is None and orig_forward is None)
        self.checkpoint = orig_module
        self.orig_forward = orig_forward

        # Dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
        # CRITICAL: Must match model dtype (bfloat16) to avoid "Found dtype Float but expected BFloat16" error
        self.dummy = torch.zeros((1,), device=train_device, dtype=dtype or torch.bfloat16, requires_grad=True)

    def __checkpointing_forward(self, dummy: torch.Tensor, *args, **kwargs):
        return self.orig_forward(*args, **kwargs) if self.checkpoint is None else self.checkpoint(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self.__checkpointing_forward,
                self.dummy,
                *args,
                **kwargs,
                use_reentrant=False
            )
        else:
            return self.orig_forward(*args, **kwargs) if self.checkpoint is None else self.checkpoint(*args, **kwargs)


class OffloadCheckpointLayer(BaseCheckpointLayer):
    """
    Gradient checkpointing layer with activation offloading.

    Wraps a module to use torch.utils.checkpoint.checkpoint with
    activation offloading to CPU for further memory reduction.
    """

    def __init__(
            self,
            orig_module: nn.Module,
            orig_forward,
            train_device: torch.device,
            conductor: LayerOffloadConductor,
            layer_index: int,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()

        assert (orig_module is None or orig_forward is None) and not (orig_module is None and orig_forward is None)
        self.checkpoint = orig_module
        self.orig_forward = orig_forward

        # CRITICAL: Must match model dtype (bfloat16) to avoid "Found dtype Float but expected BFloat16" error
        self.dummy = torch.zeros((1,), device=train_device, dtype=dtype or torch.bfloat16, requires_grad=True)
        self.conductor = conductor
        self.layer_index = layer_index

    def __checkpointing_forward(self, dummy: torch.Tensor, call_id: int, *args):
        if self.layer_index == 0:
            if not torch.is_grad_enabled():
                self.conductor.start_forward(True)

        args = self.conductor.before_layer(self.layer_index, call_id, args)
        output = self.orig_forward(*args) if self.checkpoint is None else self.checkpoint(*args)
        output = self.conductor.after_layer(self.layer_index, call_id, output)

        # Make sure at least one output tensor has a grad_fn so the checkpoint output has a grad_fn.
        # This can only happen if a checkpointed block has no trainable parameters.
        if torch.is_grad_enabled() and not has_grad_fn(output):
            output = add_dummy_grad_fn_(output)

        return output

    def forward(self, *args, **kwargs):
        call_id = _generate_call_index()
        args = _kwargs_to_args(self.orig_forward if self.checkpoint is None else self.checkpoint.forward, args, kwargs)
        if torch.is_grad_enabled():
            result = torch.utils.checkpoint.checkpoint(
                self.__checkpointing_forward,
                self.dummy,
                call_id,
                *args,
                use_reentrant=False
            )
            return result
        else:
            if self.layer_index == 0:
                self.conductor.start_forward(False)

            args = self.conductor.before_layer(self.layer_index, call_id, args)
            output = self.orig_forward(*args) if self.checkpoint is None else self.checkpoint(*args)
            return self.conductor.after_layer(self.layer_index, call_id, output)


def create_checkpoint(
        orig_module: nn.Module,
        train_device: torch.device,
        include_from_offload_param_names: list[str] | None = None,
        conductor: LayerOffloadConductor | None = None,
        layer_index: int = 0,
        compile: bool = False,
        dtype: torch.dtype | None = None,
) -> Callable:
    """
    Create a checkpoint wrapper for a module.

    Args:
        orig_module: Module to wrap with checkpointing
        train_device: Device for training (GPU)
        include_from_offload_param_names: Parameter names to include in offloading
        conductor: LayerOffloadConductor for activation offloading
        layer_index: Index of this layer in the sequence
        compile: Whether to compile the layer
        dtype: Data type for checkpointing tensors (defaults to bfloat16)

    Returns:
        Wrapped module with checkpointing enabled
    """
    if include_from_offload_param_names is None:
        include_from_offload_param_names = []
    included_offload_param_indices = _get_args_indices(orig_module.forward, include_from_offload_param_names)

    if conductor is not None:
        conductor.add_layer(orig_module, included_offload_param_indices)

    # Get dtype from module if not provided
    # Skip int8/fp8 quantized weights - use float dtype for checkpointing
    if dtype is None:
        for p in orig_module.parameters():
            if p.dtype in (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2):
                continue  # Skip quantized weights
            dtype = p.dtype
            break
        # Fallback to bfloat16 if only quantized weights found
        if dtype is None:
            dtype = torch.bfloat16

    if conductor is not None and conductor.offload_activated():
        if compile:
            layer = OffloadCheckpointLayer(
                orig_module=orig_module,
                orig_forward=None,
                train_device=train_device,
                conductor=conductor,
                layer_index=layer_index,
                dtype=dtype,
            )
            orig_module.compile(fullgraph=True)
            return layer
        else:
            # Only patch forward() if possible - inserting layers can cause issues
            layer = OffloadCheckpointLayer(
                orig_module=None,
                orig_forward=orig_module.forward,
                train_device=train_device,
                conductor=conductor,
                layer_index=layer_index,
                dtype=dtype,
            )
            orig_module.forward = layer.forward
            return orig_module
    else:
        if compile:
            layer = CheckpointLayer(orig_module=orig_module, orig_forward=None, train_device=train_device, dtype=dtype)
            layer.compile(fullgraph=True)
            return layer
        else:
            layer = CheckpointLayer(orig_module=None, orig_forward=orig_module.forward, train_device=train_device, dtype=dtype)
            orig_module.forward = layer.forward
            return orig_module


def _create_checkpoints_for_module_list(
        module_list: nn.ModuleList,
        include_from_offload_param_names: list[str],
        conductor: LayerOffloadConductor | None,
        train_device: torch.device,
        layer_index: int,
        compile: bool,
) -> int:
    """Create checkpoints for all modules in a ModuleList."""
    for i, layer in enumerate(module_list):
        if isinstance(module_list[i], BaseCheckpointLayer):
            continue
        module_list[i] = create_checkpoint(
            layer, train_device,
            include_from_offload_param_names,
            conductor, layer_index, compile=compile,
        )
        layer_index += 1
    return layer_index


def _remove_checkpoint_keys(module, state_dict, prefix, local_metadata):
    """Hook to clean up checkpoint keys from state dict."""
    for k in list(state_dict.keys()):
        if ".checkpoint." in k:
            state_dict[k.replace(".checkpoint.", ".")] = state_dict.pop(k)


def enable_checkpointing(
        model: nn.Module,
        config: "TrainerConfig",
        compile: bool,
        lists,  # List of (module_list_or_type, param_names) tuples - must be in execution order
        offload_enabled: bool = True,
) -> LayerOffloadConductor:
    """
    Enable gradient checkpointing with optional layer/activation offloading.

    Follows OneTrainer's enable_checkpointing signature exactly.

    Args:
        model: Model to enable checkpointing on
        config: TrainerConfig with memory settings
        compile: Whether to compile checkpointed layers
        lists: List of (module_list_or_type, param_names) tuples specifying
               which layers to checkpoint and which params to include in offload.
               Must be in the exact order they are executed - otherwise offloading fails.
        offload_enabled: Whether to enable layer/activation offloading

    Returns:
        LayerOffloadConductor for managing offloading
    """
    train_device = torch.device("cuda")
    temp_device = torch.device("cpu")

    # Create conductor matching OneTrainer's approach
    gc_method = config.memory.get_gradient_checkpointing_method()
    conductor = LayerOffloadConductor(
        module=model,
        train_device=train_device,
        temp_device=temp_device,
        layer_offload_fraction=config.memory.layer_offload_fraction,
        offload_activations=gc_method.offload() and config.memory.enable_activation_offloading,
        enable_async=config.memory.enable_async_offloading,
    )

    layer_index = 0
    for type_or_list, param_names in lists:
        assert isinstance(type_or_list, (nn.ModuleList, type))
        if isinstance(type_or_list, nn.ModuleList):
            module_list = type_or_list
            layer_index = _create_checkpoints_for_module_list(
                module_list,
                param_names,
                conductor if offload_enabled else None,
                train_device,
                layer_index,
                compile=compile,
            )
        else:
            t = type_or_list
            for child_module in model.modules():
                if isinstance(child_module, nn.ModuleList) and len(child_module) > 0 and isinstance(child_module[0], t):
                    module_list = child_module
                    assert all(isinstance(m, t) for m in child_module)
                    layer_index = _create_checkpoints_for_module_list(
                        module_list,
                        param_names,
                        conductor if offload_enabled else None,
                        train_device,
                        layer_index,
                        compile=compile,
                    )

    model._register_state_dict_hook(_remove_checkpoint_keys)

    logger.info(f"Gradient checkpointing: enabled for {layer_index} layers")
    if gc_method.offload() and config.memory.enable_activation_offloading:
        logger.info("Activation offloading: enabled (CPU)")
    if config.memory.layer_offload_fraction > 0:
        logger.info(f"Layer offloading: {config.memory.layer_offload_fraction * 100:.1f}%")

    return conductor


def enable_checkpointing_for_z_image_transformer(
        model: nn.Module,
        config: "TrainerConfig",
) -> LayerOffloadConductor:
    """
    Enable gradient checkpointing for Z-Image transformer.

    Follows OneTrainer's signature exactly.

    Args:
        model: ZImageTransformer2DModel
        config: TrainerConfig with memory settings

    Returns:
        LayerOffloadConductor for managing the offloading
    """
    return enable_checkpointing(model, config, False, [
        (model.noise_refiner, ["x"]),
        (model.context_refiner, ["x"]),
        (model.layers, ["x"]),
    ])


def enable_checkpointing_for_flux_transformer(
        model: nn.Module,
        config: "TrainerConfig",
) -> LayerOffloadConductor:
    """
    Enable gradient checkpointing for Flux transformer.

    Follows OneTrainer's signature exactly.

    Args:
        model: FluxTransformer2DModel
        config: TrainerConfig with memory settings

    Returns:
        LayerOffloadConductor for managing the offloading
    """
    return enable_checkpointing(model, config, False, [
        (model.transformer_blocks, ["hidden_states", "encoder_hidden_states"]),
        (model.single_transformer_blocks, ["hidden_states"]),
    ])


def enable_checkpointing_for_flux2_transformer(
        model: nn.Module,
        config: "TrainerConfig",
) -> LayerOffloadConductor:
    """
    Enable gradient checkpointing for Flux 2 transformer.

    Follows OneTrainer's signature exactly. Flux 2 has the same layer
    structure as Flux 1 (transformer_blocks + single_transformer_blocks).

    Args:
        model: Flux2Transformer2DModel
        config: TrainerConfig with memory settings

    Returns:
        LayerOffloadConductor for managing the offloading
    """
    return enable_checkpointing(model, config, False, [
        (model.transformer_blocks, ["hidden_states", "encoder_hidden_states"]),
        (model.single_transformer_blocks, ["hidden_states"]),
    ])


def enable_checkpointing_for_sdxl_unet(
        model: nn.Module,
        config: "TrainerConfig",
) -> LayerOffloadConductor:
    """
    Enable gradient checkpointing for SDXL UNet.

    Uses the diffusers BasicTransformerBlock type for detection.

    Args:
        model: UNet2DConditionModel
        config: TrainerConfig with memory settings

    Returns:
        LayerOffloadConductor for managing the offloading
    """
    from diffusers.models.attention import BasicTransformerBlock

    return enable_checkpointing(model, config, False, [
        (BasicTransformerBlock, []),
    ])


def enable_checkpointing_for_qwen_image_edit_transformer(
        model: nn.Module,
        config: "TrainerConfig",
) -> LayerOffloadConductor:
    """
    Enable gradient checkpointing for Qwen Image Edit transformer.

    The Qwen Image transformer has transformer_blocks that process both
    image tokens and text embeddings together.

    Args:
        model: QwenImageTransformer2DModel
        config: TrainerConfig with memory settings

    Returns:
        LayerOffloadConductor for managing the offloading
    """
    return enable_checkpointing(model, config, False, [
        (model.transformer_blocks, ["hidden_states", "encoder_hidden_states"]),
    ])


def enable_checkpointing_for_qwen_image_edit_encoder_layers(
        model: nn.Module,
        config: "TrainerConfig",
) -> LayerOffloadConductor:
    """
    Enable gradient checkpointing for Qwen2.5-VL text encoder layers.

    Uses Qwen2_5_VLDecoderLayer for layer detection.
    Note: Activation offloading is disabled for encoder layers since
    clip skip is not implemented for QwenVL.

    Args:
        model: Qwen2_5_VLForConditionalGeneration text encoder
        config: TrainerConfig with memory settings

    Returns:
        LayerOffloadConductor for managing the offloading
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

    return enable_checkpointing(model, config, False, [
        (Qwen2_5_VLDecoderLayer, []),
    ])
