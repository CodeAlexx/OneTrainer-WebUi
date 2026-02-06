"""
Torch utilities for EriTrainer.

Ported from OneTrainer's modules/util/torch_util.py for checkpointing
and layer offloading support.
"""

import gc
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

import packaging.version

torch_version = packaging.version.parse(torch.__version__)


def get_tensor_data(
        data: torch.Tensor | list | tuple | dict,
        include_parameter_indices: list[int] | None = None,
) -> list[torch.Tensor]:
    """Extract tensor data from nested structures."""
    tensors = []

    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        return [data.data]
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                tensors.extend(get_tensor_data(elem))
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            tensors.extend(get_tensor_data(elem))

    return tensors


def has_grad_fn(
        data: torch.Tensor | list | tuple | dict,
        include_parameter_indices: list[int] | None = None,
) -> bool:
    """Check if any tensor in the data has a grad_fn."""
    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        return data.grad_fn is not None
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                if has_grad_fn(elem):
                    return True
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            if has_grad_fn(elem):
                return True

    return False


def add_dummy_grad_fn_(
        data: torch.Tensor | list | tuple | dict,
) -> Any:
    """Add a dummy grad_fn to tensors that don't have one.

    This is needed for checkpointing to work when training LoRA,
    as the output needs a grad_fn even if the checkpointed block
    has no trainable parameters.
    """
    if isinstance(data, torch.Tensor):
        if data.grad_fn is not None:
            return data
        grad_tensor = torch.zeros(
            size=(0, *data.shape[1:]),
            requires_grad=True,
            device=data.device,
            dtype=data.dtype
        )
        return torch.cat([data, grad_tensor], dim=0)
    if isinstance(data, list):
        for i, elem in enumerate(data):
            if isinstance(elem, torch.Tensor):
                if elem.grad_fn is not None:
                    return data
                grad_tensor = torch.zeros(
                    size=(0, *elem.shape[1:]),
                    requires_grad=True,
                    device=elem.device,
                    dtype=elem.dtype
                )
                data[i] = torch.cat([elem, grad_tensor], dim=0)
                return data
            else:
                data[i] = add_dummy_grad_fn_(elem)
    if isinstance(data, tuple):
        for i, elem in enumerate(data):
            if isinstance(elem, torch.Tensor):
                if elem.grad_fn is not None:
                    return data
                grad_tensor = torch.zeros(
                    size=(0, *elem.shape[1:]),
                    requires_grad=True,
                    device=elem.device,
                    dtype=elem.dtype
                )
                data = list(data)
                data[i] = torch.cat([elem, grad_tensor], dim=0)
                data = tuple(data)
                return data
            else:
                data = list(data)
                data[i] = add_dummy_grad_fn_(elem)
                data = tuple(data)
    elif isinstance(data, dict):
        for key, elem in data.items():
            if isinstance(elem, torch.Tensor):
                if elem.grad_fn is not None:
                    return data
                grad_tensor = torch.zeros(
                    size=(0, *elem.shape[1:]),
                    requires_grad=True,
                    device=elem.device,
                    dtype=elem.dtype
                )
                data[key] = torch.cat([elem, grad_tensor], dim=0)
                return data
            else:
                data[key] = add_dummy_grad_fn_(elem)

    return data


def tensors_to_device_(
        data: torch.Tensor | list | tuple | dict,
        device: torch.device,
        include_parameter_indices: list[int] | None = None,
        non_blocking: bool = False,
        allocator: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> bool:
    """Move tensors to device in-place."""
    tensor_transferred = False

    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        if allocator is None:
            data.data = data.data.to(device=device, non_blocking=non_blocking)
        else:
            tensor = allocator(data)
            tensor.copy_(data, non_blocking=non_blocking)
            data.data = tensor
        tensor_transferred = True
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                tensor_transferred |= tensors_to_device_(elem, device, non_blocking=non_blocking, allocator=allocator)
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            tensor_transferred |= tensors_to_device_(elem, device, non_blocking=non_blocking, allocator=allocator)

    return tensor_transferred


def replace_tensors_(
        target_data: torch.Tensor | list | tuple | dict,
        source_data: torch.Tensor | list | tuple | dict,
        include_parameter_indices: list[int] | None = None,
):
    """Replace tensor data in target with data from source."""
    if isinstance(target_data, torch.Tensor) and include_parameter_indices is None:
        target_data.data = source_data.data
    elif isinstance(target_data, (list, tuple)):
        for i, elem in enumerate(target_data):
            if include_parameter_indices is None or i in include_parameter_indices:
                replace_tensors_(elem, source_data[i])
    elif isinstance(target_data, dict) and include_parameter_indices is None:
        for key, elem in target_data.items():
            replace_tensors_(elem, source_data[key])


def tensors_match_device(
        data: torch.Tensor | list | tuple | dict,
        device: torch.device,
        include_parameter_indices: list[int] | None = None,
) -> bool:
    """Check if all tensors are on the specified device."""
    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        if not device_equals(data.device, device):
            return False
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                if not tensors_match_device(elem, device):
                    return False
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            if not tensors_match_device(elem, device):
                return False

    return True


def tensors_record_stream(
        stream: torch.Stream,
        data: torch.Tensor | list | tuple | dict,
        include_parameter_indices: list[int] | None = None,
):
    """Record tensors on a CUDA stream."""
    if isinstance(data, torch.Tensor):
        if data.device.type == "cuda":
            data.record_stream(stream)
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                tensors_record_stream(stream, elem, [])
    elif isinstance(data, dict):
        for elem in data.values():
            tensors_record_stream(stream, elem)


def device_equals(device1: torch.device, device2: torch.device) -> bool:
    """Check if two devices are equal."""
    return (device1 is not None and device2 is not None
            and device1.type == device2.type
            and (0 if device1.index is None else device1.index) == (0 if device2.index is None else device2.index))


def torch_gc():
    """Run garbage collection and empty CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        if torch_version > packaging.version.Version("2.6.0"):
            try:
                torch._C._host_emptyCache()
            except AttributeError:
                pass

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def create_stream_context(stream: torch.cuda.Stream) -> torch.cuda.StreamContext | nullcontext:
    """Create a stream context for CUDA operations."""
    if isinstance(stream, torch.cuda.Stream):
        return torch.cuda.StreamContext(stream)
    return nullcontext()


def pin_tensor_(x):
    """Pin tensor memory for faster CPU<->GPU transfers."""
    if torch.cuda.is_available():
        cudart = torch.cuda.cudart()
        err = cudart.cudaHostRegister(
            x.data_ptr(),
            x.numel() * x.element_size(),
            0,
        )

        if err.value != 0:
            raise RuntimeError(
                f"CUDA Error while trying to pin memory. error: {err.value}, "
                f"ptr: {x.data_ptr()}, size: {x.numel() * x.element_size()}"
            )


def unpin_tensor_(x):
    """Unpin tensor memory."""
    if torch.cuda.is_available():
        cudart = torch.cuda.cudart()
        err = cudart.cudaHostUnregister(x.data_ptr())

        if err.value != 0:
            raise RuntimeError(
                f"CUDA Error while trying to unpin memory. error {err.value}, ptr: {x.data_ptr()}"
            )


def get_offload_tensors(module: nn.Module) -> list[torch.Tensor]:
    """Get tensors that can be offloaded from a module."""
    tensors = []

    if isinstance(module, (nn.Linear, nn.Conv2d)):
        tensors.append(module.weight)
    if isinstance(module, nn.Linear) and module.bias is not None:
        tensors.append(module.bias)

    return tensors


def get_offload_tensor_bytes(module: nn.Module) -> int:
    """Get total bytes of offloadable tensors in a module."""
    tensors = get_offload_tensors(module)
    return sum(t.element_size() * t.numel() for t in tensors)


def offload_tensors(
        module: nn.Module,
        device: torch.device,
        non_blocking: bool = False,
        allocator: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    """Offload module tensors to a device."""
    tensors = get_offload_tensors(module)

    if allocator is None:
        for tensor in tensors:
            tensor.data = tensor.data.to(device=device, non_blocking=non_blocking)
    else:
        for tensor in tensors:
            new_tensor = allocator(tensor)
            new_tensor.copy_(tensor.data, non_blocking=non_blocking)
            tensor.data = new_tensor
