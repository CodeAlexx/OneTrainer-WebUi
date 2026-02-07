"""Quantization utilities for model weight compression.

Supports:

- **NF4** quantization via bitsandbytes (4-bit NormalFloat)
- **FP8** quantization (float8_e4m3fn per-tensor scaling)
- **INT8** quantization (int8 per-tensor or per-axis)
- **GGUF** activation quantization (A8 int or float, requires ``gguf`` package)
- **SVD** decomposition layers (optional rank-based compression)

All external dependencies (bitsandbytes, gguf, accelerate) are conditionally
imported so the module loads without them.

Usage::

    from serenity.models.quantization import quantize_model, QuantMethod

    quantize_model(my_unet, method=QuantMethod.NF4, device=torch.device("cuda"))
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None  # type: ignore[assignment]

try:
    import accelerate
except ImportError:
    accelerate = None  # type: ignore[assignment]

try:
    import gguf as _gguf_pkg  # noqa: F401
    from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor
except ImportError:
    _gguf_pkg = None  # type: ignore[assignment]
    GGUFLinear = None  # type: ignore[assignment]
    dequantize_gguf_tensor = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QuantMethod(str, Enum):
    """Supported quantization methods."""

    NONE = "none"
    NF4 = "nf4"
    INT8 = "int8"
    FP8 = "fp8"
    W8A8_INT = "w8a8_int"
    W8A8_FP = "w8a8_fp"
    GGUF_A8_INT = "gguf_a8_int"
    GGUF_A8_FP = "gguf_a8_fp"


# ---------------------------------------------------------------------------
# Mixin ABCs for quantized modules
# ---------------------------------------------------------------------------

class QuantizedModuleMixin(ABC):
    """Mixin for modules that support quantization."""

    @abstractmethod
    def quantize(self, device: torch.device | None = None) -> None: ...


class QuantizedLinearMixin(ABC):
    """Mixin for quantized linear layers that can dequantize on demand."""

    @abstractmethod
    def original_weight_shape(self) -> tuple[int, ...]: ...

    @abstractmethod
    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> Tensor: ...


# ---------------------------------------------------------------------------
# Quantization math helpers
# ---------------------------------------------------------------------------

def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    """Quantize a tensor to int8 with the given scale."""
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, float]:
    """Symmetric per-tensor int8 quantization."""
    abs_max = x.abs().max()
    scale = float((abs_max.float() / 127.0).clamp(min=1e-30).item())
    q = quantize_int8(x, scale)
    return q, scale


def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Symmetric per-axis int8 quantization."""
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = quantize_int8(x, scale)
    return q, scale


def quantize_fp8(x: Tensor, scale: float | Tensor) -> Tensor:
    """Quantize a tensor to float8_e4m3fn with the given scale."""
    return x.float().mul(1.0 / scale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)


def quantize_fp8_tensorwise(x: Tensor) -> tuple[Tensor, float]:
    """Symmetric per-tensor FP8 quantization."""
    abs_max = x.abs().max()
    scale = float((abs_max.float() / 448.0).clamp(min=1e-30).item())
    q = quantize_fp8(x, scale)
    return q, scale


def quantize_fp8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Symmetric per-axis FP8 quantization."""
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 448.0).clamp(min=1e-30)
    q = quantize_fp8(x, scale)
    return q, scale


def dequantize(q: Tensor, scale: float | Tensor) -> Tensor:
    """Dequantize int8 or fp8 back to float."""
    return q.float() * scale


# ---------------------------------------------------------------------------
# NF4 linear layer (bitsandbytes)
# ---------------------------------------------------------------------------

class LinearNf4(nn.Linear, QuantizedModuleMixin, QuantizedLinearMixin):
    """4-bit NormalFloat quantized linear layer using bitsandbytes.

    Requires ``bitsandbytes`` to be installed.
    """

    is_quantized: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if bnb is None:
            raise ImportError(
                "bitsandbytes is required for NF4 quantization. "
                "Install with: pip install bitsandbytes"
            )
        super().__init__(*args, **kwargs)
        self.is_quantized = False
        self.block_size = 64
        self.nested_block_size = 256

        self._absmax = torch.empty(size=())
        self._offset = torch.empty(size=())
        self._code = torch.empty(size=())
        self._nested_absmax = torch.empty(size=())
        self._nested_code = torch.empty(size=())
        self.shape = self.weight.shape

        self.register_buffer("absmax", self._absmax)
        self.register_buffer("offset", self._offset)
        self.register_buffer("code", self._code)
        self.register_buffer("nested_absmax", self._nested_absmax)
        self.register_buffer("nested_code", self._nested_code)

        self.compute_dtype: torch.dtype | None = None
        self.quant_state: Any = None

    def original_weight_shape(self) -> tuple[int, ...]:
        return tuple(self.shape)

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.is_quantized:
            device_weight = self.weight.to(device=device)
            device_absmax = self._absmax.to(device=device)
            return bnb.functional.dequantize_4bit(
                A=device_weight,
                quant_state=bnb.functional.QuantState(
                    absmax=device_absmax,
                    shape=self.shape,
                    code=self._code,
                    blocksize=self.block_size,
                    quant_type="nf4",
                    dtype=self.compute_dtype,
                    offset=self._offset,
                    state2=self.quant_state.state2,
                ),
                quant_type="nf4",
            ).detach().to(dtype=dtype)
        return self.weight.detach().to(dtype=dtype)

    def quantize(self, device: torch.device | None = None) -> None:
        if self.is_quantized:
            return
        self.is_quantized = True

        weight = self.weight.data
        orig_device = weight.device
        if weight.dtype != torch.int8:
            if device is not None:
                weight = weight.to(device=device)

            weight, quant_state = bnb.functional.quantize_4bit(
                weight,
                blocksize=self.block_size,
                compress_statistics=True,
                quant_type="nf4",
                quant_storage=torch.uint8,
            )

            self._absmax.data = quant_state.absmax
            self._offset.data = quant_state.offset
            self._code.data = quant_state.code
            self._nested_absmax.data = quant_state.state2.absmax
            self._nested_code.data = quant_state.state2.code

            if device is not None:
                weight = weight.to(device=orig_device)

        self.requires_grad_(False)
        self.weight.data = weight

        self.quant_state = bnb.functional.QuantState(
            absmax=self._absmax,
            shape=self.shape,
            code=self._code,
            blocksize=self.block_size,
            quant_type="nf4",
            dtype=self.compute_dtype,
            offset=self._offset,
            state2=bnb.functional.QuantState(
                absmax=self._nested_absmax,
                shape=None,
                code=self._nested_code,
                blocksize=self.nested_block_size,
                quant_type="nf4",
                dtype=torch.float32,
                offset=None,
                state2=None,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        x = x.to(dtype=self.compute_dtype)
        x = bnb.matmul_4bit(x, self.weight.t(), bias=self.bias, quant_state=self.quant_state)
        return x.to(dtype=orig_dtype)


# ---------------------------------------------------------------------------
# FP8 linear layer
# ---------------------------------------------------------------------------

class LinearFp8(nn.Linear, QuantizedModuleMixin, QuantizedLinearMixin):
    """FP8 (float8_e4m3fn) quantized linear layer."""

    is_quantized: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.is_quantized = False
        self.fp8_dtype = torch.float8_e4m3fn
        self._scale = torch.tensor(1.0, dtype=torch.float)
        self.register_buffer("scale", self._scale)
        self.compute_dtype: torch.dtype | None = None

    def original_weight_shape(self) -> tuple[int, ...]:
        return tuple(self.weight.shape)

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self._scale is not None:
            return self.weight.detach().to(dtype) * self._scale.to(dtype=dtype)
        return self.weight.detach().to(dtype=dtype)

    def quantize(self, device: torch.device | None = None) -> None:
        if self.is_quantized:
            return
        self.is_quantized = True

        weight = self.weight.data
        orig_device = weight.device
        if weight.dtype != self.fp8_dtype:
            if device is not None:
                weight = weight.to(device=device)

            abs_max = weight.abs().max()
            self._scale.copy_(
                torch.clamp(abs_max, min=1e-12) / torch.finfo(self.fp8_dtype).max
            )
            weight = weight.div_(self._scale).to(dtype=self.fp8_dtype)

            if device is not None:
                weight = weight.to(device=orig_device)
        self.weight.data = weight

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.detach()
        weight = weight.to(
            dtype=self.compute_dtype if self.compute_dtype is not None else x.dtype
        )
        if self._scale is not None:
            weight = weight.mul_(self._scale)
        return nn.functional.linear(x, weight, self.bias)


# ---------------------------------------------------------------------------
# Layer replacement engine
# ---------------------------------------------------------------------------

def _create_quantized_linear(
    construct_fn: Any,
    module: nn.Linear,
    copy_parameters: bool,
) -> nn.Module:
    """Create a quantized linear layer matching the shape of *module*."""
    bias = module.bias is not None

    if accelerate is not None:
        with accelerate.init_empty_weights():
            quant = construct_fn(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=bias,
            )
    else:
        quant = construct_fn(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=bias,
        )

    if copy_parameters:
        quant.weight = type(quant.weight)(module.weight, requires_grad=False)
        if bias and module.bias is not None:
            quant.bias = type(quant.bias)(module.bias, requires_grad=False)

    return quant


def _replace_linear_layers(
    parent: nn.Module,
    construct_fn: Any,
    *,
    keep_in_fp32: list[str] | None = None,
    copy_parameters: bool = False,
    name_prefix: str = "",
    visited: set[int] | None = None,
    convert_type: type = nn.Linear,
) -> None:
    """Recursively replace ``nn.Linear`` layers with quantized versions."""
    if keep_in_fp32 is None:
        keep_in_fp32 = []
    if visited is None:
        visited = set()

    visited.add(id(parent))

    if isinstance(parent, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
        items = parent.items() if isinstance(parent, nn.ModuleDict) else enumerate(parent)
        for key, module in items:
            if isinstance(module, convert_type):
                quant = _create_quantized_linear(construct_fn, module, copy_parameters)
                parent[key] = quant  # type: ignore[index]
                del module
            elif id(module) not in visited:
                _replace_linear_layers(
                    module, construct_fn,
                    keep_in_fp32=keep_in_fp32,
                    copy_parameters=copy_parameters,
                    name_prefix=f"{name_prefix}[{key}]",
                    visited=visited,
                    convert_type=convert_type,
                )
    else:
        for attr_name in list(dir(parent)):
            if attr_name in keep_in_fp32:
                continue
            module = getattr(parent, attr_name, None)
            if isinstance(module, convert_type):
                quant = _create_quantized_linear(construct_fn, module, copy_parameters)
                setattr(parent, attr_name, quant)
                del module
            elif isinstance(module, nn.Module) and id(module) not in visited:
                child_prefix = attr_name if not name_prefix else f"{name_prefix}.{attr_name}"
                _replace_linear_layers(
                    module, construct_fn,
                    keep_in_fp32=keep_in_fp32,
                    copy_parameters=copy_parameters,
                    name_prefix=child_prefix,
                    visited=visited,
                    convert_type=convert_type,
                )


# ---------------------------------------------------------------------------
# Public API: quantize_model
# ---------------------------------------------------------------------------

def quantize_model(
    model: nn.Module,
    method: QuantMethod | str,
    device: torch.device | None = None,
    *,
    bits: int | None = None,
    keep_in_fp32: list[str] | None = None,
    copy_parameters: bool = True,
    compute_dtype: torch.dtype | None = None,
) -> nn.Module:
    """Quantize a model's linear layers in-place.

    Parameters
    ----------
    model:
        The ``nn.Module`` whose linear layers will be replaced.
    method:
        Quantization method (see ``QuantMethod``).
    device:
        Device used for quantization computation.
    bits:
        Bit width override (currently informational, method determines actual bits).
    keep_in_fp32:
        List of module attribute names to keep in FP32.
    copy_parameters:
        Whether to copy existing weights into quantized layers.
    compute_dtype:
        Runtime compute dtype for quantized forward passes.

    Returns
    -------
    nn.Module
        The same model, mutated in-place with quantized layers.
    """
    if isinstance(method, str):
        method = QuantMethod(method.lower())

    if method == QuantMethod.NONE:
        return model

    from functools import partial

    if method == QuantMethod.NF4:
        if bnb is None:
            raise ImportError(
                "bitsandbytes is required for NF4 quantization. "
                "Install with: pip install bitsandbytes"
            )
        construct_fn = partial(LinearNf4)
        _replace_linear_layers(
            model, construct_fn,
            keep_in_fp32=keep_in_fp32,
            copy_parameters=copy_parameters,
        )

    elif method == QuantMethod.INT8:
        if bnb is None:
            raise ImportError(
                "bitsandbytes is required for INT8 quantization. "
                "Install with: pip install bitsandbytes"
            )
        construct_fn = partial(bnb.nn.Linear8bitLt, has_fp16_weights=False)
        _replace_linear_layers(
            model, construct_fn,
            keep_in_fp32=keep_in_fp32,
            copy_parameters=copy_parameters,
        )

    elif method == QuantMethod.FP8:
        construct_fn = partial(LinearFp8)
        _replace_linear_layers(
            model, construct_fn,
            keep_in_fp32=keep_in_fp32,
            copy_parameters=copy_parameters,
        )

    elif method in (QuantMethod.GGUF_A8_INT, QuantMethod.GGUF_A8_FP):
        if GGUFLinear is None:
            raise ImportError(
                "gguf and diffusers are required for GGUF quantization. "
                "Install with: pip install gguf diffusers"
            )
        # GGUF layers replace existing GGUFLinear (not nn.Linear)
        logger.info("GGUF quantization is applied at load time, not via replace")
        return model

    elif method in (QuantMethod.W8A8_INT, QuantMethod.W8A8_FP):
        logger.info("W8A8 quantization requires custom LinearW8A8 layers")
        return model

    else:
        raise ValueError(f"Unsupported quantization method: {method}")

    # Apply quantization to replaced layers
    _apply_quantization(model, device=device, compute_dtype=compute_dtype)

    logger.info("Quantized model using method=%s", method.value)
    return model


def _apply_quantization(
    module: nn.Module,
    device: torch.device | None = None,
    compute_dtype: torch.dtype | None = None,
) -> None:
    """Walk the module tree and call ``.quantize()`` on quantized layers."""
    for child in module.modules():
        if isinstance(child, QuantizedModuleMixin):
            if hasattr(child, "compute_dtype") and compute_dtype is not None:
                child.compute_dtype = compute_dtype  # type: ignore[attr-defined]
            child.quantize(device=device)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def is_quantized_parameter(module: nn.Module, parameter_name: str) -> bool:
    """Check whether a parameter in *module* is from a quantized layer.

    Useful for filtering parameters that should not receive gradients or
    be included in optimizer state.
    """
    if isinstance(module, QuantizedLinearMixin):
        # NF4 has multiple quantization buffers
        if isinstance(module, LinearNf4):
            return parameter_name in {
                "weight", "absmax", "offset", "code",
                "nested_absmax", "nested_code",
            }
        # FP8 / generic
        if parameter_name == "weight":
            return True

    # bitsandbytes 8-bit
    if bnb is not None and isinstance(module, bnb.nn.Linear8bitLt):
        return parameter_name == "weight"

    return False


def get_unquantized_weight(
    module: nn.Linear,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Get the dequantized weight from a (possibly quantized) linear layer."""
    if isinstance(module, QuantizedLinearMixin):
        return module.unquantized_weight(dtype, device)
    if GGUFLinear is not None and isinstance(module, GGUFLinear):
        return dequantize_gguf_tensor(module.weight).to(dtype=dtype)
    return module.weight.detach().to(dtype=dtype)


def get_offload_tensors(module: nn.Module) -> list[Tensor]:
    """Get tensors that should be offloaded for a given module.

    Used by layer offloading to know which tensors to move between devices.
    """
    tensors: list[Tensor] = []

    if bnb is not None and isinstance(module, LinearNf4):
        if hasattr(module, "quant_state") and module.quant_state is not None:
            tensors.append(module.quant_state.absmax)

    if isinstance(module, (nn.Linear, nn.Conv2d)):
        tensors.append(module.weight)
    if isinstance(module, nn.Linear) and module.bias is not None:
        tensors.append(module.bias)

    return tensors


def get_offload_tensor_bytes(module: nn.Module) -> int:
    """Total bytes of offloadable tensors in a module."""
    return sum(t.element_size() * t.numel() for t in get_offload_tensors(module))


__all__ = [
    # Enums / types
    "QuantMethod",
    "QuantizedModuleMixin",
    "QuantizedLinearMixin",
    # Layer classes
    "LinearNf4",
    "LinearFp8",
    # Math helpers
    "quantize_int8",
    "quantize_int8_tensorwise",
    "quantize_int8_axiswise",
    "quantize_fp8",
    "quantize_fp8_tensorwise",
    "quantize_fp8_axiswise",
    "dequantize",
    # High-level API
    "quantize_model",
    # Utilities
    "is_quantized_parameter",
    "get_unquantized_weight",
    "get_offload_tensors",
    "get_offload_tensor_bytes",
]
