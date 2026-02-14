"""Block registry for the stagehand runtime.

Maps model topology to an ordered set of block entries used by the
scheduler, residency map, and transfer engine.  Built once at model load
time by walking ``model.named_modules()`` and matching against a caller-
supplied pattern.  Immutable after construction + validation.

First target model: WAN 2.2 (video diffusion with temporal + spatial
attention blocks).
"""
from __future__ import annotations

import json
import re
import struct
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from serenity.stagehand.errors import StagehandOOMError

if TYPE_CHECKING:
    pass

__all__ = [
    "BlockEntry",
    "BlockRegistry",
    "FileParamSpec",
    "ParamLayoutEntry",
]


ParamLayoutEntry = tuple[str, tuple[int, ...], torch.dtype, int, int]


@dataclass(frozen=True)
class FileParamSpec:
    """Descriptor for a parameter sourced from safetensors on disk."""

    param_name: str
    file_offset: int
    source_nbytes: int
    source_dtype: torch.dtype
    source_shape: tuple[int, ...]


# ── helpers ──────────────────────────────────────────────────────────────


def _param_size_bytes(module: nn.Module, dtype: torch.dtype) -> int:
    """Sum of all parameter sizes in *module* when stored as *dtype*."""
    total = 0
    for p in module.parameters():
        total += p.numel() * dtype.itemsize
    return total


# ── data ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BlockEntry:
    """Immutable descriptor for a single swappable block.

    Mirrors spec Section 2.2.1.  ``module_ref`` is a weak reference so the
    registry never prevents garbage collection of the underlying module.
    """

    block_id: str
    module_ref: weakref.ref  # weak reference to nn.Module
    size_bytes: int
    dtype: torch.dtype
    dependencies: tuple[str, ...]
    group: str
    exec_order: int
    quant_format: str | None = None
    quant_meta_bytes: int = 0
    # File-backed mode metadata. Empty for module-backed blocks.
    source_path: str | None = None
    param_layout: tuple[ParamLayoutEntry, ...] = field(default_factory=tuple)
    file_param_specs: tuple[FileParamSpec, ...] = field(default_factory=tuple)
    module_param_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def file_backed(self) -> bool:
        return self.source_path is not None and len(self.file_param_specs) > 0


# ── registry ─────────────────────────────────────────────────────────────


class BlockRegistry:
    """Ordered, immutable registry of swappable blocks.

    Typical usage::

        registry = BlockRegistry()
        registry.build_from_model(wan_model, block_pattern="spatial|temporal", group="wan", dtype=torch.bfloat16)
        registry.validate(pool_capacity_bytes=8 * 1024**3)
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[str, BlockEntry] = OrderedDict()
        self._frozen: bool = False

    @staticmethod
    def _decode_safetensors_dtype(dtype_name: str) -> torch.dtype:
        mapping: dict[str, torch.dtype] = {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F64": torch.float64,
            "I8": torch.int8,
            "I16": torch.int16,
            "I32": torch.int32,
            "I64": torch.int64,
            "U8": torch.uint8,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported safetensors dtype {dtype_name!r}")
        return mapping[dtype_name]

    @staticmethod
    def _parse_safetensors_index(
        safetensors_path: Path,
    ) -> dict[str, tuple[int, int, torch.dtype, tuple[int, ...]]]:
        """Return tensor index map for a safetensors file.

        Values are ``(abs_offset, nbytes, source_dtype, source_shape)``.
        """
        with safetensors_path.open("rb") as handle:
            header_len_raw = handle.read(8)
            if len(header_len_raw) != 8:
                raise ValueError(f"Invalid safetensors header for {safetensors_path}")
            header_len = struct.unpack("<Q", header_len_raw)[0]
            header_bytes = handle.read(header_len)
            if len(header_bytes) != header_len:
                raise ValueError(f"Truncated safetensors header for {safetensors_path}")

        header = json.loads(header_bytes.decode("utf-8"))
        data_base = 8 + header_len
        index: dict[str, tuple[int, int, torch.dtype, tuple[int, ...]]] = {}
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(value, dict):
                continue
            offsets = value.get("data_offsets")
            shape = value.get("shape")
            dtype_name = value.get("dtype")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not isinstance(shape, list)
                or not isinstance(dtype_name, str)
            ):
                continue
            start_rel, end_rel = int(offsets[0]), int(offsets[1])
            if end_rel < start_rel:
                raise ValueError(f"Invalid data_offsets for key {key!r} in {safetensors_path}")
            source_dtype = BlockRegistry._decode_safetensors_dtype(dtype_name)
            source_shape = tuple(int(dim) for dim in shape)
            index[key] = (
                data_base + start_rel,
                end_rel - start_rel,
                source_dtype,
                source_shape,
            )
        return index

    @staticmethod
    def _candidate_tensor_keys(block_id: str, param_name: str) -> tuple[str, ...]:
        """Return likely safetensors keys for a module parameter name.

        Adapter wrappers may rename base parameters (for example
        ``to_q.orig.weight``). The checkpoint keeps the original key
        (``to_q.weight``). This helper generates normalized candidates.
        """
        names: list[str] = []
        seen: set[str] = set()
        queue: list[str] = [param_name]
        tokens = (".orig.", ".org_module.", ".base_layer.")
        aliases = (
            ("attn1.", "self_attn."),
            ("attn2.", "cross_attn."),
            ("to_q.", "q."),
            ("to_k.", "k."),
            ("to_v.", "v."),
            ("to_out.0.", "o."),
            ("ffn.net.0.proj.", "ffn.0."),
            ("ffn.net.2.", "ffn.2."),
            ("norm2.", "norm3."),
            (".attn1.", ".self_attn."),
            (".attn2.", ".cross_attn."),
            (".to_q.", ".q."),
            (".to_k.", ".k."),
            (".to_v.", ".v."),
            (".to_out.0.", ".o."),
            (".ffn.net.0.proj.", ".ffn.0."),
            (".ffn.net.2.", ".ffn.2."),
            (".norm2.", ".norm3."),
            ("scale_shift_table", "modulation"),
        )

        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            names.append(current)

            for token in tokens:
                if token not in current:
                    continue
                normalized = current.replace(token, ".")
                normalized = normalized.replace("..", ".").strip(".")
                if normalized and normalized not in seen:
                    queue.append(normalized)

            for src, dst in aliases:
                if src not in current:
                    continue
                alias = current.replace(src, dst)
                alias = alias.replace("..", ".").strip(".")
                if alias and alias not in seen:
                    queue.append(alias)

        return tuple(f"{block_id}.{name}" for name in names)

    # ── construction ─────────────────────────────────────────────────

    def build_from_model(
        self,
        model: nn.Module,
        block_pattern: str,
        group: str,
        dtype: torch.dtype,
    ) -> None:
        """Walk *model* and register every module whose name matches *block_pattern*.

        Parameters
        ----------
        model:
            The PyTorch model to scan.
        block_pattern:
            Regex pattern matched against the fully-qualified module name
            (as returned by ``named_modules()``).
        group:
            Logical group label (e.g. ``"wan"``, ``"dit"``, ``"te1"``).
        dtype:
            Target dtype used to compute ``size_bytes``.

        Raises
        ------
        RuntimeError
            If the registry has already been frozen.
        """
        if self._frozen:
            raise RuntimeError("Cannot build_from_model on a frozen registry")

        compiled = re.compile(block_pattern)
        order = len(self._entries)

        for name, module in model.named_modules():
            if not name:
                continue
            if compiled.search(name):
                block_id = name
                size = _param_size_bytes(module, dtype)
                entry = BlockEntry(
                    block_id=block_id,
                    module_ref=weakref.ref(module),
                    size_bytes=size,
                    dtype=dtype,
                    dependencies=(),
                    group=group,
                    exec_order=order,
                )
                self._entries[block_id] = entry
                order += 1

    def validate(self, pool_capacity_bytes: int) -> None:
        """Check all blocks fit in the pool and freeze the registry.

        Raises
        ------
        StagehandOOMError
            If any single block exceeds *pool_capacity_bytes*.
        """
        for entry in self._entries.values():
            if entry.size_bytes > pool_capacity_bytes:
                raise StagehandOOMError(
                    f"Block {entry.block_id!r} ({entry.size_bytes} bytes) "
                    f"exceeds pool capacity ({pool_capacity_bytes} bytes)"
                )
        self._frozen = True

    def convert_to_file_backed(
        self,
        safetensors_path: str | Path,
        *,
        drop_module_tensors: bool = True,
    ) -> int:
        """Convert eligible block params to file-backed descriptors.

        This keeps the module scaffold (for hooks / topology) while dropping
        frozen parameter tensors from CPU RAM.

        Returns
        -------
        int
            Number of parameters converted to file-backed references.
        """
        if not self._frozen:
            raise RuntimeError("convert_to_file_backed requires a validated (frozen) registry")

        path = Path(safetensors_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"safetensors file not found: {path}")

        tensor_index = self._parse_safetensors_index(path)
        converted_params = 0
        new_entries: OrderedDict[str, BlockEntry] = OrderedDict()

        for block_id, entry in self._entries.items():
            module = entry.module_ref()
            if module is None:
                new_entries[block_id] = entry
                continue

            layout: list[ParamLayoutEntry] = []
            file_specs: list[FileParamSpec] = []
            module_param_names: list[str] = []
            offset = 0

            for param_name, param in module.named_parameters():
                shape = tuple(int(dim) for dim in param.shape)
                numel = int(param.numel())
                nbytes = numel * entry.dtype.itemsize
                layout.append((param_name, shape, entry.dtype, offset, numel))
                offset += nbytes

                if param.requires_grad:
                    module_param_names.append(param_name)
                    continue

                tensor_meta = None
                for tensor_key in self._candidate_tensor_keys(block_id, param_name):
                    tensor_meta = tensor_index.get(tensor_key)
                    if tensor_meta is not None:
                        break
                if tensor_meta is None:
                    module_param_names.append(param_name)
                    continue

                file_offset, source_nbytes, source_dtype, source_shape = tensor_meta
                if source_shape != shape:
                    module_param_names.append(param_name)
                    continue

                file_specs.append(
                    FileParamSpec(
                        param_name=param_name,
                        file_offset=file_offset,
                        source_nbytes=source_nbytes,
                        source_dtype=source_dtype,
                        source_shape=source_shape,
                    )
                )
                converted_params += 1

                if drop_module_tensors:
                    with torch.no_grad():
                        param.data = torch.empty(0, dtype=param.dtype, device="cpu")
                        if param.grad is not None:
                            param.grad = None

            if file_specs:
                new_entry = replace(
                    entry,
                    source_path=str(path),
                    param_layout=tuple(layout),
                    file_param_specs=tuple(file_specs),
                    module_param_names=tuple(module_param_names),
                )
                new_entries[block_id] = new_entry
            else:
                new_entries[block_id] = entry

        self._entries = new_entries
        return converted_params

    # ── queries ──────────────────────────────────────────────────────

    def get(self, block_id: str) -> BlockEntry:
        """Return the entry for *block_id*.

        Raises
        ------
        KeyError
            If *block_id* is not in the registry.
        """
        return self._entries[block_id]

    def blocks_in_order(self) -> list[BlockEntry]:
        """All entries sorted by ``exec_order``."""
        return sorted(self._entries.values(), key=lambda e: e.exec_order)

    def __len__(self) -> int:
        return len(self._entries)

    def groups(self) -> dict[str, list[BlockEntry]]:
        """Entries grouped by their ``group`` field."""
        result: dict[str, list[BlockEntry]] = {}
        for entry in self._entries.values():
            result.setdefault(entry.group, []).append(entry)
        return result

    def __contains__(self, block_id: str) -> bool:
        return block_id in self._entries

    def __repr__(self) -> str:
        return f"BlockRegistry(blocks={len(self)}, frozen={self._frozen})"
