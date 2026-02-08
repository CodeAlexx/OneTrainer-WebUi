"""Model loading, eviction, and VRAM budget management."""

from __future__ import annotations

import gc
import logging
import sys
import weakref
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from serenity.inference.memory.vram import (
    VRAMBudget,
    calculate_budget,
    get_free_memory,
    is_cuda_available,
)

__all__ = [
    "LoadedModel",
    "ModelManager",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _same_device(a: torch.device, b: torch.device) -> bool:
    """Compare two devices, treating ``cuda`` and ``cuda:0`` as equal."""
    if a.type != b.type:
        return False
    # When index is None, treat as index 0 for cuda devices.
    ai = a.index if a.index is not None else 0
    bi = b.index if b.index is not None else 0
    if a.type == "cpu":
        return True
    return ai == bi


def _module_size(module: nn.Module) -> int:
    """Total size of all parameters + buffers in bytes."""
    total = 0
    for p in module.parameters():
        total += p.data.nbytes
    for b in module.buffers():
        total += b.nbytes
    return total


def _module_gpu_size(module: nn.Module, device: torch.device) -> int:
    """Bytes currently residing on *device*."""
    total = 0
    for p in module.parameters():
        if _same_device(p.device, device):
            total += p.data.nbytes
    for b in module.buffers():
        if _same_device(b.device, device):
            total += b.nbytes
    return total


# ---------------------------------------------------------------------------
# LoadedModel
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """Tracks a model that has (partially) been loaded onto a device."""

    device: torch.device
    total_size: int
    loaded_size: int = 0
    config_hash: str = ""

    # Internal — we hold a weakref to the module so we don't prevent GC.
    _model_ref: weakref.ref[nn.Module] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        device: torch.device,
        config_hash: str = "",
    ) -> LoadedModel:
        """Create a ``LoadedModel`` wrapping *model*."""
        total = _module_size(model)
        loaded = _module_gpu_size(model, device)
        inst = cls(
            device=device,
            total_size=total,
            loaded_size=loaded,
            config_hash=config_hash,
            _model_ref=weakref.ref(model),
        )
        return inst

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module | None:
        """The underlying module, or ``None`` if it was garbage-collected."""
        if self._model_ref is None:
            return None
        return self._model_ref()

    @property
    def offloaded_size(self) -> int:
        """Bytes that are *not* on the target device."""
        return max(0, self.total_size - self.loaded_size)

    @property
    def is_alive(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load_to_gpu(self, budget: int) -> int:
        """Move up to *budget* bytes of parameters onto :attr:`device`.

        Returns the number of bytes actually transferred.
        """
        model = self.model
        if model is None:
            return 0

        moved = 0
        for p in model.parameters():
            if _same_device(p.device, self.device):
                continue
            p_size = p.data.nbytes
            if moved + p_size > budget:
                continue
            p.data = p.data.to(self.device, non_blocking=True)
            moved += p_size

        self.loaded_size += moved
        return moved

    def unload_from_gpu(self, bytes_to_free: int) -> int:
        """Move parameters off :attr:`device` until *bytes_to_free* is freed.

        Weights are moved to CPU.  Returns bytes actually freed.
        """
        model = self.model
        if model is None:
            return 0

        freed = 0
        cpu = torch.device("cpu")
        for p in model.parameters():
            if freed >= bytes_to_free:
                break
            if _same_device(p.device, self.device):
                p_size = p.data.nbytes
                p.data = p.data.to(cpu, non_blocking=True)
                freed += p_size

        self.loaded_size = max(0, self.loaded_size - freed)
        return freed


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages a set of loaded models and their VRAM budget.

    Provides smart eviction when VRAM runs low: models are scored by
    ``(-offloaded_memory, refcount, total_memory)`` and the lowest-scoring
    model is evicted first.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        if device is None:
            device = torch.device("cuda") if is_cuda_available() else torch.device("cpu")
        self.device = device
        self.loaded_models: list[LoadedModel] = []

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        model: nn.Module,
        budget: int | None = None,
        config_hash: str = "",
    ) -> LoadedModel:
        """Load *model* onto the managed device, respecting *budget*.

        If *budget* is ``None``, the full VRAM budget from
        :func:`calculate_budget` is used.
        """
        # Re-use an existing entry with matching config hash.
        existing = self.get_loaded(config_hash) if config_hash else None
        if existing is not None and existing.is_alive:
            return existing

        if budget is None:
            vram = calculate_budget(self.device)
            budget = vram.available

        lm = LoadedModel.from_module(model, self.device, config_hash=config_hash)
        lm.load_to_gpu(budget)
        self.loaded_models.append(lm)
        return lm

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_loaded(self, config_hash: str) -> LoadedModel | None:
        """Find an already-loaded model by its *config_hash*."""
        if not config_hash:
            return None
        for lm in self.loaded_models:
            if lm.config_hash == config_hash and lm.is_alive:
                return lm
        return None

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _eviction_key(self, lm: LoadedModel) -> tuple[int, int, int]:
        """Scoring tuple: models with the *lowest* score are evicted first.

        ``(-offloaded_memory, refcount, total_memory)``

        Models that are mostly offloaded already contribute less VRAM savings
        and so have a lower score.  A lower refcount also means less live
        interest from the rest of the program.
        """
        model = lm.model
        refcount = sys.getrefcount(model) if model is not None else 0
        return (-lm.offloaded_size, refcount, lm.total_size)

    def free_memory(self, bytes_needed: int) -> int:
        """Evict models until at least *bytes_needed* VRAM is freed.

        Returns the total bytes actually freed.
        """
        # Purge dead entries first.
        self.loaded_models = [lm for lm in self.loaded_models if lm.is_alive]

        freed = 0
        candidates = sorted(self.loaded_models, key=self._eviction_key)
        for lm in candidates:
            if freed >= bytes_needed:
                break
            freed += lm.unload_from_gpu(bytes_needed - freed)

        if freed > 0 and is_cuda_available():
            torch.cuda.empty_cache()
            gc.collect()

        return freed

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def unload_all(self) -> None:
        """Unload every tracked model from GPU."""
        for lm in self.loaded_models:
            if lm.is_alive:
                lm.unload_from_gpu(lm.loaded_size)
        self.loaded_models.clear()

        if is_cuda_available():
            torch.cuda.empty_cache()
            gc.collect()
