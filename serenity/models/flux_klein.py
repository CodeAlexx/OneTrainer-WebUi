"""Compatibility shim for legacy `FluxKleinModel` imports.

`FluxKleinModel` now aliases `Flux2KleinModel` so legacy imports continue to
resolve to the FLUX.2 Klein implementation (not Flux Schnell / FLUX.1).
"""

from __future__ import annotations

from serenity.models.flux2_klein import Flux2KleinModel


class FluxKleinModel(Flux2KleinModel):
    """Deprecated alias for `Flux2KleinModel`."""
