"""Config versioning and migration system.

Provides version-based migration for Serenity training configs, modeled
after OneTrainer's ``BaseConfig.from_dict`` migration chain.  Each saved
config embeds a ``__version`` key.  When loading, the system applies
registered migration functions in sequence to bring the data up to the
current schema version.

Usage::

    from serenity.core.config_migration import migrate_config, CURRENT_VERSION

    raw = json.loads(config_path.read_text())
    migrated = migrate_config(raw)  # brings to CURRENT_VERSION
    config = TrainConfig(**migrated)
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Current schema version
# ---------------------------------------------------------------------------

CURRENT_VERSION: int = 1

# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------

_migrations: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def register_migration(
    from_version: int,
) -> Callable[[Callable[[dict[str, Any]], dict[str, Any]]], Callable[[dict[str, Any]], dict[str, Any]]]:
    """Decorator to register a migration function for a specific version.

    The decorated function receives a config dict at ``from_version`` and
    must return a dict compatible with ``from_version + 1``.

    Example::

        @register_migration(0)
        def _migrate_0_to_1(data: dict) -> dict:
            data["new_field"] = data.pop("old_field", "default")
            return data
    """
    def decorator(
        fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        if from_version in _migrations:
            raise ValueError(
                f"Migration for version {from_version} already registered"
            )
        _migrations[from_version] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Core migration function
# ---------------------------------------------------------------------------

def migrate_config(
    data: dict[str, Any],
    from_version: int | None = None,
    to_version: int | None = None,
) -> dict[str, Any]:
    """Migrate a config dict from one version to another.

    Parameters
    ----------
    data:
        The raw config dict (will not be mutated; a deep copy is made).
    from_version:
        Source version.  If ``None``, reads ``data["__version"]``
        (defaults to 0 if absent).
    to_version:
        Target version.  If ``None``, migrates to ``CURRENT_VERSION``.

    Returns
    -------
    A new dict at the target version with ``__version`` updated.

    Raises
    ------
    ValueError:
        If a required migration function is missing for any intermediate
        version step.
    """
    result = copy.deepcopy(data)

    if from_version is None:
        from_version = result.get("__version", 0)

    if to_version is None:
        to_version = CURRENT_VERSION

    if from_version >= to_version:
        result["__version"] = to_version
        return result

    version = from_version
    while version < to_version:
        if version not in _migrations:
            raise ValueError(
                f"No migration registered for version {version} -> {version + 1}. "
                f"Available migrations: {sorted(_migrations.keys())}"
            )
        logger.debug("Migrating config from version %d to %d", version, version + 1)
        result = _migrations[version](result)
        version += 1

    result["__version"] = to_version
    return result


def get_config_version(data: dict[str, Any]) -> int:
    """Read the ``__version`` key from a config dict, defaulting to 0."""
    return data.get("__version", 0)


def needs_migration(data: dict[str, Any]) -> bool:
    """Check whether a config dict needs migration to the current version."""
    return get_config_version(data) < CURRENT_VERSION


# ---------------------------------------------------------------------------
# Built-in migrations
# ---------------------------------------------------------------------------

@register_migration(0)
def _migrate_0_to_1(data: dict[str, Any]) -> dict[str, Any]:
    """Version 0 -> 1: Normalize field names from eritrainer era.

    - Rename 'eritrainer_*' prefixed keys to 'serenity_*' if present.
    - Ensure 'training_method' exists (default to 'lora').
    - Add 'fallback_train_dtype' if missing.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key.startswith("eritrainer_"):
            new_key = key.replace("eritrainer_", "serenity_", 1)
            result[new_key] = value
        else:
            result[key] = value

    if "training_method" not in result:
        result["training_method"] = "lora"

    if "fallback_train_dtype" not in result:
        train_dtype = result.get("train_dtype", "FLOAT_16")
        if train_dtype == "FLOAT_16":
            result["fallback_train_dtype"] = "BFLOAT_16"
        else:
            result["fallback_train_dtype"] = train_dtype

    return result


__all__ = [
    "CURRENT_VERSION",
    "migrate_config",
    "register_migration",
    "get_config_version",
    "needs_migration",
]
