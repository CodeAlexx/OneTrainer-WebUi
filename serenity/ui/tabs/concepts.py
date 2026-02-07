"""Concepts management tab for the Serenity training UI.

Manages the list of training concepts (dataset subsets). Each concept
defines an image directory, name, type, and whether it is enabled.
The list is dynamic: users can add and remove concepts at will.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

from serenity.core.concept_config import ConceptConfig
from serenity.ui.state import UIState
from serenity.ui.widgets import (
    labeled_checkbox,
    labeled_combo,
    labeled_dir,
    labeled_float,
    labeled_input,
    labeled_int,
    labeled_separator,
    tooltip,
)

__all__ = ["build_concepts_tab"]

# Stable tag for the scrollable concept list container.
CONCEPT_LIST_TAG = "concept_list"

# Supported concept types (mirrors OneTrainer).
CONCEPT_TYPES = ["STANDARD", "VALIDATION", "PRIOR_PRESERVATION"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_concepts_tab(ui_state: UIState) -> None:
    """Build the Concepts tab with an add button and scrollable concept list."""
    with dpg.group(horizontal=True):
        btn = dpg.add_button(
            label="Add Concept",
            callback=lambda: _add_concept(ui_state),
        )
        tooltip(btn, "Append a new training concept to the list")
        dpg.add_text(
            tag="concept_count",
            default_value=f"Concepts: {len(ui_state.config.concepts)}",
        )

    dpg.add_spacer(height=4)

    with dpg.child_window(tag=CONCEPT_LIST_TAG, autosize_x=True, height=-1):
        _rebuild_concept_list(ui_state)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_count(ui_state: UIState) -> None:
    """Refresh the concept count label."""
    try:
        dpg.set_value("concept_count", f"Concepts: {len(ui_state.config.concepts)}")
    except Exception:
        pass


def _add_concept(ui_state: UIState) -> None:
    """Append a blank concept and rebuild the list."""
    concept = ConceptConfig(name=f"concept_{len(ui_state.config.concepts) + 1}")
    ui_state.config.concepts.append(concept)
    _rebuild_concept_list(ui_state)
    _update_count(ui_state)


def _remove_concept(ui_state: UIState, index: int) -> None:
    """Remove a concept by index and rebuild the list."""
    try:
        ui_state.config.concepts.pop(index)
    except IndexError:
        return
    _rebuild_concept_list(ui_state)
    _update_count(ui_state)


def _rebuild_concept_list(ui_state: UIState) -> None:
    """Clear and redraw every concept item inside the scroll region."""
    dpg.delete_item(CONCEPT_LIST_TAG, children_only=True)
    for i, concept in enumerate(ui_state.config.concepts):
        _build_concept_item(ui_state, i, concept)


def _build_concept_item(ui_state: UIState, index: int, concept: ConceptConfig) -> None:
    """Render a single collapsible concept entry."""
    header_label = concept.name or f"Concept {index}"
    with dpg.collapsing_header(
        label=header_label,
        parent=CONCEPT_LIST_TAG,
        default_open=True,
    ):
        # -- Name --
        labeled_input(
            "Name",
            default_value=concept.name,
            callback=lambda _s, val: _set_concept_field(
                ui_state, index, "name", val,
            ),
            tip="Display name for this concept",
        )

        # -- Image directory --
        labeled_dir(
            "Image Path",
            default_value=concept.path,
            callback=lambda _s, val: _set_concept_field(
                ui_state, index, "path", val,
            ),
            tip="Directory containing training images for this concept",
        )

        # -- Concept type --
        labeled_combo(
            "Type",
            CONCEPT_TYPES,
            default_value=CONCEPT_TYPES[0],
            callback=lambda _s, val: _set_concept_field(
                ui_state, index, "type", val,
            ),
            tip="Concept type: standard training, validation, or prior preservation",
        )

        # -- Enabled --
        labeled_checkbox(
            "Enabled",
            default_value=concept.enabled,
            callback=lambda _s, val: _set_concept_field(
                ui_state, index, "enabled", val,
            ),
            tip="Include this concept during training",
        )

        # -- Balancing / loss weight --
        labeled_float(
            "Balancing",
            default_value=concept.balancing,
            callback=lambda _s, val: _set_concept_field(
                ui_state, index, "balancing", val,
            ),
            min_value=0.0,
            max_value=100.0,
            format_str="%.2f",
            tip="Relative weight for balancing this concept against others",
        )
        labeled_float(
            "Loss Weight",
            default_value=concept.loss_weight,
            callback=lambda _s, val: _set_concept_field(
                ui_state, index, "loss_weight", val,
            ),
            min_value=0.0,
            max_value=100.0,
            format_str="%.2f",
            tip="Loss multiplier for this concept",
        )

        # -- Include subdirs --
        labeled_checkbox(
            "Include Subdirectories",
            default_value=concept.include_subdirectories,
            callback=lambda _s, val: _set_concept_field(
                ui_state, index, "include_subdirectories", val,
            ),
            tip="Recursively include images from subdirectories",
        )

        # -- Image / text variations --
        labeled_int(
            "Image Variations",
            default_value=concept.image_variations,
            callback=lambda _s, val: _set_concept_field(
                ui_state, index, "image_variations", int(val),
            ),
            min_value=1,
            max_value=100,
            tip="Number of augmented image variations per source image",
        )
        labeled_int(
            "Text Variations",
            default_value=concept.text_variations,
            callback=lambda _s, val: _set_concept_field(
                ui_state, index, "text_variations", int(val),
            ),
            min_value=1,
            max_value=100,
            tip="Number of caption variations per image",
        )

        dpg.add_spacer(height=4)

        # -- Remove button --
        # Capture index in default arg to avoid late-binding closure issues.
        dpg.add_button(
            label="Remove",
            callback=lambda _s, _a, idx=index: _remove_concept(ui_state, idx),
        )


def _set_concept_field(
    ui_state: UIState, index: int, field: str, value: object,
) -> None:
    """Write *value* into ``ui_state.config.concepts[index].<field>``."""
    try:
        concept = ui_state.config.concepts[index]
    except IndexError:
        return
    setattr(concept, field, value)
    # If the name changed, we cannot cheaply rename the collapsing header
    # without a full rebuild; the rebuild is fast enough for typical counts.
    if field == "name":
        _rebuild_concept_list(ui_state)
