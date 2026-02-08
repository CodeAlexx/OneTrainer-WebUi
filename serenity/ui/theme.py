"""Dark theme configuration for the Serenity UI with auto display scaling."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import dearpygui.dearpygui as dpg

__all__ = [
    "apply_dark_theme",
    "setup_fonts",
    "create_start_button_theme",
    "create_stop_button_theme",
    "UI_SCALE",
    "scaled",
]

log = logging.getLogger(__name__)

# -- Accent palette ---------------------------------------------------------
# Soft blue accent with warm neutrals for a modern, readable dark UI.
_ACCENT = (86, 156, 240)       # Primary accent (soft blue)
_ACCENT_HI = (110, 175, 255)   # Hover / active accent
_ACCENT_DIM = (60, 120, 200)   # Muted accent for borders, inactive tabs

_BG_DARK = (22, 22, 26)        # Window background
_BG_MID = (28, 28, 33)         # Child / panel background
_BG_POPUP = (34, 34, 40)       # Popup / tooltip background
_FRAME = (42, 42, 50)          # Input frame background
_FRAME_HI = (55, 55, 65)       # Input hover
_FRAME_ACT = (65, 65, 78)      # Input active / focused

_SURFACE = (38, 38, 44)        # Card / elevated surface
_BORDER = (52, 52, 62)         # Subtle border
_SEP = (48, 48, 56)            # Separator

_TEXT = (228, 228, 235)         # Primary text
_TEXT_DIM = (130, 130, 145)     # Disabled / secondary text

_TAB = (36, 36, 42)            # Inactive tab
_TAB_HI = (55, 100, 165)       # Hovered tab
_TAB_ACT = (50, 90, 155)       # Active / selected tab
_TAB_UNFOCUSED = (30, 30, 36)

_BTN = (50, 50, 60)            # Button rest
_BTN_HI = (62, 62, 74)         # Button hover
_BTN_ACT = (72, 72, 86)        # Button pressed

_GREEN = (30, 160, 95)
_GREEN_HI = (25, 135, 80)
_GREEN_ACT = (20, 110, 65)

_RED = (210, 55, 65)
_RED_HI = (180, 45, 55)
_RED_ACT = (155, 38, 48)


# -- Display scale detection ------------------------------------------------

def _detect_screen_height() -> int:
    """Detect primary display height in pixels."""
    # Try tkinter (most reliable, cross-platform)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        h = root.winfo_screenheight()
        root.destroy()
        return h
    except Exception:
        pass

    # Fallback: xrandr on Linux
    try:
        result = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True, text=True, timeout=2,
        )
        for line in result.stdout.splitlines():
            if "*" in line:
                # e.g. "   3840x2160     60.00*+"
                res = line.split()[0]
                return int(res.split("x")[1])
    except Exception:
        pass

    return 1080


def _compute_ui_scale() -> float:
    """Compute UI scale factor based on screen resolution.

    Returns 1.0 for 1080p, scales up for higher resolutions.
    """
    h = _detect_screen_height()
    if h <= 1200:
        return 1.0
    if h <= 1600:
        return 1.15
    return 1.4


UI_SCALE: float = _compute_ui_scale()


def scaled(value: int) -> int:
    """Scale a base pixel value (designed for 1080p) to current display."""
    return round(value * UI_SCALE)


# -- Font setup -------------------------------------------------------------

_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
]

_FONT_SEARCH_PATHS_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
]


def _find_font(paths: list[str]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None


def setup_fonts(size: int = 0) -> None:
    """Load a readable font scaled to the display. Falls back to global font scale."""
    if size == 0:
        size = scaled(16)

    font_path = _find_font(_FONT_SEARCH_PATHS)
    if font_path is None:
        log.warning("No TTF font found, using global font scale fallback")
        dpg.set_global_font_scale(1.1 * UI_SCALE)
        return

    with dpg.font_registry():
        default_font = dpg.add_font(font_path, size)

        # Try loading bold variant for headers
        bold_path = _find_font(_FONT_SEARCH_PATHS_BOLD)
        if bold_path:
            dpg.add_font(bold_path, size)

    dpg.bind_font(default_font)
    log.info("Loaded font %s at %dpx (scale=%.2f)", font_path, size, UI_SCALE)


# -- Theme -------------------------------------------------------------------

def apply_dark_theme() -> None:
    """Apply a modern dark theme that scales with display resolution."""
    s = UI_SCALE
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Backgrounds
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, _BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, _BG_MID)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, _BG_POPUP)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, _FRAME)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, _FRAME_HI)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, _FRAME_ACT)

            # Titles / headers -- collapsing headers styled as cards
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, _BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, _ACCENT_DIM)
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, _BG_MID)
            dpg.add_theme_color(dpg.mvThemeCol_Header, (36, 40, 52))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (44, 50, 66))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (50, 58, 76))

            # Tabs
            dpg.add_theme_color(dpg.mvThemeCol_Tab, _TAB)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, _TAB_HI)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, _TAB_ACT)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, _TAB_UNFOCUSED)
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, _ACCENT_DIM)

            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, _BTN)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _BTN_HI)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _BTN_ACT)

            # Interactive
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, _ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, _ACCENT_DIM)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, _ACCENT)

            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (25, 25, 30))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (55, 55, 65))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (75, 75, 88))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (95, 95, 110))

            # Separator / border
            dpg.add_theme_color(dpg.mvThemeCol_Separator, _SEP)
            dpg.add_theme_color(dpg.mvThemeCol_Border, _BORDER)

            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, _TEXT)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, _TEXT_DIM)

            # Table
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, _SURFACE)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, _BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, _SEP)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, _BG_MID)
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, _SURFACE)

            # Rounding
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, round(4 * s))
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, round(6 * s))
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, round(4 * s))
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, round(3 * s))
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, round(4 * s))
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, round(4 * s))
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, round(4 * s))

            # Spacing -- auto-scaled from 1080p base values
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, round(9 * s), round(6 * s))
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, round(7 * s), round(4 * s))
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, round(11 * s), round(9 * s))
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, round(11 * s))
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, round(10 * s))
            dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing, round(17 * s))

    dpg.bind_theme(global_theme)


def create_start_button_theme() -> int:
    """Green start-training button theme."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, _GREEN)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _GREEN_HI)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _GREEN_ACT)
    return theme


def create_stop_button_theme() -> int:
    """Red stop-training button theme."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, _RED)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _RED_HI)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _RED_ACT)
    return theme
