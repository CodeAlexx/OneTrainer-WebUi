"""Dark theme configuration for the Serenity UI."""

from __future__ import annotations

import dearpygui.dearpygui as dpg

__all__ = ["apply_dark_theme"]


def apply_dark_theme() -> None:
    """Apply a dark theme inspired by OneTrainer's color scheme."""
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Window / frame backgrounds
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 30))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (35, 35, 38))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (40, 40, 43))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (50, 50, 55))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (65, 65, 70))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (75, 75, 80))

            # Titles / headers
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (25, 25, 28))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (40, 80, 140))
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (35, 35, 38))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 90, 150))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (60, 100, 170))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (70, 110, 180))

            # Tabs
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (45, 45, 50))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (60, 100, 170))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (50, 90, 155))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, (35, 35, 38))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, (40, 70, 120))

            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, (55, 55, 62))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 70, 78))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (80, 80, 90))

            # Check / slider
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (100, 170, 240))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (80, 140, 210))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (100, 160, 230))

            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (30, 30, 33))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (60, 60, 65))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (80, 80, 85))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (100, 100, 105))

            # Separator / border
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (50, 50, 55))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (55, 55, 60))

            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 225))
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (120, 120, 130))

            # Table
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, (45, 45, 50))
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, (55, 55, 60))
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, (45, 45, 50))
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, (35, 35, 38))
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, (40, 40, 44))

            # Rounding
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 3)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)

            # Spacing
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 5)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)

    dpg.bind_theme(global_theme)


def create_start_button_theme() -> int:
    """Green start-training button theme."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (25, 135, 84))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (20, 108, 67))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (15, 90, 55))
    return theme


def create_stop_button_theme() -> int:
    """Red stop-training button theme."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (220, 53, 69))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (187, 45, 59))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (160, 38, 50))
    return theme
