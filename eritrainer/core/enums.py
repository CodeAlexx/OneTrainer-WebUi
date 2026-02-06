"""Shared enum definitions."""

from enum import Enum


class GradientCheckpointingMethod(str, Enum):
    OFF = "off"
    ON = "on"
    CPU_OFFLOADED = "cpu_offloaded"
