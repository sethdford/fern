"""Utility functions for i-LAVA pipeline."""

from fern.utils.logging import setup_logging, get_logger
from fern.utils.device import detect_device, optimize_for_device

__all__ = ["setup_logging", "get_logger", "detect_device", "optimize_for_device"]

