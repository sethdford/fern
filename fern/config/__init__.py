"""
FERN configuration module.

Automatically applies native optimizations when imported.
"""

from .native_optimizations import (
    apply_optimizations,
    is_optimized,
    APPLIED_OPTIMIZATIONS,
)

__all__ = [
    "apply_optimizations",
    "is_optimized",
    "APPLIED_OPTIMIZATIONS",
]

