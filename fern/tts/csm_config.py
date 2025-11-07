"""
Configuration management for CSM-1B TTS.

This module provides configuration classes following Clean Architecture
and reducing parameter count from 10 to 1 (Config Object pattern).

Author: i-LAVA Project
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Literal
import logging
import torch

logger = logging.getLogger(__name__)


@dataclass
class CSMConfig:
    """
    Configuration for CSM-1B Text-to-Speech model.
    
    This config object replaces 10 individual parameters with a single
    config object, following Clean Code principles.
    
    Attributes:
        model_name: Name of the CSM model to use
        device: Device for inference ('cuda', 'mps', 'cpu')
        rvq_iterations: Number of RVQ iterations (16, 20, 24, or 32)
        rvq_padding_method: Padding method for RVQ ('none', 'zero', 'repeat')
        mimi_codebooks: Number of Mimi codebooks (should match rvq_iterations)
        sample_rate: Audio sample rate in Hz
        enable_torch_compile: Whether to use torch.compile for optimization
        enable_cold_start: Whether to perform cold start warmup
        cold_start_generations: Number of cold start generations
        use_real_csm: Whether to use real CSM-1B (True) or placeholder (False)
    
    Example:
        >>> config = CSMConfig(device='cuda', use_real_csm=True)
        >>> tts = CSMTTS(config=config)
        >>> audio = tts.synthesize("Hello world")
    
    Note:
        - If device is not available, will fallback gracefully
        - mimi_codebooks should match rvq_iterations for best performance
    """
    
    model_name: str = "csm-1b"
    device: str = "cuda"
    rvq_iterations: int = 16
    rvq_padding_method: Literal["none", "zero", "repeat"] = "none"
    mimi_codebooks: int = 32
    sample_rate: int = 24000
    enable_torch_compile: bool = True
    enable_cold_start: bool = True
    cold_start_generations: int = 2
    use_real_csm: bool = True
    
    def __post_init__(self):
        """
        Validate and adjust configuration after initialization.
        
        This method:
        1. Validates device availability and falls back if needed
        2. Auto-adjusts mimi_codebooks to match rvq_iterations
        3. Validates rvq_iterations is in valid range
        
        Raises:
            ValueError: If rvq_iterations is invalid
        """
        # Validate rvq_iterations
        valid_iterations = [16, 20, 24, 32]
        if self.rvq_iterations not in valid_iterations:
            raise ValueError(
                f"rvq_iterations must be one of {valid_iterations}, "
                f"got {self.rvq_iterations}"
            )
        
        # Auto-adjust mimi_codebooks if using no padding
        if self.rvq_padding_method == "none":
            if self.mimi_codebooks != self.rvq_iterations:
                logger.debug(
                    f"Auto-adjusting mimi_codebooks from {self.mimi_codebooks} "
                    f"to {self.rvq_iterations} (no padding mode)"
                )
                self.mimi_codebooks = self.rvq_iterations
        
        # Validate and adjust device
        self.device = self._validate_device(self.device)
    
    def _validate_device(self, device: str) -> str:
        """
        Validate device availability and fallback if needed.
        
        Args:
            device: Requested device ('cuda', 'mps', 'cpu')
        
        Returns:
            Valid device string (may differ from input if fallback occurred)
        
        Note:
            Logs warning if fallback occurs.
        """
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but not available, falling back to CPU"
                )
                return "cpu"
        elif device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning(
                    "MPS requested but not available, falling back to CPU"
                )
                return "cpu"
        elif device != "cpu":
            logger.warning(
                f"Unknown device '{device}', falling back to CPU"
            )
            return "cpu"
        
        return device
    
    def to_dict(self) -> dict:
        """
        Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        
        Example:
            >>> config = CSMConfig(device='cpu')
            >>> config_dict = config.to_dict()
            >>> assert config_dict['device'] == 'cpu'
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'rvq_iterations': self.rvq_iterations,
            'rvq_padding_method': self.rvq_padding_method,
            'mimi_codebooks': self.mimi_codebooks,
            'sample_rate': self.sample_rate,
            'enable_torch_compile': self.enable_torch_compile,
            'enable_cold_start': self.enable_cold_start,
            'cold_start_generations': self.cold_start_generations,
            'use_real_csm': self.use_real_csm,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'CSMConfig':
        """
        Create config from dictionary.
        
        Args:
            config_dict: Dictionary with config values
        
        Returns:
            CSMConfig instance
        
        Example:
            >>> config_dict = {'device': 'cpu', 'use_real_csm': False}
            >>> config = CSMConfig.from_dict(config_dict)
            >>> assert config.device == 'cpu'
        """
        return cls(**config_dict)


# Convenience factory functions following Clean Architecture

def create_production_config(device: str = "cuda") -> CSMConfig:
    """
    Create production-ready CSM config.
    
    Args:
        device: Device to use ('cuda', 'mps', 'cpu')
    
    Returns:
        CSMConfig optimized for production
    
    Example:
        >>> config = create_production_config('cuda')
        >>> assert config.use_real_csm is True
        >>> assert config.enable_torch_compile is True
    """
    return CSMConfig(
        device=device,
        use_real_csm=True,
        enable_torch_compile=True,
        enable_cold_start=True,
        cold_start_generations=2,
    )


def create_development_config(device: str = "cpu") -> CSMConfig:
    """
    Create development/testing CSM config.
    
    Args:
        device: Device to use (defaults to 'cpu')
    
    Returns:
        CSMConfig optimized for development
    
    Example:
        >>> config = create_development_config()
        >>> assert config.use_real_csm is False
        >>> assert config.device == 'cpu'
    """
    return CSMConfig(
        device=device,
        use_real_csm=False,
        enable_torch_compile=False,
        enable_cold_start=False,
    )


def create_fast_config(device: str = "cuda") -> CSMConfig:
    """
    Create config optimized for speed (reduced quality).
    
    Args:
        device: Device to use
    
    Returns:
        CSMConfig optimized for low latency
    
    Example:
        >>> config = create_fast_config('cuda')
        >>> assert config.rvq_iterations == 16
    """
    return CSMConfig(
        device=device,
        use_real_csm=True,
        rvq_iterations=16,  # Faster
        enable_torch_compile=True,
        enable_cold_start=True,
    )


def create_quality_config(device: str = "cuda") -> CSMConfig:
    """
    Create config optimized for quality (slower).
    
    Args:
        device: Device to use
    
    Returns:
        CSMConfig optimized for high quality
    
    Example:
        >>> config = create_quality_config('cuda')
        >>> assert config.rvq_iterations == 32
    """
    return CSMConfig(
        device=device,
        use_real_csm=True,
        rvq_iterations=32,  # Best quality
        enable_torch_compile=True,
        enable_cold_start=True,
    )

