"""Device detection and optimization utilities."""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def detect_device() -> Tuple[str, str]:
    """
    Detect the best available compute device.
    
    Returns:
        Tuple of (device_type, device_name)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA device detected: {device_name}")
        return "cuda", device_name
    elif torch.backends.mps.is_available():
        logger.info("Apple Metal Performance Shaders (MPS) detected")
        return "mps", "Apple MPS"
    else:
        import platform
        cpu_name = platform.processor() or platform.machine()
        logger.info(f"Using CPU: {cpu_name}")
        return "cpu", cpu_name


def optimize_for_device(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """
    Optimize model for specific device.
    
    Args:
        model: PyTorch model to optimize
        device: Target device (cuda, mps, cpu)
        
    Returns:
        Optimized model
    """
    model = model.to(device)
    
    if device == "cuda":
        # Enable TF32 for CUDA
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 for CUDA operations")
        
        # Set memory efficient settings
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark mode")
    
    elif device == "mps":
        # MPS-specific optimizations
        logger.info("Applied MPS optimizations")
    
    elif device == "cpu":
        # CPU-specific optimizations
        torch.set_num_threads(torch.get_num_threads())
        logger.info(f"Using {torch.get_num_threads()} CPU threads")
    
    return model


def get_device_config(device: str) -> dict:
    """
    Get device-specific configuration parameters.
    
    Args:
        device: Device type (cuda, mps, cpu)
        
    Returns:
        Dictionary of device-specific parameters
    """
    config = {
        "device": device,
        "dtype": torch.float32,
        "compile_enabled": False,
    }
    
    if device == "cuda":
        # Use float16 for CUDA
        config["dtype"] = torch.float16
        config["compile_enabled"] = True
        
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        config["total_memory_gb"] = total_memory
        logger.info(f"CUDA device has {total_memory:.1f} GB memory")
    
    elif device == "mps":
        # Use float32 for MPS (float16 can be unstable)
        config["dtype"] = torch.float32
        config["compile_enabled"] = False
    
    elif device == "cpu":
        config["dtype"] = torch.float32
        config["compile_enabled"] = False
    
    return config

