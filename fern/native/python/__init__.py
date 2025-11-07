"""
Native optimizations for i-LAVA.

This module provides CUDA-accelerated implementations of performance-critical operations.
"""

import logging

logger = logging.getLogger(__name__)

# Try to import CUDA extension
_cuda_available = False
_rvq_cuda_ext = None

try:
    from . import rvq_cuda_ext as _rvq_cuda_ext
    _cuda_available = True
    logger.info("✓ RVQ CUDA extension loaded")
except ImportError as e:
    logger.info(f"RVQ CUDA extension not available: {e}")
    logger.info("Falling back to PyTorch implementation")


def rvq_cuda_available() -> bool:
    """Check if RVQ CUDA extension is available."""
    return _cuda_available


def decode_rvq_cuda(codes, embeddings):
    """
    Decode RVQ codes to embeddings using CUDA.
    
    Args:
        codes: Tensor [batch, num_codebooks, time_steps] int32
        embeddings: Tensor [num_codebooks, vocab_size, embedding_dim] float32
        
    Returns:
        output: Tensor [batch, time_steps, embedding_dim] float32
        
    Raises:
        RuntimeError: If CUDA extension not available
    """
    if not _cuda_available:
        raise RuntimeError("RVQ CUDA extension not available. Please build with CUDA support.")
    
    return _rvq_cuda_ext.decode_rvq(codes, embeddings)


def decode_rvq_cuda_amortized(codes, embeddings, frame_mask):
    """
    Decode RVQ codes with compute amortization (1/16 sampling).
    
    Args:
        codes: Tensor [batch, num_codebooks, time_steps] int32
        embeddings: Tensor [num_codebooks, vocab_size, embedding_dim] float32
        frame_mask: Tensor [time_steps] bool - which frames to process
        
    Returns:
        output: Tensor [batch, time_steps, embedding_dim] float32
        
    Raises:
        RuntimeError: If CUDA extension not available
    """
    if not _cuda_available:
        raise RuntimeError("RVQ CUDA extension not available.")
    
    return _rvq_cuda_ext.decode_rvq_amortized(codes, embeddings, frame_mask)


__all__ = [
    'rvq_cuda_available',
    'decode_rvq_cuda',
    'decode_rvq_cuda_amortized',
]

