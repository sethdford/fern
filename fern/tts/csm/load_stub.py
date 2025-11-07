"""
Stub loader for CSM-1B when actual model is not available.
This provides a minimal interface for testing without downloading the full model.
"""

import torch
import logging
from models import Model, ModelArgs
from generator import Generator

logger = logging.getLogger(__name__)


def load_csm_1b_stub(device: str = "cpu") -> Generator:
    """
    Load a stub CSM-1B model for testing.
    
    This creates a minimal model structure without actual weights.
    Use this for development/testing when you don't want to download
    the full model from HuggingFace.
    
    Args:
        device: Device to place model on ('cpu', 'cuda', 'mps')
        
    Returns:
        Generator instance with stub model
    """
    logger.warning("⚠️  Loading CSM-1B STUB model (no real weights)")
    logger.warning("   This is for testing only - audio will be synthetic")
    logger.warning("   To use real CSM-1B, the model will be downloaded from HuggingFace")
    
    # Determine dtype based on device FIRST
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32  # CPU doesn't handle bfloat16 well
    
    logger.info(f"Using dtype: {dtype} on device: {device}")
    
    # Create model config
    config = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    
    # Create model (with random weights) directly on device with correct dtype
    logger.info("Creating model...")
    
    # Temporarily set default dtype to avoid bfloat16 issues
    with torch.device(device):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = Model(config=config)
            model.to(device=device, dtype=dtype)
            
            # Setup caches with correct dtype
            model.setup_caches(max_batch_size=1)
        finally:
            torch.set_default_dtype(old_dtype)
    
    logger.info(f"✓ Stub model created on {device}")
    
    # Create generator
    generator = Generator(model)
    
    return generator

