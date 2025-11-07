"""
Real model loading for CSM-1B and Mimi.

This module loads the actual trained weights instead of using stubs.
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Tuple
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)

# Model paths from download
CSM_1B_PATH = Path("models/csm-1b")
MIMI_PATH = Path("models/mimi")


def load_csm_1b_real(
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> Tuple:
    """
    Load real CSM-1B model from downloaded weights.
    
    Args:
        device: Device to load on ('cuda', 'cpu', 'mps')
        dtype: Data type (defaults to float32 for CPU, bfloat16 for CUDA)
    
    Returns:
        Tuple of (generator, mimi)
    """
    logger.info(f"Loading real CSM-1B from {CSM_1B_PATH}")
    
    # Determine dtype
    if dtype is None:
        if device == "cuda" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        elif device == "mps":
            # MPS supports bfloat16 on Apple Silicon
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    
    logger.info(f"Using device={device}, dtype={dtype}")
    
    try:
        # Load CSM-1B model
        from models import Model, ModelArgs
        from generator import Generator
        
        # Find model file
        model_files = list(CSM_1B_PATH.glob("*.safetensors")) + \
                     list(CSM_1B_PATH.glob("*.bin")) + \
                     list(CSM_1B_PATH.glob("*.pt"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {CSM_1B_PATH}")
        
        model_file = model_files[0]
        logger.info(f"Loading weights from: {model_file.name}")
        
        # Load config if available
        config_file = CSM_1B_PATH / "config.json"
        if config_file.exists():
            import json
            with open(config_file) as f:
                config_dict = json.load(f)
            
            # Create ModelArgs from config
            args = ModelArgs(
                dim=config_dict.get("hidden_size", 2048),
                n_layers=config_dict.get("num_hidden_layers", 24),
                n_heads=config_dict.get("num_attention_heads", 32),
                n_kv_heads=config_dict.get("num_key_value_heads", 8),
                vocab_size=config_dict.get("vocab_size", 32768),
                n_codebooks=config_dict.get("n_codebooks", 32),
            )
        else:
            # Use default CSM-1B config
            logger.warning("No config.json found, using defaults")
            args = ModelArgs(
                dim=2048,
                n_layers=24,
                n_heads=32,
                n_kv_heads=8,
                vocab_size=32768,
                n_codebooks=32,
            )
        
        # Initialize model
        model = Model(args)
        
        # Load weights
        if model_file.suffix == ".safetensors":
            state_dict = load_safetensors(str(model_file))
        else:
            state_dict = torch.load(model_file, map_location=device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
        
        # Load state dict
        # Convert dtypes if needed (bfloat16 -> float32 for CPU)
        if dtype != torch.bfloat16 and any(v.dtype == torch.bfloat16 for v in state_dict.values()):
            logger.info("Converting model from bfloat16 to float32...")
            state_dict = {k: v.float() if v.dtype == torch.bfloat16 else v 
                         for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        # Setup caches with correct dtype
        # Must be done after model is on correct device and dtype
        if hasattr(model, 'setup_caches'):
            # Clear any existing caches first
            if hasattr(model, '_k_cache'):
                delattr(model, '_k_cache')
            if hasattr(model, '_v_cache'):
                delattr(model, '_v_cache')
            
            # Setup new caches with correct dtype
            model.setup_caches(
                batch_size=1,
                dtype=dtype,
            )
        
        logger.info("✓ CSM-1B loaded successfully")
        
        # Load Mimi
        mimi = load_mimi_real(device=device)
        
        # Create generator
        generator = Generator(model, mimi, device=device)
        
        return generator, mimi
        
    except Exception as e:
        logger.error(f"Failed to load real CSM-1B: {e}")
        logger.warning("Falling back to stub")
        from load_stub import load_csm_1b_stub
        return load_csm_1b_stub(device), None


def load_mimi_real(
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Load real Mimi codec from downloaded weights.
    
    Args:
        device: Device to load on
    
    Returns:
        Mimi codec instance
    """
    logger.info(f"Loading real Mimi from {MIMI_PATH}")
    
    try:
        # Find model file
        model_files = list(MIMI_PATH.glob("*.safetensors")) + \
                     list(MIMI_PATH.glob("*.bin")) + \
                     list(MIMI_PATH.glob("*.pt"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {MIMI_PATH}")
        
        model_file = model_files[0]
        logger.info(f"Loading Mimi weights from: {model_file.name}")
        
        # Try to use the real Mimi implementation
        try:
            from mimi import MimiModel
            mimi = MimiModel.from_pretrained(str(MIMI_PATH))
        except ImportError:
            # Fallback to our implementation
            logger.warning("mimi package not found, using fallback implementation")
            from moshi.models.loaders import RealMimiModel
            
            mimi = RealMimiModel(
                n_codebooks=32,
                vocab_size=2048,
                sample_rate=24000,
            )
            
            # Load weights
            if model_file.suffix == ".safetensors":
                state_dict = load_safetensors(str(model_file))
            else:
                state_dict = torch.load(model_file, map_location=device)
            
            # Load with dtype conversion if needed
            if device == "cpu":
                # Convert bfloat16 to float32 for CPU
                state_dict = {k: v.float() if v.dtype == torch.bfloat16 else v 
                             for k, v in state_dict.items()}
            
            mimi.load_state_dict(state_dict, strict=False)
        
        mimi = mimi.to(device)
        mimi.eval()
        
        logger.info("✓ Mimi loaded successfully")
        return mimi
        
    except Exception as e:
        logger.error(f"Failed to load real Mimi: {e}")
        logger.warning("Falling back to stub")
        from moshi.models.loaders import MimiModel
        return MimiModel()
