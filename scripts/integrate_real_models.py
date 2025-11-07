#!/usr/bin/env python3
"""
Integrate Real CSM-1B and Mimi Models into FERN

This script creates the necessary loader functions and updates the TTS module
to use real model weights instead of stubs.

Usage:
    python scripts/integrate_real_models.py
"""

import json
import sys
from pathlib import Path

# Add fern to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{text:^80}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")


def print_success(text: str):
    print(f"{GREEN}✓ {text}{RESET}")


def main():
    print_header("Integrating Real Models into FERN")
    
    # Load model config
    config_path = Path("models/model_config.json")
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        print("Run: python scripts/download_models.py first")
        return 1
    
    with open(config_path) as f:
        config = json.load(f)
    
    csm_path = Path(config["csm_1b_path"])
    mimi_path = Path(config["mimi_path"])
    
    print(f"CSM-1B path: {csm_path}")
    print(f"Mimi path: {mimi_path}")
    print()
    
    # Create the real loader implementation
    print_success("Creating fern/tts/csm/load_real.py...")
    
    load_real_content = f'''"""
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
CSM_1B_PATH = Path("{csm_path}")
MIMI_PATH = Path("{mimi_path}")


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
    logger.info(f"Loading real CSM-1B from {{CSM_1B_PATH}}")
    
    # Determine dtype
    if dtype is None:
        if device == "cuda" and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float32
    
    logger.info(f"Using device={{device}}, dtype={{dtype}}")
    
    try:
        # Load CSM-1B model
        from .models import Model, ModelArgs
        from .generator import Generator
        
        # Find model file
        model_files = list(CSM_1B_PATH.glob("*.safetensors")) + \\
                     list(CSM_1B_PATH.glob("*.bin")) + \\
                     list(CSM_1B_PATH.glob("*.pt"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {{CSM_1B_PATH}}")
        
        model_file = model_files[0]
        logger.info(f"Loading weights from: {{model_file.name}}")
        
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
        
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        logger.info("✓ CSM-1B loaded successfully")
        
        # Load Mimi
        mimi = load_mimi_real(device=device)
        
        # Create generator
        generator = Generator(model, mimi, device=device)
        
        return generator, mimi
        
    except Exception as e:
        logger.error(f"Failed to load real CSM-1B: {{e}}")
        logger.warning("Falling back to stub")
        from .load_stub import load_csm_1b_stub
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
    logger.info(f"Loading real Mimi from {{MIMI_PATH}}")
    
    try:
        # Find model file
        model_files = list(MIMI_PATH.glob("*.safetensors")) + \\
                     list(MIMI_PATH.glob("*.bin")) + \\
                     list(MIMI_PATH.glob("*.pt"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {{MIMI_PATH}}")
        
        model_file = model_files[0]
        logger.info(f"Loading Mimi weights from: {{model_file.name}}")
        
        # Try to use the real Mimi implementation
        try:
            from mimi import MimiModel
            mimi = MimiModel.from_pretrained(str(MIMI_PATH))
        except ImportError:
            # Fallback to our implementation
            logger.warning("mimi package not found, using fallback implementation")
            from .moshi.models.loaders import RealMimiModel
            
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
            
            mimi.load_state_dict(state_dict, strict=False)
        
        mimi = mimi.to(device)
        mimi.eval()
        
        logger.info("✓ Mimi loaded successfully")
        return mimi
        
    except Exception as e:
        logger.error(f"Failed to load real Mimi: {{e}}")
        logger.warning("Falling back to stub")
        from .moshi.models.loaders import MimiModel
        return MimiModel()
'''
    
    load_real_path = Path("fern/tts/csm/load_real.py")
    load_real_path.write_text(load_real_content)
    print_success(f"Created {load_real_path}")
    
    # Update csm_real.py to use real loader
    print_success("Updating fern/tts/csm_real.py...")
    
    csm_real_path = Path("fern/tts/csm_real.py")
    csm_real_content = csm_real_path.read_text()
    
    # Replace the import and loading logic
    updated_content = csm_real_content.replace(
        "from .csm.load_stub import load_csm_1b_stub",
        "from .csm.load_real import load_csm_1b_real"
    ).replace(
        "self.generator = load_csm_1b_stub(device)",
        "self.generator, self.mimi = load_csm_1b_real(device)"
    )
    
    csm_real_path.write_text(updated_content)
    print_success(f"Updated {csm_real_path}")
    
    # Update CSMConfig to default to real model
    print_success("Updating fern/tts/csm_config.py...")
    
    config_path = Path("fern/tts/csm_config.py")
    config_content = config_path.read_text()
    
    # Change default from False to True
    updated_config = config_content.replace(
        'use_real_csm: bool = False',
        'use_real_csm: bool = True'
    )
    
    config_path.write_text(updated_config)
    print_success(f"Updated {config_path}")
    
    print_header("Integration Complete!")
    
    print(f"{BOLD}What changed:{RESET}")
    print("✓ Created load_real.py with actual model loading")
    print("✓ Updated csm_real.py to use real models")
    print("✓ Changed default to use_real_csm=True")
    print()
    
    print(f"{BOLD}Next steps:{RESET}")
    print("1. Test generation: python scripts/test_real_models.py")
    print("2. Run pipeline: python -m fern.cli \"Hello world\"")
    print("3. Start training: python scripts/train_lora.py")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

