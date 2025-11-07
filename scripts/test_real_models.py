#!/usr/bin/env python3
"""
Test Real CSM-1B and Mimi Models

Quick test to verify that real models load and generate audio correctly.

Usage:
    python scripts/test_real_models.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add fern to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{text:^80}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")


def print_success(text: str):
    print(f"{GREEN}✓ {text}{RESET}")


def print_warning(text: str):
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_error(text: str):
    print(f"{RED}✗ {text}{RESET}")


def print_info(text: str):
    print(f"{BLUE}ℹ {text}{RESET}")


def test_mimi_codec():
    """Test Mimi encoding/decoding."""
    print_header("Testing Mimi Codec")
    
    try:
        # Need to add CSM to path for imports to work
        import sys
        csm_path = Path(__file__).parent.parent / "fern" / "tts" / "csm"
        if str(csm_path) not in sys.path:
            sys.path.insert(0, str(csm_path))
        
        from load_real import load_mimi_real
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print_info(f"Loading Mimi on {device}...")
        
        mimi = load_mimi_real(device=device)
        print_success("Mimi loaded")
        
        # Create test audio (1 second at 24kHz)
        print_info("Creating test audio (1s @ 24kHz)...")
        audio = torch.randn(1, 24000).to(device)
        
        # Encode
        print_info("Encoding audio to codes...")
        with torch.no_grad():
            codes = mimi.encode(audio)
        
        print_success(f"Encoded to shape: {codes.shape}")
        
        # Decode
        print_info("Decoding codes back to audio...")
        with torch.no_grad():
            audio_recon = mimi.decode(codes)
        
        print_success(f"Decoded to shape: {audio_recon.shape}")
        
        # Check reconstruction quality
        if audio_recon.shape == audio.shape:
            mse = torch.mean((audio - audio_recon) ** 2).item()
            print_info(f"Reconstruction MSE: {mse:.6f}")
            
            if mse < 1.0:
                print_success("Good reconstruction quality!")
            else:
                print_warning("High reconstruction error (might be expected)")
        
        return True
        
    except Exception as e:
        print_error(f"Mimi test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csm_generation():
    """Test CSM-1B text-to-speech generation."""
    print_header("Testing CSM-1B Generation")
    
    try:
        from fern.tts.csm_real import RealCSMTTS
        
        # Use MPS if available (supports bfloat16), otherwise CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print_info(f"Loading CSM-1B on {device}...")
        
        tts = RealCSMTTS(device=device)
        print_success("CSM-1B loaded")
        
        # Test generation
        test_text = "Hello world, this is a test of the CSM text to speech system."
        print_info(f'Generating: "{test_text}"')
        
        audio = tts.synthesize(test_text)
        
        print_success(f"Generated audio: {audio.shape}")
        print_info(f"Duration: {len(audio) / 24000:.2f}s")
        print_info(f"Sample rate: 24000 Hz")
        print_info(f"Data type: {audio.dtype}")
        
        # Check audio properties
        if len(audio) > 0:
            print_success("Audio generated successfully!")
            
            # Check for silence (bad sign)
            audio_abs_max = np.abs(audio).max()
            if audio_abs_max < 0.001:
                print_warning("Audio seems very quiet (might be silent)")
            else:
                print_success(f"Audio has good amplitude (max: {audio_abs_max:.3f})")
            
            return True
        else:
            print_error("Audio is empty!")
            return False
        
    except Exception as e:
        print_error(f"CSM generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full FERN pipeline."""
    print_header("Testing Full FERN Pipeline")
    
    try:
        from fern import VoiceToVoicePipeline, FERNConfig
        import os
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print_warning("OPENAI_API_KEY not set, skipping full pipeline test")
            return None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print_info(f"Initializing pipeline on {device}...")
        
        config = FERNConfig(
            device=device,
            use_real_csm=True,
            log_level="INFO",
        )
        
        pipeline = VoiceToVoicePipeline(config=config)
        print_success("Pipeline initialized")
        
        # Test text processing
        test_text = "Hello, how are you today?"
        print_info(f'Processing: "{test_text}"')
        
        audio = pipeline.process_text(test_text)
        
        print_success(f"Pipeline generated audio: {audio.shape}")
        print_info(f"Duration: {len(audio) / 24000:.2f}s")
        
        return True
        
    except Exception as e:
        print_error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print_header("FERN Real Model Tests")
    
    # Check device
    if torch.cuda.is_available():
        print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print_success("MPS (Apple Silicon) available")
    else:
        print_warning("Running on CPU (will be slower)")
    
    print()
    
    # Run tests
    results = {
        "Mimi Codec": test_mimi_codec(),
        "CSM Generation": test_csm_generation(),
        "Full Pipeline": test_full_pipeline(),
    }
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for name, result in results.items():
        if result is True:
            print_success(f"{name}: PASSED")
        elif result is False:
            print_error(f"{name}: FAILED")
        else:
            print_warning(f"{name}: SKIPPED")
    
    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print()
        print_success("All tests passed! Real models are working! 🎉")
        print()
        print(f"{BOLD}You can now:{RESET}")
        print("1. Generate speech: python -m fern.cli \"Your text here\"")
        print("2. Start training: python scripts/train_lora.py")
        print("3. Run benchmarks: python examples/benchmark.py")
        return 0
    else:
        print()
        print_error("Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

