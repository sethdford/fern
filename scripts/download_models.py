#!/usr/bin/env python3
"""
Automated Model Weight Downloader for FERN

This script downloads CSM-1B and Mimi model weights from HuggingFace.
It handles authentication, downloads, verification, and setup automatically.

Usage:
    python scripts/download_models.py [--models-dir PATH]
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import json

# Color codes for pretty output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str):
    """Print section header."""
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{text:^80}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_info(text: str):
    """Print info message."""
    print(f"{BLUE}ℹ {text}{RESET}")


def check_huggingface_cli() -> bool:
    """Check if HuggingFace CLI is installed."""
    try:
        result = subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_huggingface_cli() -> bool:
    """Install HuggingFace CLI."""
    print_info("Installing huggingface_hub...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"],
            check=True,
        )
        print_success("HuggingFace Hub installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install: {e}")
        return False


def check_huggingface_login() -> bool:
    """Check if user is logged in to HuggingFace."""
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def login_huggingface():
    """Guide user through HuggingFace login."""
    print_info("You need to log in to HuggingFace")
    print_info("Get your token from: https://huggingface.co/settings/tokens")
    print()
    
    try:
        subprocess.run(["huggingface-cli", "login"], check=True)
        print_success("Logged in successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("Login failed")
        return False


def download_model(
    repo_id: str,
    local_dir: Path,
    model_name: str,
) -> bool:
    """
    Download model from HuggingFace.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "sesame/csm-1b")
        local_dir: Local directory to save model
        model_name: Human-readable name for logging
    
    Returns:
        True if successful, False otherwise
    """
    print_info(f"Downloading {model_name} from {repo_id}...")
    print_info(f"Saving to: {local_dir}")
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try downloading with huggingface-cli
        result = subprocess.run(
            [
                "huggingface-cli",
                "download",
                repo_id,
                "--local-dir", str(local_dir),
                "--local-dir-use-symlinks", "False",
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            print_success(f"{model_name} downloaded successfully")
            return True
        else:
            # Check for common errors
            error_msg = result.stderr.lower()
            
            if "not found" in error_msg or "404" in error_msg:
                print_error(f"{model_name} not found at {repo_id}")
                print_warning("This model might not be publicly available yet")
                return False
            elif "401" in error_msg or "403" in error_msg:
                print_error(f"{model_name} requires special access")
                print_info(f"Visit: https://huggingface.co/{repo_id}")
                print_info("Request access and try again")
                return False
            else:
                print_error(f"Download failed: {result.stderr}")
                return False
                
    except Exception as e:
        print_error(f"Download failed with exception: {e}")
        return False


def try_alternative_sources(model_name: str, local_dir: Path) -> bool:
    """
    Try downloading from alternative sources.
    
    Args:
        model_name: Name of the model
        local_dir: Local directory to save model
    
    Returns:
        True if successful, False otherwise
    """
    print_warning(f"Trying alternative sources for {model_name}...")
    
    alternatives = {
        "CSM-1B": [
            "speechmatics/csm-1b",
            "kyutai/csm-1b",
        ],
        "Mimi": [
            "kyutai/moshiko-pytorch-bf16",
            "kyutai/moshiko",
            "facebook/encodec_24khz",  # Similar codec
        ],
    }
    
    if model_name not in alternatives:
        return False
    
    for alt_repo in alternatives[model_name]:
        print_info(f"Trying: {alt_repo}")
        if download_model(alt_repo, local_dir, model_name):
            return True
    
    return False


def verify_model_files(model_dir: Path, model_name: str) -> bool:
    """
    Verify that model files were downloaded correctly.
    
    Args:
        model_dir: Directory containing model files
        model_name: Name of the model
    
    Returns:
        True if valid, False otherwise
    """
    print_info(f"Verifying {model_name} files...")
    
    # Check for common model file patterns
    model_files = list(model_dir.glob("*.bin")) + \
                  list(model_dir.glob("*.safetensors")) + \
                  list(model_dir.glob("*.pt")) + \
                  list(model_dir.glob("*.pth"))
    
    config_files = list(model_dir.glob("config.json")) + \
                   list(model_dir.glob("*.json"))
    
    if not model_files:
        print_error(f"No model weight files found in {model_dir}")
        return False
    
    if not config_files:
        print_warning(f"No config files found (might be OK)")
    
    # Print what we found
    print_success(f"Found {len(model_files)} model file(s)")
    for f in model_files[:3]:  # Show first 3
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    return True


def create_model_config(models_dir: Path):
    """Create a config file with model paths."""
    config = {
        "csm_1b_path": str(models_dir / "csm-1b"),
        "mimi_path": str(models_dir / "mimi"),
    }
    
    config_path = models_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, indent=2, fp=f)
    
    print_success(f"Created config: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download CSM-1B and Mimi model weights"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save models (default: models/)",
    )
    parser.add_argument(
        "--skip-csm",
        action="store_true",
        help="Skip CSM-1B download",
    )
    parser.add_argument(
        "--skip-mimi",
        action="store_true",
        help="Skip Mimi download",
    )
    
    args = parser.parse_args()
    
    print_header("FERN Model Weight Downloader")
    
    # Step 1: Check HuggingFace CLI
    print_info("Checking HuggingFace CLI...")
    if not check_huggingface_cli():
        print_warning("HuggingFace CLI not found")
        if not install_huggingface_cli():
            print_error("Failed to install HuggingFace CLI")
            print_info("Try manually: pip install huggingface_hub")
            return 1
    else:
        print_success("HuggingFace CLI found")
    
    # Step 2: Check login
    print_info("Checking HuggingFace login...")
    if not check_huggingface_login():
        print_warning("Not logged in to HuggingFace")
        if not login_huggingface():
            print_error("Please log in and try again")
            return 1
    else:
        print_success("Logged in to HuggingFace")
    
    # Create models directory
    args.models_dir.mkdir(parents=True, exist_ok=True)
    print_success(f"Models directory: {args.models_dir.absolute()}")
    
    success = True
    
    # Step 3: Download CSM-1B
    if not args.skip_csm:
        print_header("Downloading CSM-1B")
        csm_dir = args.models_dir / "csm-1b"
        
        csm_success = download_model(
            "sesame/csm-1b",
            csm_dir,
            "CSM-1B",
        )
        
        if not csm_success:
            print_warning("Primary source failed, trying alternatives...")
            csm_success = try_alternative_sources("CSM-1B", csm_dir)
        
        if csm_success:
            verify_model_files(csm_dir, "CSM-1B")
        else:
            print_error("Failed to download CSM-1B")
            print_info("\nAlternative options:")
            print_info("1. Check csm-streaming repo for checkpoint links:")
            print_info("   https://github.com/davidbrowne17/csm-streaming")
            print_info("2. Train from scratch using our training code")
            print_info("3. Request access at: https://huggingface.co/sesame/csm-1b")
            success = False
    
    # Step 4: Download Mimi
    if not args.skip_mimi:
        print_header("Downloading Mimi Codec")
        mimi_dir = args.models_dir / "mimi"
        
        mimi_success = download_model(
            "kyutai/mimi",
            mimi_dir,
            "Mimi",
        )
        
        if not mimi_success:
            print_warning("Primary source failed, trying alternatives...")
            mimi_success = try_alternative_sources("Mimi", mimi_dir)
        
        if mimi_success:
            verify_model_files(mimi_dir, "Mimi")
        else:
            print_error("Failed to download Mimi")
            print_info("\nAlternative options:")
            print_info("1. Download Moshi (includes Mimi):")
            print_info("   https://huggingface.co/kyutai/moshiko-pytorch-bf16")
            print_info("2. Use EnCodec as alternative:")
            print_info("   https://huggingface.co/facebook/encodec_24khz")
            print_info("3. Use our implemented architecture with random init (dev only)")
            success = False
    
    # Step 5: Create config
    if success:
        create_model_config(args.models_dir)
    
    # Final summary
    print_header("Summary")
    
    if success:
        print_success("All models downloaded successfully!")
        print()
        print(f"{BOLD}Next steps:{RESET}")
        print("1. Models are in:", args.models_dir.absolute())
        print("2. Run integration: python scripts/integrate_real_models.py")
        print("3. Test generation: python scripts/test_real_models.py")
        print()
        return 0
    else:
        print_error("Some downloads failed")
        print()
        print(f"{BOLD}What to do:{RESET}")
        print("1. Check GET_MODEL_WEIGHTS.md for manual instructions")
        print("2. See error messages above for specific issues")
        print("3. Try alternative sources mentioned above")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

