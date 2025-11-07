#!/usr/bin/env python3
"""
Train CSM-1B with LoRA on Elise dataset.

Usage:
    python scripts/train_lora.py --config configs/ceylia_lora.yaml
    python scripts/train_lora.py --dataset Jinsaryko/Elise --output_dir models/adapters/ceylia
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fern.training.config import TrainingConfig, create_default_config
from fern.training.dataset import EliseDataset
from fern.training.trainer import CSMLoRATrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_csm_model(model_name: str = "sesame/csm-1b", device: str = "cuda"):
    """
    Load CSM-1B base model.
    
    TODO: This is a placeholder. Actual implementation should:
    1. Load CSM-1B from HuggingFace or local checkpoint
    2. Prepare model for fine-tuning
    3. Enable gradient checkpointing if needed
    """
    logger.info(f"Loading CSM model: {model_name}")
    
    try:
        # Import CSM model loading utilities
        from fern.tts.csm import models
        
        # Load model
        # For now, use stub for testing
        from fern.tts.csm.load_stub import load_csm_1b_stub
        
        model = load_csm_1b_stub(device=device)
        logger.info("CSM model loaded (using stub)")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load CSM model: {e}")
        logger.info("Using dummy model for testing")
        
        # Create dummy model for testing
        import torch.nn as nn
        
        class DummyCSM(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)
            
            def forward(self, x):
                return self.linear(x)
        
        return DummyCSM()


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CSM-1B with LoRA"
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to training config YAML",
    )
    
    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="Jinsaryko/Elise",
        help="HuggingFace dataset name or local path",
    )
    
    # Model
    parser.add_argument(
        "--base_model",
        type=str,
        default="sesame/csm-1b",
        help="Base CSM model to fine-tune",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/adapters/default"),
        help="Output directory for checkpoints",
    )
    
    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate (overrides config)",
    )
    
    # LoRA
    parser.add_argument(
        "--lora_rank",
        type=int,
        help="LoRA rank (overrides config)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        help="LoRA alpha (overrides config)",
    )
    
    # W&B
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ilava-finetuning",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="W&B run name",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu/mps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = TrainingConfig.from_yaml(args.config)
    else:
        logger.info("Using default config")
        config = create_default_config(
            dataset_name=args.dataset,
            output_dir=str(args.output_dir),
        )
    
    # Override config with CLI args
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.lora_rank:
        config.lora.rank = args.lora_rank
    if args.lora_alpha:
        config.lora.alpha = args.lora_alpha
    
    config.device = args.device
    config.seed = args.seed
    config.use_wandb = not args.no_wandb
    config.wandb_project = args.wandb_project
    config.wandb_run_name = args.wandb_run_name
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Log config
    logger.info("Training configuration:")
    logger.info(f"  Dataset: {config.data.dataset_name}")
    logger.info(f"  Base model: {config.base_model}")
    logger.info(f"  Output dir: {config.output_dir}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Effective batch: {config.effective_batch_size}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  LoRA rank: {config.lora.rank}")
    logger.info(f"  LoRA alpha: {config.lora.alpha}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  W&B: {config.use_wandb}")
    
    # Save config
    config_path = config.output_dir / "training_config.yaml"
    config.save_yaml(config_path)
    logger.info(f"Config saved to {config_path}")
    
    # Load model
    model = load_csm_model(config.base_model, config.device)
    
    # Create trainer
    trainer = CSMLoRATrainer(
        config=config,
        model=model,
    )
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("Training complete!")
    logger.info(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()

