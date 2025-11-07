"""
Training configuration for CSM-1B fine-tuning.

Defines Pydantic models for:
- LoRA hyperparameters
- Training hyperparameters
- Data preprocessing settings
- Device/hardware configuration
"""

import os
from pathlib import Path
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""
    
    rank: int = Field(
        default=16,
        ge=1,
        le=128,
        description="LoRA rank (typically 8-64)",
    )
    alpha: int = Field(
        default=32,
        ge=1,
        description="LoRA alpha scaling parameter",
    )
    dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="LoRA dropout probability",
    )
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "speaker_emb",
        ],
        description="Model modules to apply LoRA to",
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Bias training strategy",
    )
    
    @property
    def scaling(self) -> float:
        """Compute LoRA scaling factor."""
        return self.alpha / self.rank


class DataConfig(BaseModel):
    """Data preprocessing and loading configuration."""
    
    dataset_name: str = Field(
        default="Jinsaryko/Elise",
        description="HuggingFace dataset name or local path",
    )
    cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for caching preprocessed data",
    )
    
    # Splits
    train_split: float = Field(default=0.8, ge=0.0, le=1.0)
    val_split: float = Field(default=0.1, ge=0.0, le=1.0)
    test_split: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Audio preprocessing
    sample_rate: int = Field(default=24000, description="Target sample rate")
    max_duration: float = Field(
        default=10.0,
        ge=0.1,
        description="Max audio duration in seconds",
    )
    min_duration: float = Field(
        default=0.5,
        ge=0.0,
        description="Min audio duration in seconds",
    )
    normalize_audio: bool = Field(
        default=True,
        description="Normalize audio amplitude",
    )
    trim_silence: bool = Field(
        default=True,
        description="Trim leading/trailing silence",
    )
    
    @model_validator(mode='after')
    def validate_splits(self) -> 'DataConfig':
        """Ensure splits sum to 1.0."""
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Splits must sum to 1.0, got {total:.4f}"
            )
        return self


class TrainingConfig(BaseModel):
    """Complete training configuration."""
    
    # Model
    base_model: str = Field(
        default="sesame/csm-1b",
        description="Base CSM model to fine-tune",
    )
    lora: LoRAConfig = Field(
        default_factory=LoRAConfig,
        description="LoRA configuration",
    )
    
    # Data
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data configuration",
    )
    
    # Training hyperparameters
    batch_size: int = Field(default=4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
        description="Gradient accumulation steps",
    )
    num_epochs: int = Field(default=10, ge=1, description="Number of epochs")
    learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        description="Peak learning rate",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight decay (L2 regularization)",
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="LR warmup steps",
    )
    
    # Optimization
    optimizer: Literal["adam", "adamw", "sgd"] = Field(
        default="adamw",
        description="Optimizer type",
    )
    scheduler: Literal["linear", "cosine", "constant"] = Field(
        default="cosine",
        description="LR scheduler type",
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Gradient clipping threshold",
    )
    
    # Mixed precision
    mixed_precision: Optional[Literal["fp16", "bf16"]] = Field(
        default="bf16",
        description="Mixed precision training (bf16 for A100+)",
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description="Enable gradient checkpointing (saves memory)",
    )
    
    # Logging & checkpointing
    output_dir: Path = Field(
        default=Path("models/adapters/default"),
        description="Output directory for checkpoints",
    )
    logging_steps: int = Field(default=10, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    save_steps: int = Field(default=200, ge=1)
    save_total_limit: int = Field(
        default=3,
        ge=1,
        description="Max checkpoints to keep",
    )
    
    # Weights & Biases (optional)
    use_wandb: bool = Field(
        default=False,
        description="Enable W&B logging",
    )
    wandb_project: str = Field(
        default="ilava-finetuning",
        description="W&B project name",
    )
    wandb_run_name: Optional[str] = Field(
        default=None,
        description="W&B run name (auto-generated if None)",
    )
    
    # Device
    device: str = Field(
        default="cuda",
        description="Training device (cuda/cpu/mps)",
    )
    seed: int = Field(default=42, description="Random seed")
    
    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size."""
        return self.batch_size * self.gradient_accumulation_steps
    
    @field_validator("output_dir", mode='before')
    @classmethod
    def ensure_path(cls, v):
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v
    
    @model_validator(mode='after')
    def create_output_dir(self) -> 'TrainingConfig':
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'TrainingConfig':
        """Load from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def save_yaml(self, path: Path) -> None:
        """Save to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)


# Factory functions for common configurations

def create_default_config(
    dataset_name: str = "Jinsaryko/Elise",
    output_dir: str = "models/adapters/default",
) -> TrainingConfig:
    """Create default training configuration."""
    return TrainingConfig(
        data=DataConfig(dataset_name=dataset_name),
        output_dir=Path(output_dir),
    )


def create_fast_config(
    dataset_name: str = "Jinsaryko/Elise",
    output_dir: str = "models/adapters/fast",
) -> TrainingConfig:
    """Create fast training config (for testing)."""
    return TrainingConfig(
        lora=LoRAConfig(rank=8, alpha=16),
        data=DataConfig(dataset_name=dataset_name),
        batch_size=8,
        gradient_accumulation_steps=2,
        num_epochs=3,
        learning_rate=5e-4,
        eval_steps=50,
        save_steps=100,
        output_dir=Path(output_dir),
    )


def create_quality_config(
    dataset_name: str = "Jinsaryko/Elise",
    output_dir: str = "models/adapters/quality",
) -> TrainingConfig:
    """Create high-quality training config (longer, larger LoRA)."""
    return TrainingConfig(
        lora=LoRAConfig(rank=32, alpha=64, dropout=0.1),
        data=DataConfig(dataset_name=dataset_name),
        batch_size=2,
        gradient_accumulation_steps=8,
        num_epochs=20,
        learning_rate=1e-4,
        warmup_steps=200,
        eval_steps=200,
        save_steps=500,
        output_dir=Path(output_dir),
    )

