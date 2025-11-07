"""
Training module for CSM-1B fine-tuning.

This module provides infrastructure for fine-tuning CSM-1B models
using LoRA (Low-Rank Adaptation) on custom voice datasets.

Key Components:
- TrainingConfig: Hyperparameter configuration
- EliseDataset: Dataset loader for Elise/custom datasets
- CSMLoRATrainer: Fine-tuning trainer with LoRA
- EvaluationMetrics: Validation and testing metrics
"""

from fern.training.config import TrainingConfig, LoRAConfig
from fern.training.dataset import EliseDataset, VoiceDataset
from fern.training.trainer import CSMLoRATrainer
from fern.training.evaluation import EvaluationMetrics

__all__ = [
    "TrainingConfig",
    "LoRAConfig",
    "EliseDataset",
    "VoiceDataset",
    "CSMLoRATrainer",
    "EvaluationMetrics",
]

