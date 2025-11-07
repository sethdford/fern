"""
Dataset loaders for CSM-1B fine-tuning.

Provides:
- EliseDataset: Loader for Jinsaryko/Elise dataset
- VoiceDataset: Generic voice dataset loader
- Audio preprocessing utilities
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from datasets import load_dataset

logger = logging.getLogger(__name__)


class EliseDataset(Dataset):
    """
    Dataset loader for Jinsaryko/Elise HuggingFace dataset.
    
    Features:
    - Automatic download from HuggingFace
    - Audio preprocessing (resampling, normalization, trimming)
    - Caching for efficiency
    - Train/val/test splitting
    
    Args:
        split: Dataset split ('train', 'val', or 'test')
        sample_rate: Target audio sample rate
        max_duration: Maximum audio duration in seconds
        min_duration: Minimum audio duration in seconds
        normalize_audio: Whether to normalize audio amplitude
        trim_silence: Whether to trim leading/trailing silence
        cache_dir: Directory for caching preprocessed data
        
    Example:
        >>> dataset = EliseDataset(split='train', sample_rate=24000)
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['audio', 'text', 'phonemes', 'speaker_id', ...])
    """
    
    def __init__(
        self,
        split: str = "train",
        sample_rate: int = 24000,
        max_duration: float = 10.0,
        min_duration: float = 0.5,
        normalize_audio: bool = True,
        trim_silence: bool = True,
        cache_dir: Optional[str] = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.normalize_audio = normalize_audio
        self.trim_silence = trim_silence
        self.cache_dir = cache_dir
        
        # Validate splits
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Splits must sum to 1.0")
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Load dataset
        logger.info(f"Loading Elise dataset (split={split})...")
        self._load_dataset()
        logger.info(f"Loaded {len(self)} samples")
    
    def _load_dataset(self) -> None:
        """Load dataset from HuggingFace and split."""
        # Load full dataset
        try:
            dataset = load_dataset(
                "Jinsaryko/Elise",
                split="train",
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Split into train/val/test
        total_samples = len(dataset)
        train_end = int(total_samples * self.train_split)
        val_end = train_end + int(total_samples * self.val_split)
        
        if self.split == "train":
            self.data = dataset.select(range(0, train_end))
        elif self.split == "val":
            self.data = dataset.select(range(train_end, val_end))
        elif self.split == "test":
            self.data = dataset.select(range(val_end, total_samples))
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        logger.info(
            f"Split: {self.split}, "
            f"Samples: {len(self.data)}/{total_samples}"
        )
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys:
            - audio: Tensor of shape (T,) with audio samples
            - text: String with text transcript
            - phonemes: String with phoneme sequence
            - speaker_id: Integer speaker ID (always 0 for Ceylia)
            - duration: Float audio duration in seconds
            - sample_rate: Integer sample rate
            - pitch_mean: Float mean pitch (Hz)
            - pitch_std: Float pitch std dev (Hz)
            - speaking_rate: String speaking rate category
        """
        sample = self.data[idx]
        
        # Extract audio
        audio = self._preprocess_audio(sample["audio"])
        
        # Extract metadata
        return {
            "audio": audio,
            "text": sample["text"],
            "phonemes": sample["phonemes"],
            "speaker_id": 0,  # Ceylia is speaker 0
            "speaker_name": sample["speaker_name"],
            "duration": len(audio) / self.sample_rate,
            "sample_rate": self.sample_rate,
            "pitch_mean": sample.get("utterance_pitch_mean", 0.0),
            "pitch_std": sample.get("utterance_pitch_std", 0.0),
            "speaking_rate": sample.get("speaking_rate", "normal"),
        }
    
    def _preprocess_audio(self, audio_dict: Dict) -> torch.Tensor:
        """
        Preprocess audio sample.
        
        Args:
            audio_dict: Dict with 'array' and 'sampling_rate' keys
            
        Returns:
            Preprocessed audio tensor of shape (T,)
        """
        # Extract audio array and sample rate
        audio = torch.from_numpy(audio_dict["array"]).float()
        orig_sr = audio_dict["sampling_rate"]
        
        # Resample if needed
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate,
            )
            audio = resampler(audio)
        
        # Trim silence
        if self.trim_silence:
            audio = self._trim_silence(audio)
        
        # Clip to max duration
        max_samples = int(self.max_duration * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Pad to min duration
        min_samples = int(self.min_duration * self.sample_rate)
        if len(audio) < min_samples:
            audio = torch.nn.functional.pad(
                audio,
                (0, min_samples - len(audio)),
            )
        
        # Normalize amplitude
        if self.normalize_audio:
            audio = self._normalize_audio(audio)
        
        return audio
    
    @staticmethod
    def _trim_silence(
        audio: torch.Tensor,
        threshold: float = 0.01,
    ) -> torch.Tensor:
        """Trim leading and trailing silence."""
        # Find non-silent regions
        mask = torch.abs(audio) > threshold
        if not mask.any():
            return audio
        
        # Find first and last non-silent samples
        indices = torch.where(mask)[0]
        start = indices[0].item()
        end = indices[-1].item() + 1
        
        return audio[start:end]
    
    @staticmethod
    def _normalize_audio(audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        max_val = torch.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio


class VoiceDataset(Dataset):
    """
    Generic voice dataset loader for custom datasets.
    
    Use this for datasets other than Elise. Expects data in format:
    {
        "audio_path": "path/to/audio.wav",
        "text": "transcript text",
        "speaker_id": 0,
    }
    
    Args:
        data_file: Path to JSONL file with dataset
        sample_rate: Target audio sample rate
        max_duration: Maximum audio duration in seconds
        normalize_audio: Whether to normalize audio amplitude
        
    Example:
        >>> dataset = VoiceDataset("data/my_dataset.jsonl")
        >>> sample = dataset[0]
    """
    
    def __init__(
        self,
        data_file: Path,
        sample_rate: int = 24000,
        max_duration: float = 10.0,
        normalize_audio: bool = True,
    ):
        self.data_file = Path(data_file)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.normalize_audio = normalize_audio
        
        # Load data
        self._load_data()
        logger.info(f"Loaded {len(self)} samples from {data_file}")
    
    def _load_data(self) -> None:
        """Load data from JSONL file."""
        import json
        
        self.samples = []
        with open(self.data_file) as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load audio
        audio, sr = torchaudio.load(sample["audio_path"])
        audio = audio[0]  # Take first channel if stereo
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate,
            )
            audio = resampler(audio)
        
        # Clip to max duration
        max_samples = int(self.max_duration * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Normalize
        if self.normalize_audio:
            max_val = torch.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
        
        return {
            "audio": audio,
            "text": sample["text"],
            "speaker_id": sample.get("speaker_id", 0),
            "duration": len(audio) / self.sample_rate,
            "sample_rate": self.sample_rate,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Pads audio to same length within batch.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors with keys:
        - audio: (B, T) padded audio
        - text: List of strings
        - speaker_ids: (B,) speaker IDs
        - durations: (B,) audio durations
        - attention_mask: (B, T) mask for padded regions
    """
    # Find max audio length in batch
    max_len = max(len(sample["audio"]) for sample in batch)
    
    # Pad audio
    audio_batch = []
    attention_mask = []
    
    for sample in batch:
        audio = sample["audio"]
        pad_len = max_len - len(audio)
        
        # Pad audio
        padded_audio = torch.nn.functional.pad(audio, (0, pad_len))
        audio_batch.append(padded_audio)
        
        # Create attention mask (1 for real audio, 0 for padding)
        mask = torch.ones(max_len)
        mask[len(audio):] = 0
        attention_mask.append(mask)
    
    # Stack tensors
    audio_batch = torch.stack(audio_batch)  # (B, T)
    attention_mask = torch.stack(attention_mask)  # (B, T)
    speaker_ids = torch.tensor([s["speaker_id"] for s in batch])  # (B,)
    durations = torch.tensor([s["duration"] for s in batch])  # (B,)
    
    return {
        "audio": audio_batch,
        "text": [s["text"] for s in batch],
        "speaker_ids": speaker_ids,
        "durations": durations,
        "attention_mask": attention_mask,
        "phonemes": [s.get("phonemes", "") for s in batch],
    }

