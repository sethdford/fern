#!/usr/bin/env python3
"""
Pre-tokenize CSM training data for faster training.

Based on Speechmatics blog: https://blog.speechmatics.com/sesame-finetune

Pre-tokenization provides 2-3x speedup by:
1. Tokenizing audio with Mimi codec once (instead of every epoch)
2. Tokenizing text with Llama tokenizer once
3. Storing tokens in efficient pickle format
4. Reducing I/O during training

Usage:
    python scripts/pretokenize_csm.py \
        --train_data datasets/elise/train_metadata.json \
        --val_data datasets/elise/val_metadata.json \
        --output datasets/elise/tokenized.pkl
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict
import sys

import torch
import numpy as np
from tqdm import tqdm
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fern.tts.csm_real import RealCSMTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_metadata(metadata_path: Path) -> List[Dict]:
    """
    Load dataset metadata.
    
    Expected format (JSON):
    [
        {
            "audio_path": "path/to/audio.wav",
            "text": "Transcription text",
            "speaker_id": 0
        },
        ...
    ]
    """
    with open(metadata_path) as f:
        return json.load(f)


def pretokenize_dataset(
    metadata: List[Dict],
    csm_model,
    device: str = "cuda",
) -> List[Dict]:
    """
    Pre-tokenize a dataset.
    
    Args:
        metadata: List of dicts with audio_path, text, speaker_id
        csm_model: CSM model instance (for Mimi codec)
        device: Device to use
    
    Returns:
        List of dicts with tokenized data
    """
    tokenized_data = []
    
    logger.info(f"Pre-tokenizing {len(metadata)} examples...")
    
    for example in tqdm(metadata, desc="Tokenizing"):
        try:
            audio_path = example["audio_path"]
            text = example["text"]
            speaker_id = example.get("speaker_id", 0)
            
            # Load audio
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed (CSM expects 24kHz)
            target_sr = 24000
            if sr != target_sr:
                from scipy import signal
                audio = signal.resample(
                    audio,
                    int(len(audio) * target_sr / sr)
                )
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().to(device)
            
            # Tokenize audio with Mimi
            with torch.no_grad():
                audio_tokens = csm_model.mimi.encode(audio_tensor.unsqueeze(0))
            
            # Tokenize text with Llama tokenizer
            # Prepend speaker ID as per Speechmatics blog
            text_with_speaker = f"[{speaker_id}]{text}"
            
            # Get tokenizer from CSM
            from fern.tts.csm.generator import load_llama3_tokenizer
            tokenizer = load_llama3_tokenizer()
            text_tokens = tokenizer.encode(text_with_speaker, add_special_tokens=True)
            
            # Store tokenized data
            tokenized_data.append({
                "audio_tokens": audio_tokens.cpu().numpy(),
                "text_tokens": np.array(text_tokens),
                "speaker_id": speaker_id,
                "text": text,  # Keep for reference
                "duration": len(audio) / target_sr,  # For bucketing
            })
            
        except Exception as e:
            logger.warning(f"Failed to tokenize {example.get('audio_path', 'unknown')}: {e}")
            continue
    
    logger.info(f"✓ Successfully tokenized {len(tokenized_data)}/{len(metadata)} examples")
    
    return tokenized_data


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize CSM training data")
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training metadata JSON")
    parser.add_argument("--val_data", type=str, required=True,
                       help="Path to validation metadata JSON")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for tokenized data pickle")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load metadata
    logger.info("Loading metadata...")
    train_metadata = load_metadata(Path(args.train_data))
    val_metadata = load_metadata(Path(args.val_data))
    
    logger.info(f"Train examples: {len(train_metadata)}")
    logger.info(f"Val examples: {len(val_metadata)}")
    
    # Load CSM model (we need Mimi codec and tokenizer)
    logger.info("Loading CSM model (for Mimi codec)...")
    csm = RealCSMTTS(device=args.device)
    
    # Pre-tokenize datasets
    train_tokens = pretokenize_dataset(train_metadata, csm, args.device)
    val_tokens = pretokenize_dataset(val_metadata, csm, args.device)
    
    # Save tokenized data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving tokenized data to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump({
            "train": train_tokens,
            "val": val_tokens,
            "metadata": {
                "num_train": len(train_tokens),
                "num_val": len(val_tokens),
                "train_duration": sum(ex["duration"] for ex in train_tokens),
                "val_duration": sum(ex["duration"] for ex in val_tokens),
            }
        }, f)
    
    logger.info("✓ Pre-tokenization complete!")
    logger.info(f"  Train: {len(train_tokens)} examples, "
               f"{sum(ex['duration'] for ex in train_tokens) / 3600:.2f} hours")
    logger.info(f"  Val: {len(val_tokens)} examples, "
               f"{sum(ex['duration'] for ex in val_tokens) / 3600:.2f} hours")
    logger.info(f"\nNow run training with:")
    logger.info(f"  python scripts/train_lora.py --data {output_path}")


if __name__ == "__main__":
    main()

