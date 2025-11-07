#!/usr/bin/env python3
"""
Prepare dataset for training.

Downloads and preprocesses the Elise dataset (or custom dataset).

Usage:
    python scripts/prepare_dataset.py --dataset Jinsaryko/Elise --output_dir data/elise
"""

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from fern.training.dataset import EliseDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for training"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="Jinsaryko/Elise",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/elise"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Target sample rate",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=10.0,
        help="Max audio duration (seconds)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Train split ratio",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Test split ratio",
    )
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Preparing dataset: {args.dataset}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load datasets for all splits
    splits = ["train", "val", "test"]
    stats = {}
    
    for split in splits:
        logger.info(f"\nProcessing {split} split...")
        
        dataset = EliseDataset(
            split=split,
            sample_rate=args.sample_rate,
            max_duration=args.max_duration,
            cache_dir=str(args.output_dir / "cache"),
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
        )
        
        logger.info(f"  Samples: {len(dataset)}")
        
        # Compute statistics
        durations = []
        texts = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            durations.append(sample["duration"])
            texts.append(sample["text"])
        
        import numpy as np
        
        stats[split] = {
            "num_samples": len(dataset),
            "total_duration": sum(durations),
            "mean_duration": np.mean(durations),
            "std_duration": np.std(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "mean_text_length": np.mean([len(t) for t in texts]),
        }
        
        logger.info(f"  Total duration: {stats[split]['total_duration']:.2f}s")
        logger.info(f"  Mean duration: {stats[split]['mean_duration']:.2f}s")
        logger.info(f"  Mean text length: {stats[split]['mean_text_length']:.1f} chars")
    
    # Save statistics
    stats_file = args.output_dir / "dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nDataset statistics saved to {stats_file}")
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 50)
    
    total_samples = sum(s["num_samples"] for s in stats.values())
    total_duration = sum(s["total_duration"] for s in stats.values())
    
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total duration: {total_duration / 3600:.2f} hours")
    logger.info(f"\nSplit breakdown:")
    for split in splits:
        logger.info(
            f"  {split:5s}: {stats[split]['num_samples']:4d} samples "
            f"({stats[split]['total_duration'] / 60:.1f} min)"
        )
    
    logger.info("\nDataset ready for training!")
    logger.info(f"Cache directory: {args.output_dir / 'cache'}")


if __name__ == "__main__":
    main()

