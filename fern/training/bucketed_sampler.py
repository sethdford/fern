"""
Bucketed Sampling for Efficient Training.

Groups samples by similar duration to minimize padding waste.

Based on: https://blog.speechmatics.com/sesame-finetune

Example:
    >>> from fern.training.bucketed_sampler import BucketBatchSampler
    >>> sampler = BucketBatchSampler(
    ...     durations=[1.5, 2.3, 1.8, 9.2, 3.1, 8.5],
    ...     batch_size=2,
    ...     bucket_boundaries=[2.0, 4.0, 6.0, 8.0],
    ... )
    >>> for batch_indices in sampler:
    ...     print(batch_indices)
    [0, 2]  # Similar durations: 1.5s, 1.8s
    [1, 4]  # Similar durations: 2.3s, 3.1s
    [3, 5]  # Similar durations: 9.2s, 8.5s
"""

import bisect
import logging
from typing import List, Iterator, Optional
import numpy as np
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class BucketBatchSampler(Sampler):
    """
    Batch sampler that groups samples by duration into buckets.
    
    This minimizes padding waste by ensuring samples in the same batch
    have similar durations.
    
    Args:
        durations: List of sample durations (seconds)
        batch_size: Number of samples per batch
        bucket_boundaries: List of duration boundaries for buckets (seconds)
        drop_last: Whether to drop incomplete batches
        shuffle: Whether to shuffle within buckets
        
    Example:
        >>> durations = [1.2, 5.3, 1.8, 9.1, 2.5, 8.7, 1.5, 2.8]
        >>> sampler = BucketBatchSampler(
        ...     durations=durations,
        ...     batch_size=2,
        ...     bucket_boundaries=[2.0, 4.0, 6.0, 8.0],
        ... )
        >>> num_batches = len(sampler)
        >>> print(f"Created {num_batches} batches from {len(durations)} samples")
    """
    
    def __init__(
        self,
        durations: List[float],
        batch_size: int,
        bucket_boundaries: Optional[List[float]] = None,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.durations = durations
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Default bucket boundaries if not provided
        if bucket_boundaries is None:
            bucket_boundaries = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.bucket_boundaries = sorted(bucket_boundaries)
        
        # Create buckets
        self.buckets = self._create_buckets()
        
        # Log statistics
        self._log_bucket_stats()
    
    def _create_buckets(self) -> List[List[int]]:
        """
        Assign each sample to a bucket based on duration.
        
        Returns:
            List of buckets, where each bucket is a list of sample indices
        """
        # Create empty buckets
        n_buckets = len(self.bucket_boundaries) + 1
        buckets = [[] for _ in range(n_buckets)]
        
        # Assign samples to buckets
        for idx, duration in enumerate(self.durations):
            bucket_idx = bisect.bisect_left(self.bucket_boundaries, duration)
            buckets[bucket_idx].append(idx)
        
        return buckets
    
    def _log_bucket_stats(self) -> None:
        """Log bucket statistics."""
        logger.info("Bucketed Sampling Statistics:")
        logger.info(f"  Total samples: {len(self.durations)}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Num buckets: {len(self.buckets)}")
        
        for i, bucket in enumerate(self.buckets):
            if len(bucket) == 0:
                continue
            
            # Get duration range for this bucket
            if i == 0:
                range_str = f"< {self.bucket_boundaries[0]:.1f}s"
            elif i == len(self.buckets) - 1:
                range_str = f"> {self.bucket_boundaries[-1]:.1f}s"
            else:
                range_str = f"{self.bucket_boundaries[i-1]:.1f}s - {self.bucket_boundaries[i]:.1f}s"
            
            # Compute statistics
            bucket_durations = [self.durations[idx] for idx in bucket]
            mean_dur = np.mean(bucket_durations)
            std_dur = np.std(bucket_durations)
            
            logger.info(
                f"  Bucket {i} ({range_str}): "
                f"{len(bucket)} samples, "
                f"mean={mean_dur:.2f}s, "
                f"std={std_dur:.2f}s"
            )
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over batches.
        
        Yields:
            List of sample indices for each batch
        """
        # Shuffle within each bucket
        if self.shuffle:
            shuffled_buckets = []
            for bucket in self.buckets:
                shuffled_bucket = list(bucket)
                np.random.shuffle(shuffled_bucket)
                shuffled_buckets.append(shuffled_bucket)
        else:
            shuffled_buckets = [list(bucket) for bucket in self.buckets]
        
        # Create batches from each bucket
        for bucket in shuffled_buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                
                # Drop last incomplete batch if requested
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                
                yield batch
    
    def __len__(self) -> int:
        """
        Return number of batches.
        
        Returns:
            Number of batches that will be yielded
        """
        total_batches = 0
        
        for bucket in self.buckets:
            n_samples = len(bucket)
            n_batches = n_samples // self.batch_size
            
            # Add 1 if there's a remainder and we're not dropping last
            if n_samples % self.batch_size != 0 and not self.drop_last:
                n_batches += 1
            
            total_batches += n_batches
        
        return total_batches


def compute_padding_stats(
    durations: List[float],
    batch_indices: List[List[int]],
) -> dict:
    """
    Compute padding statistics for batches.
    
    Args:
        durations: List of all sample durations
        batch_indices: List of batches (each batch is list of indices)
    
    Returns:
        Dictionary with padding statistics
    
    Example:
        >>> durations = [1.0, 1.2, 5.0, 5.5]
        >>> batch_indices = [[0, 1], [2, 3]]  # Bucketed
        >>> stats = compute_padding_stats(durations, batch_indices)
        >>> print(f"Avg padding: {stats['avg_padding_ratio']:.1%}")
    """
    total_padding = 0.0
    total_actual = 0.0
    
    for batch_idx in batch_indices:
        batch_durations = [durations[i] for i in batch_idx]
        max_duration = max(batch_durations)
        
        # Total time if no padding
        actual_time = sum(batch_durations)
        
        # Total time with padding
        padded_time = max_duration * len(batch_durations)
        
        # Padding waste
        padding = padded_time - actual_time
        
        total_padding += padding
        total_actual += actual_time
    
    # Compute statistics
    padding_ratio = total_padding / (total_actual + total_padding) if total_actual > 0 else 0.0
    
    return {
        "total_padding": total_padding,
        "total_actual": total_actual,
        "padding_ratio": padding_ratio,
        "avg_padding_ratio": padding_ratio,
    }


def compare_sampling_strategies(
    durations: List[float],
    batch_size: int,
    bucket_boundaries: Optional[List[float]] = None,
) -> dict:
    """
    Compare bucketed vs. random sampling strategies.
    
    Args:
        durations: List of sample durations
        batch_size: Batch size
        bucket_boundaries: Bucket boundaries for bucketed sampling
    
    Returns:
        Dictionary with comparison results
    
    Example:
        >>> durations = [np.random.uniform(1, 10) for _ in range(100)]
        >>> results = compare_sampling_strategies(durations, batch_size=8)
        >>> print(f"Padding reduction: {results['padding_reduction']:.1%}")
    """
    # Random sampling (baseline)
    n_batches = len(durations) // batch_size
    random_batch_indices = [
        list(range(i * batch_size, (i + 1) * batch_size))
        for i in range(n_batches)
    ]
    random_stats = compute_padding_stats(durations, random_batch_indices)
    
    # Bucketed sampling
    bucketed_sampler = BucketBatchSampler(
        durations=durations,
        batch_size=batch_size,
        bucket_boundaries=bucket_boundaries,
        drop_last=True,
        shuffle=False,
    )
    bucketed_batch_indices = list(bucketed_sampler)
    bucketed_stats = compute_padding_stats(durations, bucketed_batch_indices)
    
    # Compute improvement
    padding_reduction = (
        (random_stats['padding_ratio'] - bucketed_stats['padding_ratio'])
        / random_stats['padding_ratio']
        if random_stats['padding_ratio'] > 0
        else 0.0
    )
    
    return {
        "random_padding": random_stats['padding_ratio'],
        "bucketed_padding": bucketed_stats['padding_ratio'],
        "padding_reduction": padding_reduction,
        "random_batches": len(random_batch_indices),
        "bucketed_batches": len(bucketed_batch_indices),
    }


if __name__ == "__main__":
    # Demo
    import matplotlib.pyplot as plt
    
    # Generate synthetic durations
    np.random.seed(42)
    durations = [
        *np.random.uniform(1, 3, 40).tolist(),  # Short utterances
        *np.random.uniform(3, 6, 30).tolist(),  # Medium
        *np.random.uniform(6, 10, 20).tolist(), # Long
    ]
    
    # Compare strategies
    results = compare_sampling_strategies(durations, batch_size=4)
    
    print("\n" + "=" * 60)
    print("Bucketed Sampling Comparison")
    print("=" * 60)
    print(f"Dataset: {len(durations)} samples")
    print(f"Batch size: 4")
    print(f"\nRandom Sampling:")
    print(f"  Padding ratio: {results['random_padding']:.1%}")
    print(f"  Num batches: {results['random_batches']}")
    print(f"\nBucketed Sampling:")
    print(f"  Padding ratio: {results['bucketed_padding']:.1%}")
    print(f"  Num batches: {results['bucketed_batches']}")
    print(f"\nImprovement:")
    print(f"  Padding reduction: {results['padding_reduction']:.1%}")
    print("=" * 60)

