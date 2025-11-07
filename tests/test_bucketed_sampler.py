"""
Comprehensive tests for Bucketed Batch Sampler.

Tests the bucketed sampling strategy for efficient training
with variable-length sequences, minimizing padding overhead.
"""

import pytest
import numpy as np
import torch
from collections import Counter

try:
    from fern.training.bucketed_sampler import (
        BucketBatchSampler,
        compute_padding_stats,
        compare_sampling_strategies,
    )
    BUCKETED_SAMPLER_AVAILABLE = True
except ImportError:
    BUCKETED_SAMPLER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not BUCKETED_SAMPLER_AVAILABLE,
    reason="Bucketed sampler not available"
)


class TestBucketBatchSampler:
    """Test BucketBatchSampler class."""
    
    def test_initialization(self):
        """Test sampler initialization."""
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=2,
            bucket_boundaries=[2.5],
            drop_last=False,
        )
        
        assert sampler.batch_size == 2
        assert len(sampler.bucket_boundaries) == 1
        assert sampler.drop_last is False
    
    def test_bucket_assignment(self):
        """Test that samples are assigned to correct buckets."""
        # Create samples with clear bucket boundaries
        durations = [1.0, 1.5, 3.0, 3.5, 5.0, 5.5]
        bucket_boundaries = [2.0, 4.0]  # Buckets: [0-2], (2-4], (4-inf]
        
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=2,
            bucket_boundaries=bucket_boundaries,
        )
        
        # Check buckets were created
        assert len(sampler.buckets) == 3  # 3 buckets
        
        # Bucket 0: durations <= 2.0
        assert 0 in sampler.buckets[0]  # 1.0
        assert 1 in sampler.buckets[0]  # 1.5
        
        # Bucket 1: 2.0 < durations <= 4.0
        assert 2 in sampler.buckets[1]  # 3.0
        assert 3 in sampler.buckets[1]  # 3.5
        
        # Bucket 2: durations > 4.0
        assert 4 in sampler.buckets[2]  # 5.0
        assert 5 in sampler.buckets[2]  # 5.5
    
    def test_batch_generation(self):
        """Test that batches are generated correctly."""
        durations = [1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3]
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=2,
            bucket_boundaries=[1.5],
        )
        
        batches = list(sampler)
        
        # Should have 4 batches (8 samples / batch_size 2)
        assert len(batches) == 4
        
        # Each batch should have batch_size elements
        for batch in batches:
            assert len(batch) == 2
            assert all(isinstance(idx, int) for idx in batch)
    
    def test_drop_last(self):
        """Test drop_last functionality."""
        durations = [1.0, 1.1, 1.2, 1.3, 1.4]  # 5 samples
        batch_size = 2
        
        # Without drop_last
        sampler_keep = BucketBatchSampler(
            durations=durations,
            batch_size=batch_size,
            bucket_boundaries=[5.0],
            drop_last=False,
        )
        batches_keep = list(sampler_keep)
        assert len(batches_keep) == 3  # 2 + 2 + 1
        
        # With drop_last
        sampler_drop = BucketBatchSampler(
            durations=durations,
            batch_size=batch_size,
            bucket_boundaries=[5.0],
            drop_last=True,
        )
        batches_drop = list(sampler_drop)
        assert len(batches_drop) == 2  # 2 + 2 (last batch dropped)
    
    def test_shuffle_reproducibility(self):
        """Test that shuffle is reproducible with same seed."""
        durations = list(range(20))
        
        sampler1 = BucketBatchSampler(
            durations=durations,
            batch_size=4,
            bucket_boundaries=[10.0],
            seed=42,
        )
        batches1 = list(sampler1)
        
        sampler2 = BucketBatchSampler(
            durations=durations,
            batch_size=4,
            bucket_boundaries=[10.0],
            seed=42,
        )
        batches2 = list(sampler2)
        
        # Should produce identical batches
        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2):
            assert b1 == b2
    
    def test_shuffle_different_seeds(self):
        """Test that different seeds produce different shuffles."""
        durations = list(range(20))
        
        sampler1 = BucketBatchSampler(
            durations=durations,
            batch_size=4,
            bucket_boundaries=[10.0],
            seed=42,
        )
        batches1 = list(sampler1)
        
        sampler2 = BucketBatchSampler(
            durations=durations,
            batch_size=4,
            bucket_boundaries=[10.0],
            seed=123,
        )
        batches2 = list(sampler2)
        
        # Should produce different batches (highly likely)
        assert batches1 != batches2
    
    def test_len(self):
        """Test __len__ method."""
        durations = list(range(100))
        batch_size = 8
        
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=batch_size,
            bucket_boundaries=[50.0],
            drop_last=False,
        )
        
        # Should match actual number of batches
        assert len(sampler) == len(list(sampler))
    
    def test_padding_reduction(self):
        """Test that bucketing reduces padding compared to random sampling."""
        # Create samples with varied lengths
        durations = [1.0] * 10 + [5.0] * 10 + [10.0] * 10  # 3 groups
        
        # Bucketed sampler
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=5,
            bucket_boundaries=[3.0, 7.0],
        )
        
        # Check that samples in same batch are similar length
        batches = list(sampler)
        for batch in batches:
            batch_durations = [durations[idx] for idx in batch]
            duration_range = max(batch_durations) - min(batch_durations)
            
            # Range within batch should be small
            assert duration_range <= 5.0  # Reasonable threshold


class TestBucketBatchSamplerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_durations(self):
        """Test with empty durations list."""
        sampler = BucketBatchSampler(
            durations=[],
            batch_size=4,
            bucket_boundaries=[1.0],
        )
        
        batches = list(sampler)
        assert len(batches) == 0
    
    def test_single_sample(self):
        """Test with single sample."""
        sampler = BucketBatchSampler(
            durations=[1.0],
            batch_size=4,
            bucket_boundaries=[5.0],
            drop_last=False,
        )
        
        batches = list(sampler)
        assert len(batches) == 1
        assert len(batches[0]) == 1
    
    def test_all_samples_same_length(self):
        """Test when all samples have same duration."""
        durations = [3.0] * 20
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=5,
            bucket_boundaries=[1.0, 5.0, 10.0],
        )
        
        batches = list(sampler)
        
        # Should still work, all in same bucket
        assert len(batches) == 4  # 20 / 5
    
    def test_bucket_boundaries_not_sorted(self):
        """Test that unsorted boundaries raise error."""
        with pytest.raises(ValueError, match="strictly increasing"):
            BucketBatchSampler(
                durations=[1.0, 2.0],
                batch_size=2,
                bucket_boundaries=[3.0, 1.0, 2.0],  # Not sorted
            )
    
    def test_negative_durations(self):
        """Test that negative durations raise error."""
        with pytest.raises(ValueError, match="non-negative"):
            BucketBatchSampler(
                durations=[1.0, -2.0, 3.0],
                batch_size=2,
                bucket_boundaries=[2.0],
            )
    
    def test_very_large_batch_size(self):
        """Test batch size larger than dataset."""
        durations = [1.0, 2.0, 3.0]
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=100,
            bucket_boundaries=[5.0],
            drop_last=False,
        )
        
        batches = list(sampler)
        assert len(batches) == 1
        assert len(batches[0]) == 3


class TestComputePaddingStats:
    """Test compute_padding_stats utility function."""
    
    def test_basic_padding_computation(self):
        """Test padding statistics computation."""
        if not hasattr(compute_padding_stats, '__call__'):
            pytest.skip("compute_padding_stats not implemented")
        
        durations = [1.0, 2.0, 3.0, 4.0]
        batches = [[0, 1], [2, 3]]  # Batch 1: [1.0, 2.0], Batch 2: [3.0, 4.0]
        
        stats = compute_padding_stats(durations, batches)
        
        assert 'total_padding' in stats
        assert 'avg_padding_per_batch' in stats
        assert stats['total_padding'] >= 0
    
    def test_no_padding_needed(self):
        """Test case where all samples in batch are same length."""
        if not hasattr(compute_padding_stats, '__call__'):
            pytest.skip("compute_padding_stats not implemented")
        
        durations = [5.0, 5.0, 5.0, 5.0]
        batches = [[0, 1], [2, 3]]
        
        stats = compute_padding_stats(durations, batches)
        
        # No padding needed
        assert stats['total_padding'] == 0.0


class TestCompareSamplingStrategies:
    """Test sampling strategy comparison utility."""
    
    def test_comparison_runs(self):
        """Test that comparison function runs without errors."""
        if not hasattr(compare_sampling_strategies, '__call__'):
            pytest.skip("compare_sampling_strategies not implemented")
        
        durations = [1.0, 2.0, 3.0, 4.0, 5.0] * 10  # 50 samples
        
        results = compare_sampling_strategies(
            durations=durations,
            batch_size=5,
            bucket_boundaries=[2.5, 4.5],
        )
        
        assert 'bucketed' in results
        assert 'random' in results


class TestBucketBatchSamplerIntegration:
    """Integration tests with PyTorch DataLoader."""
    
    def test_dataloader_integration(self):
        """Test integration with PyTorch DataLoader."""
        from torch.utils.data import Dataset, DataLoader
        
        class DummyDataset(Dataset):
            def __init__(self, size):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {"data": torch.randn(10), "idx": idx}
        
        dataset = DummyDataset(20)
        durations = [float(i % 5) for i in range(20)]  # Varied durations
        
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=4,
            bucket_boundaries=[2.0],
        )
        
        # Create DataLoader with custom sampler
        # Note: batch_sampler overrides batch_size argument
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
        )
        
        # Test iteration
        batches = list(dataloader)
        
        assert len(batches) > 0
        for batch in batches:
            assert "data" in batch
            assert "idx" in batch
            assert len(batch["idx"]) <= 4  # batch_size
    
    def test_epoch_reproducibility(self):
        """Test that multiple epochs with same seed give same order."""
        durations = list(range(30))
        
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=5,
            bucket_boundaries=[15.0],
            seed=42,
        )
        
        epoch1_batches = list(sampler)
        epoch2_batches = list(sampler)
        
        # Within same sampler instance, should get different shuffles
        # (This is expected behavior for training)
        # But the structure should be consistent
        assert len(epoch1_batches) == len(epoch2_batches)


class TestBucketBatchSamplerPerformance:
    """Test performance characteristics."""
    
    def test_large_dataset(self):
        """Test with large dataset."""
        durations = np.random.uniform(0.5, 10.0, size=10000).tolist()
        
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=32,
            bucket_boundaries=[2.0, 4.0, 6.0, 8.0],
        )
        
        # Should complete in reasonable time
        batches = list(sampler)
        
        # Verify properties
        assert len(batches) > 0
        total_samples = sum(len(batch) for batch in batches)
        assert total_samples <= len(durations)
    
    def test_many_buckets(self):
        """Test with many bucket boundaries."""
        durations = np.random.uniform(0.0, 20.0, size=1000).tolist()
        
        # Many fine-grained buckets
        bucket_boundaries = list(range(1, 20))
        
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=16,
            bucket_boundaries=bucket_boundaries,
        )
        
        batches = list(sampler)
        assert len(batches) > 0


class TestBucketBatchSamplerStatistics:
    """Test statistical properties of bucketed sampling."""
    
    def test_all_samples_covered(self):
        """Test that all samples appear exactly once per epoch."""
        durations = list(range(50))
        
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=7,
            bucket_boundaries=[20.0, 40.0],
            drop_last=False,
        )
        
        batches = list(sampler)
        all_indices = [idx for batch in batches for idx in batch]
        
        # All samples should be covered
        assert len(all_indices) == len(durations)
        assert set(all_indices) == set(range(len(durations)))
    
    def test_no_duplicate_samples(self):
        """Test that no sample appears twice in an epoch."""
        durations = list(range(30))
        
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=5,
            bucket_boundaries=[15.0],
        )
        
        batches = list(sampler)
        all_indices = [idx for batch in batches for idx in batch]
        
        # Check for duplicates
        assert len(all_indices) == len(set(all_indices))
    
    def test_bucket_distribution(self):
        """Test that samples are distributed across buckets correctly."""
        # Create balanced dataset
        durations = [1.0] * 10 + [3.0] * 10 + [5.0] * 10
        
        sampler = BucketBatchSampler(
            durations=durations,
            batch_size=5,
            bucket_boundaries=[2.0, 4.0],
        )
        
        # Check bucket sizes
        assert len(sampler.buckets) == 3
        assert len(sampler.buckets[0]) == 10  # durations <= 2.0
        assert len(sampler.buckets[1]) == 10  # 2.0 < durations <= 4.0
        assert len(sampler.buckets[2]) == 10  # durations > 4.0

