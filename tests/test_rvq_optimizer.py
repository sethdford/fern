"""Tests for RVQ optimizer."""

import pytest
import torch
from fern.tts.rvq_optimizer import RVQOptimizer


class TestRVQOptimizer:
    """Test RVQOptimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = RVQOptimizer(
            num_iterations=16,
            num_codebooks=32,
            padding_method="mean"
        )
        
        assert optimizer.num_iterations == 16
        assert optimizer.num_codebooks == 32
        assert optimizer.padding_method == "mean"
    
    def test_valid_iterations(self):
        """Test only valid iteration counts are accepted."""
        # Valid
        for iterations in [16, 20, 24, 32]:
            optimizer = RVQOptimizer(num_iterations=iterations)
            assert optimizer.num_iterations == iterations
        
        # Invalid
        with pytest.raises(ValueError):
            RVQOptimizer(num_iterations=15)
    
    def test_mean_padding(self):
        """Test mean padding method."""
        optimizer = RVQOptimizer(
            num_iterations=16,
            num_codebooks=32,
            padding_method="mean"
        )
        
        # Create test tensor [batch, seq_len, num_iterations]
        rvq_output = torch.randn(2, 100, 16)
        
        padded = optimizer.pad_rvq_output(rvq_output, target_size=32)
        
        assert padded.shape == (2, 100, 32)
        
        # Check that padding uses mean
        original_mean = rvq_output.mean(dim=2, keepdim=True)
        padding_part = padded[:, :, 16:]
        
        # All padding values should be close to the mean
        assert torch.allclose(padding_part, original_mean.expand(-1, -1, 16), atol=1e-5)
    
    def test_concat_padding(self):
        """Test concat padding method."""
        optimizer = RVQOptimizer(
            num_iterations=16,
            num_codebooks=32,
            padding_method="concat"
        )
        
        rvq_output = torch.randn(2, 100, 16)
        padded = optimizer.pad_rvq_output(rvq_output, target_size=32)
        
        assert padded.shape == (2, 100, 32)
        
        # First 16 should be original
        assert torch.allclose(padded[:, :, :16], rvq_output)
    
    def test_no_padding(self):
        """Test no padding method."""
        optimizer = RVQOptimizer(
            num_iterations=16,
            num_codebooks=16,
            padding_method="none"
        )
        
        rvq_output = torch.randn(2, 100, 16)
        padded = optimizer.pad_rvq_output(rvq_output, target_size=32)
        
        # Should return as-is when padding_method is "none"
        assert padded.shape == rvq_output.shape
    
    def test_latency_reduction_estimate(self):
        """Test latency reduction estimation."""
        estimates = {
            16: 0.50,
            20: 0.35,
            24: 0.25,
            32: 0.00,
        }
        
        for iterations, expected in estimates.items():
            optimizer = RVQOptimizer(num_iterations=iterations)
            reduction = optimizer.estimate_latency_reduction()
            assert reduction == expected
    
    def test_quality_impact_estimate(self):
        """Test quality impact estimation."""
        optimizer = RVQOptimizer(num_iterations=16)
        quality = optimizer.estimate_quality_impact()
        
        assert "gpu_snr_db" in quality
        assert "cpu_snr_db" in quality
        assert "description" in quality
        
        # Check ranges
        assert isinstance(quality["gpu_snr_db"], tuple)
        assert isinstance(quality["cpu_snr_db"], tuple)
    
    def test_configuration_summary(self):
        """Test configuration summary generation."""
        optimizer = RVQOptimizer(
            num_iterations=16,
            num_codebooks=32,
            padding_method="mean"
        )
        
        summary = optimizer.get_configuration_summary()
        
        assert "16" in summary
        assert "32" in summary
        assert "mean" in summary
        assert "50%" in summary  # 50% latency reduction

