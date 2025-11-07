"""
Comprehensive tests for CSM Forward Pass.

Tests the training forward pass implementation for CSM-1B model,
including backbone loss, decoder loss, and combined loss computation.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch

try:
    from fern.tts.csm_forward import (
        CSMTrainingForward,
        apply_training_forward,
        create_causal_mask,
    )
    CSM_FORWARD_AVAILABLE = True
except ImportError:
    CSM_FORWARD_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CSM_FORWARD_AVAILABLE,
    reason="CSM forward module not available"
)


class TestCreateCausalMask:
    """Test causal mask creation."""
    
    def test_causal_mask_shape(self):
        """Test mask has correct shape."""
        mask = create_causal_mask(seq_len=10, device=torch.device('cpu'))
        
        assert mask.shape == (10, 10)
        assert mask.dtype == torch.bool
    
    def test_causal_mask_structure(self):
        """Test mask has correct causal structure."""
        mask = create_causal_mask(seq_len=5, device=torch.device('cpu'))
        
        # Upper triangle should be True (masked out)
        assert mask[0, 1].item() is True
        assert mask[0, 4].item() is True
        
        # Diagonal and lower triangle should be False
        assert mask[0, 0].item() is False
        assert mask[1, 0].item() is False
        assert mask[4, 0].item() is False
    
    def test_causal_mask_device(self):
        """Test mask is on correct device."""
        device = torch.device('cpu')
        mask = create_causal_mask(seq_len=10, device=device)
        
        assert mask.device == device
    
    def test_causal_mask_size_1(self):
        """Test edge case with sequence length 1."""
        mask = create_causal_mask(seq_len=1, device=torch.device('cpu'))
        
        assert mask.shape == (1, 1)
        assert mask[0, 0].item() is False


class TestCSMTrainingForward:
    """Test CSMTrainingForward class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock CSM model."""
        model = Mock(spec=nn.Module)
        model.device = torch.device('cpu')
        model.dtype = torch.float32
        
        # Mock model attributes
        model.n_codebooks = 32
        model.vocab_size = 2048
        
        # Mock backbone output
        def model_forward(*args, **kwargs):
            tokens = args[0] if args else kwargs.get('tokens')
            B, S = tokens.shape
            embed_dim = 1024
            # Return embeddings
            return torch.randn(B, S, embed_dim)
        
        model.side_effect = model_forward
        
        # Mock c0_head for backbone
        model.c0_head = torch.randn(1024, 2048)  # [embed_dim, vocab_size]
        
        # Mock decoder components
        model.decoder = Mock()
        model.decoder.return_value = torch.randn(4, 32, 1024)  # [B*frames, n_codebooks, embed_dim]
        
        model.projection = Mock()
        model.projection.side_effect = lambda x: x  # Identity
        
        model.audio_head = torch.randn(31, 1024, 2048)  # [n_codebooks-1, embed_dim, vocab]
        
        return model
    
    def test_initialization(self, mock_model):
        """Test CSMTrainingForward initialization."""
        training_forward = CSMTrainingForward(
            model=mock_model,
            decoder_loss_weight=0.5
        )
        
        assert training_forward.model == mock_model
        assert training_forward.decoder_loss_weight == 0.5
        assert hasattr(mock_model, 'c0_head')
    
    def test_forward_for_training_basic(self, mock_model):
        """Test basic forward pass."""
        training_forward = CSMTrainingForward(
            model=mock_model,
            decoder_loss_weight=0.5
        )
        
        # Create dummy batch
        tokens = torch.randint(0, 1000, (2, 64))  # [B, S]
        tokens_mask = torch.ones_like(tokens)
        
        # Run forward
        loss = training_forward.forward_for_training(
            tokens=tokens,
            tokens_mask=tokens_mask
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.numel() == 1  # Scalar loss
    
    def test_forward_for_training_loss_weights(self, mock_model):
        """Test different loss weight configurations."""
        # Only backbone loss
        training_forward_c0_only = CSMTrainingForward(
            model=mock_model,
            decoder_loss_weight=0.0  # No decoder loss
        )
        
        tokens = torch.randint(0, 1000, (2, 64))
        tokens_mask = torch.ones_like(tokens)
        
        loss_c0_only = training_forward_c0_only.forward_for_training(
            tokens=tokens,
            tokens_mask=tokens_mask
        )
        
        assert isinstance(loss_c0_only, torch.Tensor)
        
        # Only decoder loss
        training_forward_decoder_only = CSMTrainingForward(
            model=mock_model,
            decoder_loss_weight=1.0  # Only decoder loss
        )
        
        loss_decoder_only = training_forward_decoder_only.forward_for_training(
            tokens=tokens,
            tokens_mask=tokens_mask
        )
        
        assert isinstance(loss_decoder_only, torch.Tensor)
    
    def test_forward_for_training_shapes(self, mock_model):
        """Test with different batch shapes."""
        training_forward = CSMTrainingForward(
            model=mock_model,
            decoder_loss_weight=0.5
        )
        
        # Small batch
        tokens_small = torch.randint(0, 1000, (1, 32))
        loss_small = training_forward.forward_for_training(tokens=tokens_small)
        assert loss_small.shape == ()
        
        # Large batch
        tokens_large = torch.randint(0, 1000, (8, 128))
        loss_large = training_forward.forward_for_training(tokens=tokens_large)
        assert loss_large.shape == ()
    
    def test_forward_for_training_without_decoder(self):
        """Test forward pass when model doesn't have decoder."""
        # Create minimal model without decoder
        model = Mock(spec=nn.Module)
        model.device = torch.device('cpu')
        model.dtype = torch.float32
        
        # Only has backbone components
        def model_forward(*args, **kwargs):
            tokens = args[0] if args else kwargs.get('tokens')
            B, S = tokens.shape
            return torch.randn(B, S, 1024)
        
        model.side_effect = model_forward
        model.c0_head = torch.randn(1024, 2048)
        
        # No decoder attributes
        del model.decoder
        del model.n_codebooks
        
        training_forward = CSMTrainingForward(
            model=model,
            decoder_loss_weight=0.5
        )
        
        tokens = torch.randint(0, 1000, (2, 64))
        loss = training_forward.forward_for_training(tokens=tokens)
        
        # Should still work (fallback to c0 loss only)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestApplyTrainingForward:
    """Test apply_training_forward function."""
    
    def test_apply_adds_method(self):
        """Test that apply adds forward_for_training method."""
        model = Mock(spec=nn.Module)
        model.device = torch.device('cpu')
        model.dtype = torch.float32
        
        # Apply training forward
        modified_model = apply_training_forward(model, decoder_loss_weight=0.5)
        
        # Check method was added
        assert hasattr(modified_model, 'forward_for_training')
        assert callable(modified_model.forward_for_training)
    
    def test_apply_preserves_original_model(self):
        """Test that original model attributes are preserved."""
        model = Mock(spec=nn.Module)
        model.device = torch.device('cpu')
        model.dtype = torch.float32
        model.custom_attr = "test"
        
        modified_model = apply_training_forward(model)
        
        # Original attributes should still exist
        assert hasattr(modified_model, 'custom_attr')
        assert modified_model.custom_attr == "test"


class TestCSMForwardIntegration:
    """Integration tests for CSM forward pass."""
    
    def test_end_to_end_forward(self):
        """Test end-to-end forward pass."""
        # Create a simple mock model that can actually run
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(2000, 512)
                self.transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
                self.c0_head = nn.Parameter(torch.randn(512, 2000))
                
                # Decoder components
                self.n_codebooks = 32
                self.decoder = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
                self.projection = nn.Linear(512, 512)
                self.audio_head = nn.Parameter(torch.randn(31, 512, 2000))
            
            def forward(self, tokens, **kwargs):
                x = self.embed(tokens)
                h = self.transformer(x)
                return h
        
        model = SimpleModel()
        
        # Apply training forward
        training_forward = CSMTrainingForward(
            model=model,
            decoder_loss_weight=0.5
        )
        
        # Create batch
        tokens = torch.randint(0, 2000, (4, 64))
        
        # Forward pass
        loss = training_forward.forward_for_training(tokens=tokens)
        
        # Loss should be computed
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0
        
        # Backward should work
        loss.backward()


class TestCSMForwardEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_batch(self):
        """Test handling of edge case with minimal input."""
        model = Mock(spec=nn.Module)
        model.device = torch.device('cpu')
        model.dtype = torch.float32
        
        def model_forward(*args, **kwargs):
            tokens = args[0] if args else kwargs.get('tokens')
            B, S = tokens.shape
            return torch.randn(B, S, 1024)
        
        model.side_effect = model_forward
        model.c0_head = torch.randn(1024, 2048)
        
        training_forward = CSMTrainingForward(model=model)
        
        # Very small input
        tokens = torch.randint(0, 1000, (1, 2))
        loss = training_forward.forward_for_training(tokens=tokens)
        
        assert isinstance(loss, torch.Tensor)
    
    def test_mismatched_dimensions(self):
        """Test handling of dimension mismatches."""
        model = Mock(spec=nn.Module)
        model.device = torch.device('cpu')
        model.dtype = torch.float32
        
        # Model returns wrong dimension
        def model_forward(*args, **kwargs):
            return torch.randn(2, 64, 512)  # Different embed_dim
        
        model.side_effect = model_forward
        model.c0_head = torch.randn(1024, 2048)  # Mismatched
        
        training_forward = CSMTrainingForward(model=model)
        
        tokens = torch.randint(0, 1000, (2, 64))
        
        # Should handle gracefully or raise clear error
        try:
            loss = training_forward.forward_for_training(tokens=tokens)
            # If it succeeds, should still be a valid loss
            assert isinstance(loss, torch.Tensor)
        except RuntimeError:
            # Expected for dimension mismatch
            pass


class TestCSMForwardGradientFlow:
    """Test gradient flow through the forward pass."""
    
    def test_gradients_flow_to_backbone(self):
        """Test that gradients flow to backbone parameters."""
        class SimpleBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(512, 512)
                self.c0_head = nn.Parameter(torch.randn(512, 1000))
            
            def forward(self, tokens, **kwargs):
                # Embed and process
                x = torch.randn(tokens.shape[0], tokens.shape[1], 512, requires_grad=True)
                return self.linear(x)
        
        model = SimpleBackbone()
        
        training_forward = CSMTrainingForward(model=model, decoder_loss_weight=0.0)
        
        tokens = torch.randint(0, 1000, (2, 32))
        loss = training_forward.forward_for_training(tokens=tokens)
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        assert model.linear.weight.grad is not None
        assert model.c0_head.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCSMForwardCUDA:
    """Test CUDA functionality."""
    
    def test_forward_on_cuda(self):
        """Test forward pass on CUDA device."""
        model = Mock(spec=nn.Module)
        model.device = torch.device('cuda')
        model.dtype = torch.float32
        
        def model_forward(*args, **kwargs):
            tokens = args[0] if args else kwargs.get('tokens')
            B, S = tokens.shape
            return torch.randn(B, S, 1024, device='cuda')
        
        model.side_effect = model_forward
        model.c0_head = torch.randn(1024, 2048, device='cuda')
        
        training_forward = CSMTrainingForward(model=model)
        
        tokens = torch.randint(0, 1000, (2, 64), device='cuda')
        loss = training_forward.forward_for_training(tokens=tokens)
        
        assert loss.device.type == 'cuda'

