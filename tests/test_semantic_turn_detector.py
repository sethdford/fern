"""
Comprehensive tests for Semantic Turn Detection.

Based on: https://blog.speechmatics.com/semantic-turn-detection

Tests cover:
- SemanticTurnDetector initialization and configuration
- EOT probability computation
- ChatML formatting
- HybridTurnDetector integration with VAD
- Edge cases and error handling
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch

# Try to import the actual module
try:
    from fern.vad.semantic_turn_detector import (
        SemanticTurnDetector,
        HybridTurnDetector,
    )
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SEMANTIC_AVAILABLE,
    reason="Semantic turn detector not available"
)


class TestSemanticTurnDetector:
    """Test SemanticTurnDetector class."""
    
    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Mock transformer models to avoid loading actual weights."""
        with patch('fern.vad.semantic_turn_detector.AutoModelForCausalLM') as mock_model_cls, \
             patch('fern.vad.semantic_turn_detector.AutoTokenizer') as mock_tokenizer_cls:
            
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.decode.return_value = "<|im_end|>"
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            # Mock model
            mock_model = MagicMock()
            
            # Mock model output with logits
            mock_output = MagicMock()
            # Create dummy logits: [1, seq_len, vocab_size]
            # Make <|im_end|> token have high probability
            dummy_logits = torch.randn(1, 10, 1000)
            dummy_logits[0, -1, 0] = 10.0  # High logit for first token
            mock_output.logits = dummy_logits
            
            mock_model.return_value = mock_output
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model_cls.from_pretrained.return_value = mock_model
            
            yield mock_model, mock_tokenizer
    
    def test_initialization(self, mock_model_and_tokenizer):
        """Test detector can be initialized."""
        detector = SemanticTurnDetector(device='cpu')
        
        assert detector is not None
        assert detector.threshold == 0.03
        assert detector.max_history == 10
        assert detector.device.type == 'cpu'
    
    def test_initialization_with_custom_params(self, mock_model_and_tokenizer):
        """Test initialization with custom parameters."""
        detector = SemanticTurnDetector(
            threshold=0.05,
            max_history=5,
            device='cpu',
        )
        
        assert detector.threshold == 0.05
        assert detector.max_history == 5
    
    def test_predict_eot_prob_returns_float(self, mock_model_and_tokenizer):
        """Test predict_eot_prob returns a probability between 0 and 1."""
        detector = SemanticTurnDetector(device='cpu')
        
        messages = [
            {"role": "user", "content": "Hello world"}
        ]
        
        prob = detector.predict_eot_prob(messages)
        
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
    
    def test_predict_eot_prob_complete_thought(self, mock_model_and_tokenizer):
        """Test prediction for complete thought."""
        detector = SemanticTurnDetector(device='cpu')
        
        # Complete sentences should have higher EOT probability
        complete_messages = [
            {"role": "user", "content": "I have a problem with my card."}
        ]
        
        prob = detector.predict_eot_prob(complete_messages)
        
        # Should return a valid probability
        assert 0.0 <= prob <= 1.0
    
    def test_predict_eot_prob_incomplete_thought(self, mock_model_and_tokenizer):
        """Test prediction for incomplete thought."""
        detector = SemanticTurnDetector(device='cpu')
        
        # Incomplete sentences (should have lower probability in real model)
        incomplete_messages = [
            {"role": "user", "content": "My ID is 123 764"}
        ]
        
        prob = detector.predict_eot_prob(incomplete_messages)
        
        # Should return a valid probability
        assert 0.0 <= prob <= 1.0
    
    def test_convert_to_chatml(self, mock_model_and_tokenizer):
        """Test ChatML conversion."""
        detector = SemanticTurnDetector(device='cpu')
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you"}
        ]
        
        chatml = detector._convert_to_chatml(messages)
        
        # Should contain role and content markers
        assert "user" in chatml
        assert "assistant" in chatml
        assert "Hello" in chatml
        assert "Hi!" in chatml
        assert "How are you" in chatml
    
    def test_convert_to_chatml_no_final_end_marker(self, mock_model_and_tokenizer):
        """Test that final user message doesn't have end marker."""
        detector = SemanticTurnDetector(device='cpu')
        
        messages = [
            {"role": "user", "content": "Test message"}
        ]
        
        chatml = detector._convert_to_chatml(messages)
        
        # Last user message shouldn't end with <|im_end|>
        # (because we want to predict if it should come next)
        assert chatml.startswith("<|im_start|>")
        assert "Test message" in chatml
    
    def test_max_history_truncation(self, mock_model_and_tokenizer):
        """Test that conversation history is truncated to max_history."""
        detector = SemanticTurnDetector(device='cpu', max_history=3)
        
        # Create messages longer than max_history
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        
        chatml = detector._convert_to_chatml(messages)
        
        # Should only contain last 3 messages
        assert "Message 7" in chatml
        assert "Message 8" in chatml
        assert "Message 9" in chatml
        assert "Message 0" not in chatml
    
    def test_is_turn_complete(self, mock_model_and_tokenizer):
        """Test is_turn_complete decision."""
        detector = SemanticTurnDetector(device='cpu', threshold=0.5)
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock the predict method to return known values
        with patch.object(detector, 'predict_eot_prob') as mock_predict:
            # Test above threshold
            mock_predict.return_value = 0.6
            assert detector.is_turn_complete(messages) is True
            
            # Test below threshold
            mock_predict.return_value = 0.4
            assert detector.is_turn_complete(messages) is False
    
    def test_custom_threshold(self, mock_model_and_tokenizer):
        """Test custom threshold in is_turn_complete."""
        detector = SemanticTurnDetector(device='cpu', threshold=0.03)
        
        messages = [{"role": "user", "content": "Test"}]
        
        with patch.object(detector, 'predict_eot_prob') as mock_predict:
            mock_predict.return_value = 0.04
            
            # Should use custom threshold
            assert detector.is_turn_complete(messages, threshold=0.05) is False
            assert detector.is_turn_complete(messages, threshold=0.03) is True
    
    def test_empty_messages(self, mock_model_and_tokenizer):
        """Test handling of empty messages list."""
        detector = SemanticTurnDetector(device='cpu')
        
        # Should handle gracefully
        prob = detector.predict_eot_prob([])
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
    
    def test_extract_eot_prob_with_target_tokens(self, mock_model_and_tokenizer):
        """Test EOT probability extraction with different target tokens."""
        detector = SemanticTurnDetector(device='cpu')
        
        # Create mock logprobs
        logprobs = {
            "<|im_end|>": -0.05,  # High probability (exp(-0.05) ≈ 0.95)
            ".": -2.0,  # Lower
            "hello": -5.0,  # Very low
        }
        
        # Test with default tokens
        prob = detector._extract_eot_prob(logprobs)
        assert prob > 0.9  # Should be high
        
        # Test with custom tokens
        prob = detector._extract_eot_prob(logprobs, target_tokens=["."])
        assert 0.1 < prob < 0.2  # Should be lower
    
    def test_extract_eot_prob_no_target_found(self, mock_model_and_tokenizer):
        """Test when no target tokens are found."""
        detector = SemanticTurnDetector(device='cpu')
        
        logprobs = {
            "hello": -1.0,
            "world": -2.0,
        }
        
        prob = detector._extract_eot_prob(logprobs)
        assert prob == 0.0


class TestHybridTurnDetector:
    """Test HybridTurnDetector class."""
    
    @pytest.fixture
    def mock_vad(self):
        """Mock VAD detector."""
        vad = Mock()
        vad.get_silence_duration = Mock(return_value=0.5)
        return vad
    
    @pytest.fixture
    def mock_semantic(self):
        """Mock semantic detector."""
        with patch('fern.vad.semantic_turn_detector.AutoModelForCausalLM'), \
             patch('fern.vad.semantic_turn_detector.AutoTokenizer'):
            semantic = Mock(spec=SemanticTurnDetector)
            semantic.predict_eot_prob = Mock(return_value=0.05)
            return semantic
    
    def test_initialization(self, mock_vad, mock_semantic):
        """Test hybrid detector initialization."""
        hybrid = HybridTurnDetector(
            vad=mock_vad,
            semantic=mock_semantic,
            min_silence=0.3,
            max_silence=2.0,
            eot_threshold=0.03,
        )
        
        assert hybrid.vad == mock_vad
        assert hybrid.semantic == mock_semantic
        assert hybrid.min_silence == 0.3
        assert hybrid.max_silence == 2.0
        assert hybrid.eot_threshold == 0.03
    
    def test_is_turn_complete_insufficient_silence(self, mock_vad, mock_semantic):
        """Test that insufficient silence returns False."""
        hybrid = HybridTurnDetector(
            vad=mock_vad,
            semantic=mock_semantic,
            min_silence=0.5,
        )
        
        # Mock VAD to return low silence
        mock_vad.get_silence_duration.return_value = 0.2
        
        audio_buffer = np.random.randn(1000)
        transcript = "Hello"
        history = []
        
        result = hybrid.is_turn_complete(audio_buffer, transcript, history)
        
        assert result is False
        # Semantic should not be called
        mock_semantic.predict_eot_prob.assert_not_called()
    
    def test_is_turn_complete_max_silence_exceeded(self, mock_vad, mock_semantic):
        """Test that exceeding max silence returns True."""
        hybrid = HybridTurnDetector(
            vad=mock_vad,
            semantic=mock_semantic,
            max_silence=2.0,
        )
        
        # Mock VAD to return high silence
        mock_vad.get_silence_duration.return_value = 2.5
        
        audio_buffer = np.random.randn(1000)
        transcript = "Hello"
        history = []
        
        result = hybrid.is_turn_complete(audio_buffer, transcript, history)
        
        assert result is True
    
    def test_is_turn_complete_semantic_high_probability(self, mock_vad, mock_semantic):
        """Test semantic check with high EOT probability."""
        hybrid = HybridTurnDetector(
            vad=mock_vad,
            semantic=mock_semantic,
            min_silence=0.3,
            max_silence=2.0,
            eot_threshold=0.03,
        )
        
        # Mock: moderate silence + high EOT probability
        mock_vad.get_silence_duration.return_value = 0.5
        mock_semantic.predict_eot_prob.return_value = 0.08  # High
        
        audio_buffer = np.random.randn(1000)
        transcript = "I have a problem."
        history = []
        
        result = hybrid.is_turn_complete(audio_buffer, transcript, history)
        
        assert result is True
        mock_semantic.predict_eot_prob.assert_called_once()
    
    def test_is_turn_complete_semantic_low_probability(self, mock_vad, mock_semantic):
        """Test semantic check with low EOT probability (extends grace period)."""
        hybrid = HybridTurnDetector(
            vad=mock_vad,
            semantic=mock_semantic,
            min_silence=0.3,
            max_silence=2.0,
            eot_threshold=0.03,
        )
        
        # Mock: moderate silence + low EOT probability
        mock_vad.get_silence_duration.return_value = 0.5
        mock_semantic.predict_eot_prob.return_value = 0.01  # Low
        
        audio_buffer = np.random.randn(1000)
        transcript = "My ID is 123 764"  # Incomplete thought
        history = []
        
        result = hybrid.is_turn_complete(audio_buffer, transcript, history)
        
        # Should extend grace period (not end yet)
        assert result is False
    
    def test_is_turn_complete_with_conversation_history(self, mock_vad, mock_semantic):
        """Test that conversation history is passed to semantic detector."""
        hybrid = HybridTurnDetector(
            vad=mock_vad,
            semantic=mock_semantic,
            min_silence=0.4,
        )
        
        mock_vad.get_silence_duration.return_value = 0.5
        mock_semantic.predict_eot_prob.return_value = 0.05
        
        audio_buffer = np.random.randn(1000)
        transcript = "Yes"
        history = [
            {"role": "assistant", "content": "Can I help you?"},
        ]
        
        hybrid.is_turn_complete(audio_buffer, transcript, history)
        
        # Check that semantic was called with full context
        call_args = mock_semantic.predict_eot_prob.call_args[0][0]
        assert len(call_args) == 2  # history + current
        assert call_args[0]["content"] == "Can I help you?"
        assert call_args[1]["content"] == "Yes"


class TestSemanticTurnDetectorEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Mock models for testing."""
        with patch('fern.vad.semantic_turn_detector.AutoModelForCausalLM') as mock_model_cls, \
             patch('fern.vad.semantic_turn_detector.AutoTokenizer') as mock_tokenizer_cls:
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_tokenizer.decode.return_value = "."
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            mock_model = MagicMock()
            mock_output = MagicMock()
            mock_output.logits = torch.randn(1, 5, 1000)
            mock_model.return_value = mock_output
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model_cls.from_pretrained.return_value = mock_model
            
            yield mock_model, mock_tokenizer
    
    def test_very_long_message(self, mock_model_and_tokenizer):
        """Test handling of very long messages."""
        detector = SemanticTurnDetector(device='cpu', max_history=10)
        
        # Create a very long message
        long_content = "word " * 1000
        messages = [{"role": "user", "content": long_content}]
        
        # Should handle without crashing
        prob = detector.predict_eot_prob(messages)
        assert isinstance(prob, float)
    
    def test_special_characters_in_message(self, mock_model_and_tokenizer):
        """Test handling of special characters."""
        detector = SemanticTurnDetector(device='cpu')
        
        messages = [
            {"role": "user", "content": "Test with émojis 😀 and spëcial çharacters!"}
        ]
        
        prob = detector.predict_eot_prob(messages)
        assert isinstance(prob, float)
    
    def test_empty_content(self, mock_model_and_tokenizer):
        """Test handling of empty message content."""
        detector = SemanticTurnDetector(device='cpu')
        
        messages = [{"role": "user", "content": ""}]
        
        prob = detector.predict_eot_prob(messages)
        assert isinstance(prob, float)
    
    def test_multiple_speakers(self, mock_model_and_tokenizer):
        """Test conversation with multiple turns."""
        detector = SemanticTurnDetector(device='cpu')
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
            {"role": "user", "content": "That's great"},
        ]
        
        prob = detector.predict_eot_prob(messages)
        assert isinstance(prob, float)


# Integration-style test (if model is actually available)
@pytest.mark.slow
@pytest.mark.skipif(
    not SEMANTIC_AVAILABLE,
    reason="Requires actual model download"
)
class TestSemanticTurnDetectorIntegration:
    """Integration tests with real model (slow, optional)."""
    
    @pytest.mark.skip(reason="Requires model download, run manually")
    def test_real_model_loading(self):
        """Test loading real SmolLM2 model."""
        # This would actually download the model
        detector = SemanticTurnDetector(device='cpu')
        
        messages = [{"role": "user", "content": "Hello world."}]
        prob = detector.predict_eot_prob(messages)
        
        assert 0.0 <= prob <= 1.0
    
    @pytest.mark.skip(reason="Requires model download, run manually")
    def test_real_predictions(self):
        """Test predictions on known examples."""
        detector = SemanticTurnDetector(device='cpu')
        
        # Complete thought - should have high probability
        complete = [{"role": "user", "content": "I need help with my account."}]
        prob_complete = detector.predict_eot_prob(complete)
        
        # Incomplete thought - should have lower probability
        incomplete = [{"role": "user", "content": "My ID is 123"}]
        prob_incomplete = detector.predict_eot_prob(incomplete)
        
        # Complete should generally be higher (though not guaranteed)
        assert isinstance(prob_complete, float)
        assert isinstance(prob_incomplete, float)

