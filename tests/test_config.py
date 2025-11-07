"""Tests for configuration module."""

import pytest
import os
from fern.config import FERNConfig, DeviceType, RVQPaddingMethod


class TestFERNConfig:
    """Test FERNConfig."""
    
    def test_default_config(self):
        """Test default configuration with API key."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        config = FERNConfig()
        
        assert config.device == DeviceType.CUDA
        assert config.rvq_iterations == 16
        assert config.enable_streaming is True
        assert config.llm_model == "gpt-4o-mini"
    
    def test_custom_config(self):
        """Test custom configuration."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        config = FERNConfig(
            device="cpu",
            rvq_iterations=32,
            enable_streaming=False,
        )
        
        assert config.device == DeviceType.CPU
        assert config.rvq_iterations == 32
        assert config.enable_streaming is False
    
    def test_rvq_iterations_validation(self):
        """Test RVQ iterations must be 16, 20, 24, or 32."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        # Valid values
        for iterations in [16, 20, 24, 32]:
            config = FERNConfig(rvq_iterations=iterations)
            assert config.rvq_iterations == iterations
        
        # Invalid value
        with pytest.raises(Exception):
            FERNConfig(rvq_iterations=15)
    
    def test_mimi_codebooks_auto_adjust(self):
        """Test mimi_codebooks auto-adjusts to match rvq_iterations."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            rvq_iterations=16,
            rvq_padding_method=RVQPaddingMethod.NONE,
        )
        
        # Should auto-adjust to match iterations
        assert config.mimi_codebooks == 16
    
    def test_api_key_from_env(self):
        """Test API key is read from environment."""
        os.environ["OPENAI_API_KEY"] = "test-key-from-env"
        config = FERNConfig()
        
        assert config.openai_api_key == "test-key-from-env"
    
    def test_api_key_required(self):
        """Test API key is required."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        with pytest.raises(ValueError):
            FERNConfig()


class TestSubConfigs:
    """Test sub-configuration classes."""
    
    def test_asr_config_from_ilava(self):
        """Test ASRConfig creation from FERNConfig."""
        from fern.config import ASRConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        config = FERNConfig(device="cpu")
        asr_config = ASRConfig.from_ilava_config(config)
        
        assert asr_config.device == "cpu"
        assert asr_config.model == config.whisper_model
    
    def test_llm_config_from_ilava(self):
        """Test LLMConfig creation from FERNConfig."""
        from fern.config import LLMConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        config = FERNConfig()
        llm_config = LLMConfig.from_ilava_config(config)
        
        assert llm_config.model == "gpt-4o-mini"
        assert llm_config.api_key == "test-key"
    
    def test_tts_config_from_ilava(self):
        """Test TTSConfig creation from FERNConfig."""
        from fern.config import TTSConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        config = FERNConfig(rvq_iterations=20)
        tts_config = TTSConfig.from_ilava_config(config)
        
        assert tts_config.rvq_iterations == 20
        assert tts_config.model == "csm-1b"

