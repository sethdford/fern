"""
Tests for config fixes - following TDD principles.

RED phase: These tests currently FAIL.
GREEN phase: We'll fix the code to make them PASS.
REFACTOR phase: Clean up if needed.
"""

import pytest
import os
from fern.config import FERNConfig, DeviceType, RVQPaddingMethod


class TestMimiCodebooksAutoAdjustment:
    """Test suite for mimi_codebooks auto-adjustment feature."""
    
    def test_codebooks_match_iterations_when_no_padding(self):
        """
        Given: RVQ iterations set to 16 with no padding
        When: Config is created
        Then: mimi_codebooks should auto-adjust to 16
        """
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            rvq_iterations=16,
            rvq_padding_method=RVQPaddingMethod.NONE,
        )
        
        assert config.mimi_codebooks == 16, \
            f"Expected codebooks=16, got {config.mimi_codebooks}"
    
    def test_codebooks_match_20_iterations(self):
        """
        Given: RVQ iterations set to 20 with no padding
        When: Config is created  
        Then: mimi_codebooks should auto-adjust to 20
        """
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            rvq_iterations=20,
            rvq_padding_method=RVQPaddingMethod.NONE,
        )
        
        assert config.mimi_codebooks == 20
    
    def test_codebooks_not_adjusted_with_padding(self):
        """
        Given: RVQ iterations=16 with MEAN padding
        When: Config is created with mimi_codebooks=32
        Then: mimi_codebooks should stay 32 (no auto-adjust with padding)
        """
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            rvq_iterations=16,
            rvq_padding_method=RVQPaddingMethod.MEAN,
            mimi_codebooks=32,
        )
        
        assert config.mimi_codebooks == 32
    
    def test_explicit_codebooks_overridden_when_no_padding(self):
        """
        Given: Explicit mimi_codebooks=32 with iterations=16 and no padding
        When: Config is created
        Then: mimi_codebooks should be overridden to 16
        """
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            rvq_iterations=16,
            rvq_padding_method=RVQPaddingMethod.NONE,
            mimi_codebooks=32,  # Explicitly set, should be overridden
        )
        
        assert config.mimi_codebooks == 16


class TestAPIKeyValidation:
    """Test suite for OpenAI API key validation."""
    
    def test_api_key_from_environment(self):
        """
        Given: API key set in environment variable
        When: Config is created without explicit key
        Then: Config should use key from environment
        """
        os.environ["OPENAI_API_KEY"] = "test-key-from-env"
        
        config = FERNConfig()
        
        assert config.openai_api_key == "test-key-from-env", \
            f"Expected env key, got {config.openai_api_key}"
    
    def test_api_key_explicit_overrides_env(self):
        """
        Given: API key in both env and explicit parameter
        When: Config is created with explicit key
        Then: Explicit key should be used
        """
        os.environ["OPENAI_API_KEY"] = "env-key"
        
        config = FERNConfig(openai_api_key="explicit-key")
        
        assert config.openai_api_key == "explicit-key"
    
    def test_api_key_required_when_missing(self):
        """
        Given: No API key in environment or parameter
        When: Config is created
        Then: ValueError should be raised
        """
        # Remove env key if present
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        with pytest.raises((ValueError, Exception)) as exc_info:
            FERNConfig()
        
        # Pydantic V2 wraps ValidationError, check message contains our text
        error_msg = str(exc_info.value).lower()
        assert "api key" in error_msg or "openai" in error_msg
    
    def test_empty_api_key_rejected(self):
        """
        Given: Empty string provided as API key
        When: Config is created
        Then: ValueError should be raised
        """
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        with pytest.raises(ValueError):
            FERNConfig(openai_api_key="")
    
    def test_api_key_in_llm_config(self):
        """
        Given: Valid API key in config
        When: LLMConfig is created from FERNConfig
        Then: LLMConfig should have the API key
        """
        from fern.config import LLMConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        config = FERNConfig()
        llm_config = LLMConfig.from_ilava_config(config)
        
        assert llm_config.api_key == "test-key", \
            f"Expected 'test-key', got '{llm_config.api_key}'"


class TestConfigIntegration:
    """Integration tests for config system."""
    
    def test_complete_config_creation(self):
        """
        Given: All required parameters
        When: Config is created
        Then: All values should be set correctly
        """
        os.environ["OPENAI_API_KEY"] = "integration-test-key"
        
        config = FERNConfig(
            device="cpu",
            rvq_iterations=20,
            rvq_padding_method=RVQPaddingMethod.NONE,
            tts_use_real_csm=False,
        )
        
        assert config.device == DeviceType.CPU
        assert config.rvq_iterations == 20
        assert config.mimi_codebooks == 20  # Auto-adjusted
        assert config.tts_use_real_csm is False
        assert config.openai_api_key == "integration-test-key"
    
    def test_tts_config_has_use_real_csm(self):
        """
        Given: Config with tts_use_real_csm set
        When: TTSConfig is created from FERNConfig
        Then: TTSConfig should have use_real_csm
        """
        from fern.config import TTSConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        config = FERNConfig(tts_use_real_csm=False)
        tts_config = TTSConfig.from_ilava_config(config)
        
        assert hasattr(tts_config, 'use_real_csm')
        assert tts_config.use_real_csm is False


if __name__ == "__main__":
    # Run tests with: pytest tests/test_config_fixes.py -v
    pytest.main([__file__, "-v"])

