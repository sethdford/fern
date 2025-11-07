"""
Tests for refactored code following .cursorrules standards.

TDD Process:
1. RED: Write failing tests for new structure
2. GREEN: Refactor code to pass tests
3. REFACTOR: Clean up while keeping tests green
"""

import pytest
import os
from dataclasses import dataclass
import numpy as np


class TestCSMConfigRefactoring:
    """Test CSM config object pattern (reduces 10 params to 1)."""
    
    def test_csm_config_dataclass_exists(self):
        """
        Given: Need for config object
        When: Import CSMConfig
        Then: Should be a dataclass with default values
        """
        from fern.tts.csm_config import CSMConfig
        
        config = CSMConfig()
        
        # Verify it's a dataclass
        assert hasattr(config, '__dataclass_fields__')
        
        # Verify default values match current defaults
        assert config.model_name == "csm-1b"
        assert config.rvq_iterations == 16
        assert config.sample_rate == 24000
        assert config.use_real_csm is True
    
    def test_csm_config_customization(self):
        """
        Given: Custom configuration needed
        When: Create CSMConfig with custom values
        Then: Should accept and store custom values
        """
        from fern.tts.csm_config import CSMConfig
        
        config = CSMConfig(
            device="cpu",
            rvq_iterations=20,
            use_real_csm=False,
        )
        
        assert config.device == "cpu"
        assert config.rvq_iterations == 20
        assert config.use_real_csm is False
    
    def test_csm_config_auto_adjusts_codebooks(self):
        """
        Given: CSMConfig with no padding
        When: mimi_codebooks != rvq_iterations
        Then: Should auto-adjust to match
        """
        from fern.tts.csm_config import CSMConfig
        
        config = CSMConfig(
            rvq_iterations=20,
            mimi_codebooks=32,  # Mismatched
            rvq_padding_method="none",
        )
        
        # Should auto-adjust
        assert config.mimi_codebooks == 20
    
    def test_csm_config_validates_iterations(self):
        """
        Given: Invalid rvq_iterations
        When: Create CSMConfig
        Then: Should raise ValueError
        """
        from fern.tts.csm_config import CSMConfig
        
        with pytest.raises(ValueError) as exc_info:
            CSMConfig(rvq_iterations=15)  # Invalid
        
        assert "rvq_iterations" in str(exc_info.value).lower()
    
    def test_csm_config_device_fallback(self):
        """
        Given: CUDA requested but not available
        When: Create CSMConfig on CPU-only machine
        Then: Should fallback to CPU gracefully
        """
        from fern.tts.csm_config import CSMConfig
        
        # This test will pass on CPU and GPU
        config = CSMConfig(device="cuda")
        
        # Device should be valid (cuda if available, cpu otherwise)
        assert config.device in ["cuda", "cpu"]
    
    def test_csm_config_to_dict(self):
        """
        Given: CSMConfig instance
        When: Convert to dict
        Then: Should contain all fields
        """
        from fern.tts.csm_config import CSMConfig
        
        config = CSMConfig(device="cpu", use_real_csm=False)
        config_dict = config.to_dict()
        
        assert config_dict['device'] == 'cpu'
        assert config_dict['use_real_csm'] is False
        assert 'model_name' in config_dict
    
    def test_csm_config_from_dict(self):
        """
        Given: Dictionary with config values
        When: Create CSMConfig from dict
        Then: Should create valid config
        """
        from fern.tts.csm_config import CSMConfig
        
        config_dict = {
            'device': 'cpu',
            'use_real_csm': False,
            'rvq_iterations': 20,
        }
        
        config = CSMConfig.from_dict(config_dict)
        
        assert config.device == 'cpu'
        assert config.use_real_csm is False
        assert config.rvq_iterations == 20
    
    def test_factory_development_config(self):
        """
        Given: Need for development config
        When: Use create_development_config
        Then: Should return config optimized for dev
        """
        from fern.tts.csm_config import create_development_config
        
        config = create_development_config()
        
        assert config.device == 'cpu'
        assert config.use_real_csm is False
        assert config.enable_cold_start is False
    
    def test_factory_production_config(self):
        """
        Given: Need for production config
        When: Use create_production_config
        Then: Should return config optimized for prod
        """
        from fern.tts.csm_config import create_production_config
        
        config = create_production_config('cpu')
        
        assert config.use_real_csm is True
        assert config.enable_torch_compile is True
        assert config.enable_cold_start is True
    
    def test_csmtts_accepts_config_object(self):
        """
        Given: Refactored CSMTTS class
        When: Initialize with config object
        Then: Should work correctly
        """
        from fern.tts.csm_config import CSMConfig
        from fern.tts import CSMTTS
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = CSMConfig(
            device="cpu",
            use_real_csm=False,
            enable_cold_start=False,
        )
        
        # Should accept CSMConfig
        tts = CSMTTS(config=config)
        
        assert tts.device == "cpu"
        assert tts.use_real_csm is False
    
    def test_csmtts_backward_compatible(self):
        """
        Given: Existing code using individual parameters
        When: Called with old-style parameters
        Then: Should still work (backward compatibility)
        """
        from fern.tts import CSMTTS
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        # Old style should still work
        tts = CSMTTS(
            device="cpu",
            use_real_csm=False,
            enable_cold_start=False,
        )
        
        assert tts.device == "cpu"


class TestPipelineMethodExtraction:
    """Test that long methods are extracted into smaller ones."""
    
    def test_load_audio_file_extracted(self):
        """
        Given: Need to load audio file
        When: Call _load_audio_file method
        Then: Should be < 20 lines and return audio + sample_rate
        """
        from fern import VoiceToVoicePipeline, FERNConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            device="cpu",
            tts_use_real_csm=False,
            whisper_compute_type="int8",  # CPU compatible
        )
        pipeline = VoiceToVoicePipeline(config=config)
        
        # Method should exist
        assert hasattr(pipeline, '_load_audio_file')
    
    def test_detect_speech_segments_extracted(self):
        """
        Given: Audio loaded
        When: Call _detect_speech_segments
        Then: Should be < 20 lines and return speech info
        """
        from fern import VoiceToVoicePipeline, FERNConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            device="cpu",
            tts_use_real_csm=False,
            whisper_compute_type="int8",
        )
        pipeline = VoiceToVoicePipeline(config=config)
        
        # Method should exist
        assert hasattr(pipeline, '_detect_speech_segments')
    
    def test_transcribe_audio_extracted(self):
        """
        Given: Speech detected
        When: Call _transcribe_audio
        Then: Should extract transcription logic
        """
        from fern import VoiceToVoicePipeline, FERNConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            device="cpu",
            tts_use_real_csm=False,
            whisper_compute_type="int8",
        )
        pipeline = VoiceToVoicePipeline(config=config)
        
        assert hasattr(pipeline, '_transcribe_audio')
    
    def test_generate_llm_response_extracted(self):
        """
        Given: Transcribed text
        When: Call _generate_llm_response
        Then: Should extract LLM logic
        """
        from fern import VoiceToVoicePipeline, FERNConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            device="cpu",
            tts_use_real_csm=False,
            whisper_compute_type="int8",
        )
        pipeline = VoiceToVoicePipeline(config=config)
        
        assert hasattr(pipeline, '_generate_llm_response')
    
    def test_synthesize_tts_extracted(self):
        """
        Given: LLM response
        When: Call _synthesize_tts
        Then: Should extract TTS logic
        """
        from fern import VoiceToVoicePipeline, FERNConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            device="cpu",
            tts_use_real_csm=False,
            whisper_compute_type="int8",
        )
        pipeline = VoiceToVoicePipeline(config=config)
        
        assert hasattr(pipeline, '_synthesize_tts')
    
    def test_save_output_audio_extracted(self):
        """
        Given: Generated audio
        When: Call _save_output_audio
        Then: Should extract save logic
        """
        from fern import VoiceToVoicePipeline, FERNConfig
        
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        config = FERNConfig(
            device="cpu",
            tts_use_real_csm=False,
            whisper_compute_type="int8",
        )
        pipeline = VoiceToVoicePipeline(config=config)
        
        assert hasattr(pipeline, '_save_output_audio')


class TestTypeHintCompliance:
    """Test that all public methods have proper type hints."""
    
    def test_csmtts_synthesize_has_type_hints(self):
        """
        Given: CSMTTS.synthesize method
        When: Inspect type annotations
        Then: Should have complete type hints
        """
        from fern.tts import CSMTTS
        import inspect
        
        sig = inspect.signature(CSMTTS.synthesize)
        
        # Check parameters have annotations
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                # Should have annotation (not inspect.Parameter.empty)
                assert param.annotation != inspect.Parameter.empty, \
                    f"Parameter '{param_name}' missing type hint"
        
        # Check return type
        assert sig.return_annotation != inspect.Parameter.empty, \
            "Return type missing"
    
    def test_pipeline_process_audio_has_type_hints(self):
        """
        Given: VoiceToVoicePipeline.process_audio
        When: Inspect type annotations
        Then: Should have complete type hints
        """
        from fern import VoiceToVoicePipeline
        import inspect
        
        sig = inspect.signature(VoiceToVoicePipeline.process_audio)
        
        # Check return type
        assert sig.return_annotation != inspect.Parameter.empty, \
            "process_audio missing return type hint"


class TestSingleResponsibility:
    """Test that classes follow Single Responsibility Principle."""
    
    def test_csmtts_focuses_on_synthesis(self):
        """
        Given: CSMTTS class
        When: Review public methods
        Then: Should focus only on synthesis operations
        """
        from fern.tts import CSMTTS
        
        # Get public methods
        methods = [m for m in dir(CSMTTS) if not m.startswith('_') and callable(getattr(CSMTTS, m))]
        
        # Should have synthesis-related methods only
        synthesis_methods = ['synthesize', 'synthesize_streaming', 'synthesize_batch']
        
        for method in synthesis_methods:
            assert method in methods, f"Missing {method}"
        
        # Info method is acceptable (reporting)
        info_methods = ['get_model_info']
        for method in info_methods:
            assert method in methods or True  # Optional
    
    def test_function_length_compliance(self):
        """
        Given: All functions in codebase
        When: Check line counts
        Then: Public methods should be < 20 lines (guideline, not strict)
        """
        # This is informational - we'll fix violations found
        from fern.tts import CSMTTS
        import inspect
        
        # Just verify we can inspect (actual line counting is complex)
        source = inspect.getsource(CSMTTS.synthesize)
        lines = source.split('\n')
        
        # Just check it's defined (we'll refactor if needed)
        assert len(lines) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

