"""Integration tests for complete pipeline."""

import pytest
import os
import numpy as np
import tempfile
import soundfile as sf

# Skip if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


class TestPipelineIntegration:
    """Integration tests for complete pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        from fern import VoiceToVoicePipeline, FERNConfig
        
        config = FERNConfig(
            device="cpu",
            rvq_iterations=16,
            log_level="ERROR"
        )
        
        pipeline = VoiceToVoicePipeline(config=config)
        
        assert pipeline is not None
        assert pipeline.asr is not None
        assert pipeline.llm is not None
        assert pipeline.tts is not None
        assert pipeline.vad is not None
    
    def test_text_processing(self):
        """Test processing text input."""
        from fern import VoiceToVoicePipeline, FERNConfig
        
        config = FERNConfig(
            device="cpu",
            rvq_iterations=16,
            log_level="ERROR"
        )
        
        pipeline = VoiceToVoicePipeline(config=config)
        
        # Process text
        audio = pipeline.process_text("Hello, how are you?")
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
    
    def test_conversation_context(self):
        """Test conversation maintains context."""
        from fern import VoiceToVoicePipeline, FERNConfig
        
        config = FERNConfig(
            device="cpu",
            rvq_iterations=16,
            log_level="ERROR"
        )
        
        pipeline = VoiceToVoicePipeline(config=config)
        
        # First turn
        pipeline.process_text("My name is Alice")
        
        # Second turn - should remember name
        history = pipeline.get_conversation_history()
        
        assert "Alice" in history
    
    def test_clear_context(self):
        """Test clearing conversation context."""
        from fern import VoiceToVoicePipeline, FERNConfig
        
        config = FERNConfig(
            device="cpu",
            rvq_iterations=16,
            log_level="ERROR"
        )
        
        pipeline = VoiceToVoicePipeline(config=config)
        
        # Add conversation
        pipeline.process_text("Hello")
        
        # Clear
        pipeline.clear_context()
        
        history = pipeline.get_conversation_history()
        assert "No conversation history" in history
    
    def test_pipeline_info(self):
        """Test getting pipeline info."""
        from fern import VoiceToVoicePipeline, FERNConfig
        
        config = FERNConfig(
            device="cpu",
            rvq_iterations=20,
            log_level="ERROR"
        )
        
        pipeline = VoiceToVoicePipeline(config=config)
        
        info = pipeline.get_pipeline_info()
        
        assert info["device"] == "cpu"
        assert info["rvq_iterations"] == 20
        assert "tts_info" in info


class TestEndToEnd:
    """End-to-end tests."""
    
    def test_complete_workflow(self):
        """Test complete workflow from text to audio."""
        from fern import VoiceToVoicePipeline, FERNConfig
        
        config = FERNConfig(
            device="cpu",
            rvq_iterations=16,
            enable_metrics=True,
            log_level="ERROR"
        )
        
        pipeline = VoiceToVoicePipeline(config=config)
        
        # Process multiple turns
        texts = [
            "What is the capital of France?",
            "Thank you",
        ]
        
        for text in texts:
            audio = pipeline.process_text(text)
            assert len(audio) > 0
        
        # Check history
        history = pipeline.get_conversation_history()
        assert len(history) > 0
    
    def test_audio_file_output(self):
        """Test saving audio to file."""
        from fern import VoiceToVoicePipeline, FERNConfig
        
        config = FERNConfig(
            device="cpu",
            rvq_iterations=16,
            log_level="ERROR"
        )
        
        pipeline = VoiceToVoicePipeline(config=config)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name
        
        try:
            # Process and save
            audio = pipeline.process_text(
                "Hello world",
                output_path=output_path
            )
            
            # Verify file exists and is valid
            assert os.path.exists(output_path)
            
            # Load and verify
            loaded_audio, sr = sf.read(output_path)
            assert len(loaded_audio) > 0
            assert sr == 24000
        
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

