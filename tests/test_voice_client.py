"""
Tests for the Python voice client.

Following TDD principles with comprehensive coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from client_voice import VoiceClient, detect_device, Colors


class TestDeviceDetection:
    """Test device detection logic."""
    
    def test_detect_cuda_available(self):
        """Test CUDA device detection."""
        with patch('torch.cuda.is_available', return_value=True):
            device = detect_device()
            assert device == "cuda"
    
    def test_detect_mps_available(self):
        """Test MPS device detection."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = detect_device()
                assert device == "mps"
    
    def test_detect_cpu_fallback(self):
        """Test CPU fallback when no accelerators available."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = detect_device()
                assert device == "cpu"


class TestVoiceClient:
    """Test VoiceClient functionality."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mocked AI components."""
        with patch('client_voice.WhisperASR') as mock_asr, \
             patch('client_voice.GeminiDialogueManager') as mock_llm, \
             patch('client_voice.RealCSMTTS') as mock_tts, \
             patch('client_voice.StreamingTTS') as mock_streaming, \
             patch('client_voice.VADDetector') as mock_vad:
            
            # Setup mocks
            mock_llm_instance = MagicMock()
            mock_llm_instance.model_name = "gemini-1.5-flash"
            mock_llm.return_value = mock_llm_instance
            
            mock_asr_instance = MagicMock()
            mock_asr.return_value = mock_asr_instance
            
            mock_tts_instance = MagicMock()
            mock_tts.return_value = mock_tts_instance
            
            mock_streaming_instance = MagicMock()
            mock_streaming.return_value = mock_streaming_instance
            
            mock_vad_instance = MagicMock()
            mock_vad.return_value = mock_vad_instance
            
            yield {
                'asr': mock_asr,
                'asr_instance': mock_asr_instance,
                'llm': mock_llm,
                'llm_instance': mock_llm_instance,
                'tts': mock_tts,
                'tts_instance': mock_tts_instance,
                'streaming': mock_streaming,
                'streaming_instance': mock_streaming_instance,
                'vad': mock_vad,
                'vad_instance': mock_vad_instance,
            }
    
    def test_client_initialization(self, mock_components):
        """Test client initializes with all components."""
        client = VoiceClient(
            google_api_key="test-key",
            device="cpu",
            sample_rate=16000
        )
        
        assert client.sample_rate == 16000
        assert client.device == "cpu"
        assert client.is_recording is False
        assert client.running is True
        
        # Verify components were initialized
        mock_components['asr'].assert_called_once()
        mock_components['llm'].assert_called_once_with(api_key="test-key")
        mock_components['tts'].assert_called_once_with(device="cpu")
        mock_components['vad'].assert_called_once()
    
    def test_process_audio_transcribes_correctly(self, mock_components):
        """Test audio processing with transcription."""
        # Setup
        client = VoiceClient("test-key", device="cpu")
        audio_data = np.random.randn(16000).astype(np.float32)
        
        # Mock VAD to return filtered audio
        mock_components['vad_instance'].filter_silence.return_value = audio_data
        
        # Mock ASR to return transcription
        mock_components['asr_instance'].transcribe.return_value = {
            "text": "Hello world",
            "segments": []
        }
        
        # Mock LLM response
        mock_components['llm_instance'].generate_response.return_value = "Hi there!"
        
        # Mock TTS
        mock_components['tts_instance'].synthesize.return_value = np.zeros(24000)
        mock_components['streaming_instance'].synthesize_stream_sentences.return_value = [
            np.zeros(24000)
        ]
        
        # Execute
        client._process_audio(audio_data)
        
        # Verify VAD was used
        mock_components['vad_instance'].filter_silence.assert_called_once()
        
        # Verify transcription
        mock_components['asr_instance'].transcribe.assert_called_once()
        
        # Verify LLM was called with transcribed text
        mock_components['llm_instance'].generate_response.assert_called_once_with("Hello world")
    
    def test_process_audio_handles_empty_transcription(self, mock_components):
        """Test handling of empty transcription."""
        client = VoiceClient("test-key", device="cpu")
        audio_data = np.random.randn(16000).astype(np.float32)
        
        # Mock VAD to return filtered audio
        mock_components['vad_instance'].filter_silence.return_value = audio_data
        
        # Mock ASR to return empty transcription
        mock_components['asr_instance'].transcribe.return_value = {
            "text": "",
            "segments": []
        }
        
        # Execute
        client._process_audio(audio_data)
        
        # Verify LLM was NOT called
        mock_components['llm_instance'].generate_response.assert_not_called()
    
    def test_process_audio_handles_vad_filtering(self, mock_components):
        """Test VAD filters out non-speech audio."""
        client = VoiceClient("test-key", device="cpu")
        audio_data = np.random.randn(16000).astype(np.float32)
        
        # Mock VAD to return empty (no speech)
        mock_components['vad_instance'].filter_silence.return_value = np.array([])
        
        # Execute
        client._process_audio(audio_data)
        
        # Verify transcription was NOT attempted
        mock_components['asr_instance'].transcribe.assert_not_called()
        mock_components['llm_instance'].generate_response.assert_not_called()
    
    def test_audio_callback_records_frames(self, mock_components):
        """Test audio callback records frames correctly."""
        client = VoiceClient("test-key", device="cpu")
        client.is_recording = True
        
        # Simulate audio callback
        indata = np.random.randn(1024, 1)
        client._audio_callback(indata, 1024, None, None)
        
        # Verify frame was added to buffer
        assert len(client.audio_buffer) == 1
        np.testing.assert_array_equal(client.audio_buffer[0], indata)
    
    def test_audio_callback_ignores_when_not_recording(self, mock_components):
        """Test audio callback ignores frames when not recording."""
        client = VoiceClient("test-key", device="cpu")
        client.is_recording = False
        
        # Simulate audio callback
        indata = np.random.randn(1024, 1)
        client._audio_callback(indata, 1024, None, None)
        
        # Verify frame was NOT added to buffer
        assert len(client.audio_buffer) == 0
    
    def test_reset_conversation(self, mock_components):
        """Test conversation reset functionality."""
        client = VoiceClient("test-key", device="cpu")
        
        # Execute
        client._reset_conversation()
        
        # Verify LLM history was cleared
        mock_components['llm_instance'].clear_history.assert_called_once()


class TestColors:
    """Test color codes are defined."""
    
    def test_colors_defined(self):
        """Test all color codes are defined."""
        assert hasattr(Colors, 'BLUE')
        assert hasattr(Colors, 'GREEN')
        assert hasattr(Colors, 'YELLOW')
        assert hasattr(Colors, 'RED')
        assert hasattr(Colors, 'BOLD')
        assert hasattr(Colors, 'END')
        
        # Verify they're strings
        assert isinstance(Colors.BLUE, str)
        assert isinstance(Colors.END, str)


@pytest.mark.integration
class TestVoiceClientIntegration:
    """Integration tests requiring real components."""
    
    @pytest.mark.skip(reason="Requires API key and model downloads")
    def test_full_pipeline(self):
        """Test full pipeline with real components."""
        import os
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY not set")
        
        client = VoiceClient(api_key, device="cpu")
        
        # Generate test audio (1 second of noise)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        # Process (should handle gracefully)
        client._process_audio(audio)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

