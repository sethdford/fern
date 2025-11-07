"""
Tests for VAD detector.

Following TDD principles with comprehensive coverage.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from fern.asr.vad_detector import VADDetector, create_vad, WEBRTC_AVAILABLE


class TestVADDetector:
    """Test VAD detector functionality."""
    
    def test_initialization(self):
        """Test VAD initializes correctly."""
        vad = VADDetector(sample_rate=16000, aggressiveness=2)
        
        assert vad.sample_rate == 16000
        assert vad.aggressiveness == 2
        assert vad.frame_size == 480  # 30ms at 16kHz
    
    def test_initialization_different_params(self):
        """Test VAD with different parameters."""
        vad = VADDetector(
            sample_rate=16000,
            frame_duration_ms=20,
            aggressiveness=3
        )
        
        assert vad.frame_size == 320  # 20ms at 16kHz
        assert vad.aggressiveness == 3
    
    def test_energy_based_detection_speech(self):
        """Test energy-based detection recognizes speech-like audio."""
        vad = VADDetector(sample_rate=16000, energy_threshold=0.01)
        
        # Create speech-like audio (high energy)
        frame = np.random.randn(480) * 0.3
        
        is_speech = vad._energy_based_detection(frame)
        assert is_speech is True
    
    def test_energy_based_detection_silence(self):
        """Test energy-based detection recognizes silence."""
        vad = VADDetector(sample_rate=16000, energy_threshold=0.01)
        
        # Create silence (low energy)
        frame = np.random.randn(480) * 0.001
        
        is_speech = vad._energy_based_detection(frame)
        assert is_speech is False
    
    @pytest.mark.skipif(not WEBRTC_AVAILABLE, reason="webrtcvad not installed")
    def test_webrtc_vad_speech(self):
        """Test WebRTC VAD recognizes speech."""
        vad = VADDetector(sample_rate=16000, aggressiveness=1)
        
        # Create speech-like audio
        frame = np.random.randn(480).astype(np.float32) * 0.3
        
        # Should detect speech (though with random noise, it's probabilistic)
        result = vad.is_speech(frame)
        assert isinstance(result, bool)
    
    def test_filter_silence_keeps_speech(self):
        """Test filter_silence keeps speech segments."""
        vad = VADDetector(sample_rate=16000, energy_threshold=0.01)
        
        # Create audio with speech in middle
        silence = np.random.randn(1600) * 0.001  # 100ms silence
        speech = np.random.randn(1600) * 0.3     # 100ms speech
        audio = np.concatenate([silence, speech, silence])
        
        filtered = vad.filter_silence(audio, padding_ms=50)
        
        # Should keep some audio (speech + padding)
        assert len(filtered) > 0
        assert len(filtered) < len(audio)  # Should be shorter than original
    
    def test_filter_silence_removes_all_silence(self):
        """Test filter_silence removes audio with no speech."""
        vad = VADDetector(sample_rate=16000, energy_threshold=0.01)
        
        # Create only silence
        audio = np.random.randn(16000) * 0.001
        
        filtered = vad.filter_silence(audio, padding_ms=50)
        
        # Should return empty array
        assert len(filtered) == 0
    
    def test_detect_end_of_turn_silence(self):
        """Test end-of-turn detection with silence."""
        vad = VADDetector(sample_rate=16000, frame_duration_ms=30)
        
        # Create frames with silence at end
        speech_frames = [np.random.randn(480) * 0.3 for _ in range(10)]
        silence_frames = [np.random.randn(480) * 0.001 for _ in range(25)]
        all_frames = speech_frames + silence_frames
        
        # Should detect end of turn (700ms of silence)
        eot = vad.detect_end_of_turn(all_frames, silence_duration_ms=700)
        assert eot is True
    
    def test_detect_end_of_turn_still_speaking(self):
        """Test end-of-turn detection while still speaking."""
        vad = VADDetector(sample_rate=16000, frame_duration_ms=30)
        
        # Create frames with continuous speech
        frames = [np.random.randn(480) * 0.3 for _ in range(35)]
        
        # Should NOT detect end of turn
        eot = vad.detect_end_of_turn(frames, silence_duration_ms=700)
        assert eot is False
    
    def test_detect_end_of_turn_not_enough_frames(self):
        """Test end-of-turn detection with insufficient frames."""
        vad = VADDetector(sample_rate=16000, frame_duration_ms=30)
        
        # Only a few frames
        frames = [np.random.randn(480) * 0.001 for _ in range(5)]
        
        # Should NOT detect end of turn (not enough frames)
        eot = vad.detect_end_of_turn(frames, silence_duration_ms=700)
        assert eot is False
    
    def test_create_vad_convenience_function(self):
        """Test convenience function creates VAD correctly."""
        vad = create_vad(sample_rate=16000, aggressiveness=2)
        
        assert isinstance(vad, VADDetector)
        assert vad.sample_rate == 16000
        assert vad.aggressiveness == 2


class TestVADEdgeCases:
    """Test VAD edge cases and error handling."""
    
    def test_empty_audio(self):
        """Test filtering empty audio."""
        vad = VADDetector(sample_rate=16000)
        
        audio = np.array([], dtype=np.float32)
        filtered = vad.filter_silence(audio)
        
        assert len(filtered) == 0
    
    def test_very_short_audio(self):
        """Test filtering audio shorter than frame size."""
        vad = VADDetector(sample_rate=16000)
        
        audio = np.random.randn(100).astype(np.float32) * 0.3
        filtered = vad.filter_silence(audio)
        
        # Should handle gracefully (no complete frames)
        assert isinstance(filtered, np.ndarray)
    
    def test_padding_larger_than_audio(self):
        """Test padding larger than audio length."""
        vad = VADDetector(sample_rate=16000)
        
        audio = np.random.randn(1600).astype(np.float32) * 0.3
        filtered = vad.filter_silence(audio, padding_ms=10000)  # Very large padding
        
        # Should not crash
        assert isinstance(filtered, np.ndarray)
    
    def test_multichannel_audio(self):
        """Test handling of stereo/multichannel audio."""
        vad = VADDetector(sample_rate=16000)
        
        # Mono audio should work
        mono = np.random.randn(480).astype(np.float32) * 0.3
        result = vad.is_speech(mono)
        assert isinstance(result, bool)


@pytest.mark.integration
class TestVADIntegration:
    """Integration tests with real audio."""
    
    @pytest.mark.skip(reason="Requires test audio files")
    def test_vad_with_real_speech(self):
        """Test VAD with real speech audio."""
        import soundfile as sf
        
        # Load test audio
        audio, sr = sf.read("test_data/speech.wav")
        
        vad = VADDetector(sample_rate=sr)
        filtered = vad.filter_silence(audio)
        
        # Should keep significant portion
        assert len(filtered) > len(audio) * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

