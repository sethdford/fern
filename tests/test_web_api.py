"""
Tests for the web API endpoints.

Following TDD principles with comprehensive coverage.
"""

import pytest
import numpy as np
import io
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from web_client.app import app, detect_device


class TestDeviceDetection:
    """Test device detection in web API."""
    
    def test_detect_cuda(self):
        """Test CUDA detection."""
        with patch('torch.cuda.is_available', return_value=True):
            assert detect_device() == "cuda"
    
    def test_detect_mps(self):
        """Test MPS detection."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                assert detect_device() == "mps"
    
    def test_detect_cpu(self):
        """Test CPU fallback."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                assert detect_device() == "cpu"


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_models():
    """Mock all model components."""
    with patch('web_client.app.get_llm') as mock_llm, \
         patch('web_client.app.get_tts') as mock_tts, \
         patch('web_client.app.get_asr') as mock_asr:
        
        # Setup LLM mock
        llm_instance = MagicMock()
        llm_instance.generate_response.return_value = "Test response"
        mock_llm.return_value = llm_instance
        
        # Setup TTS mock
        tts_instance = MagicMock()
        tts_instance.synthesize.return_value = np.zeros(24000, dtype=np.float32)
        
        streaming_instance = MagicMock()
        streaming_instance.synthesize_stream.return_value = [np.zeros(4800, dtype=np.float32) for _ in range(5)]
        streaming_instance.synthesize_stream_sentences.return_value = [np.zeros(12000, dtype=np.float32)]
        
        mock_tts.return_value = (tts_instance, streaming_instance)
        
        # Setup ASR mock
        asr_instance = MagicMock()
        asr_instance.transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "segments": []
        }
        mock_asr.return_value = asr_instance
        
        yield {
            'llm': llm_instance,
            'tts': tts_instance,
            'streaming': streaming_instance,
            'asr': asr_instance,
        }


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test /health returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "llm_loaded" in data
        assert "tts_loaded" in data
        assert "asr_loaded" in data


class TestChatEndpoint:
    """Test /api/chat endpoint."""
    
    def test_chat_success(self, client, mock_models):
        """Test successful chat."""
        response = client.post(
            "/api/chat",
            json={"message": "Hello"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"] == "Test response"
        assert data["audio_base64"] is None
    
    def test_chat_with_history(self, client, mock_models):
        """Test chat with conversation history."""
        response = client.post(
            "/api/chat",
            json={"message": "Hello", "include_history": True}
        )
        
        assert response.status_code == 200
        mock_models['llm'].generate_response.assert_called_once_with("Hello", include_history=True)
    
    def test_chat_empty_message(self, client, mock_models):
        """Test chat with empty message."""
        response = client.post(
            "/api/chat",
            json={"message": ""}
        )
        
        assert response.status_code == 200


class TestTranscribeEndpoint:
    """Test /api/transcribe endpoint."""
    
    def test_transcribe_audio(self, client, mock_models):
        """Test audio transcription."""
        # Create fake audio file
        import soundfile as sf
        audio_buffer = io.BytesIO()
        audio_data = np.random.randn(16000).astype(np.float32)
        sf.write(audio_buffer, audio_data, 16000, format='WAV')
        audio_buffer.seek(0)
        
        # Upload
        response = client.post(
            "/api/transcribe",
            files={"audio": ("test.wav", audio_buffer, "audio/wav")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["text"] == "Hello world"
        assert data["language"] == "en"
    
    def test_transcribe_no_file(self, client):
        """Test transcribe without file."""
        response = client.post("/api/transcribe")
        assert response.status_code == 422  # Validation error


class TestSynthesizeEndpoint:
    """Test /api/synthesize endpoint."""
    
    def test_synthesize_text(self, client, mock_models):
        """Test text-to-speech synthesis."""
        response = client.post(
            "/api/synthesize",
            json={"text": "Hello world"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
    
    def test_synthesize_empty_text(self, client, mock_models):
        """Test synthesis with empty text."""
        response = client.post(
            "/api/synthesize",
            json={"text": ""}
        )
        
        assert response.status_code == 200


class TestChatWithVoiceEndpoint:
    """Test /api/chat-with-voice endpoint."""
    
    def test_chat_with_voice(self, client, mock_models):
        """Test full conversation with audio."""
        response = client.post(
            "/api/chat-with-voice",
            json={"message": "Hello"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "audio_base64" in data
        assert data["response"] == "Test response"
        assert data["audio_base64"] is not None
        assert len(data["audio_base64"]) > 0


class TestStreamingEndpoint:
    """Test /api/stream endpoint."""
    
    def test_stream_audio(self, client, mock_models):
        """Test streaming audio generation."""
        response = client.get("/api/stream/Hello%20world")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        
        # Verify content is returned
        assert len(response.content) > 0


class TestWebSocket:
    """Test WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_chat(self, mock_models):
        """Test WebSocket conversation."""
        from fastapi.testclient import TestClient
        
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Send message
                websocket.send_json({"type": "message", "text": "Hello"})
                
                # Receive response
                data = websocket.receive_json()
                assert data["type"] == "response"
                assert data["text"] == "Test response"
                
                # Receive audio chunks
                chunks_received = 0
                while True:
                    data = websocket.receive_json()
                    if data["type"] == "audio_chunk":
                        chunks_received += 1
                        assert "data" in data
                    elif data["type"] == "complete":
                        break
                
                assert chunks_received > 0


class TestCORSConfiguration:
    """Test CORS configuration."""
    
    def test_cors_default(self, client):
        """Test default CORS allows all origins."""
        response = client.options(
            "/api/chat",
            headers={"Origin": "http://example.com"}
        )
        
        # Should allow the origin
        assert response.status_code == 200
    
    def test_cors_configurable(self):
        """Test CORS can be configured via environment."""
        import os
        os.environ["CORS_ORIGINS"] = "http://localhost:3000,http://example.com"
        
        # Reload app to pick up new config
        from importlib import reload
        import web_client.app as app_module
        reload(app_module)
        
        client = TestClient(app_module.app)
        response = client.options(
            "/api/chat",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        
        # Cleanup
        del os.environ["CORS_ORIGINS"]


@pytest.mark.integration
class TestWebAPIIntegration:
    """Integration tests with real models."""
    
    @pytest.mark.skip(reason="Requires model downloads and API keys")
    def test_full_pipeline(self, client):
        """Test full pipeline with real models."""
        # This would test the entire pipeline
        # Requires GOOGLE_API_KEY and downloaded models
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

