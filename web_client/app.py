"""
FERN Web Client - FastAPI Backend

Provides REST API for voice conversational AI:
- /api/chat - Text-based conversation
- /api/synthesize - TTS generation
- /api/stream - Streaming audio response
- WebSocket support for real-time audio

Run:
    uvicorn web_client.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
from pathlib import Path
import base64
import io
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import soundfile as sf
import numpy as np

from fern.llm.gemini_manager import GeminiDialogueManager
from fern.tts.csm_real import RealCSMTTS
from fern.tts.csm_streaming import StreamingTTS


# Initialize FastAPI
app = FastAPI(
    title="FERN Voice API",
    description="Conversational AI with streaming voice",
    version="1.0.0"
)

# CORS for web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (lazy loading)
llm: Optional[GeminiDialogueManager] = None
tts: Optional[RealCSMTTS] = None
streaming_tts: Optional[StreamingTTS] = None


def get_llm():
    """Get or initialize LLM."""
    global llm
    if llm is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set")
        llm = GeminiDialogueManager(api_key=api_key)
    return llm


def get_tts():
    """Get or initialize TTS."""
    global tts, streaming_tts
    if tts is None:
        device = "cuda" if os.path.exists("/proc/driver/nvidia/version") else "cpu"
        tts = RealCSMTTS(device=device)
        streaming_tts = StreamingTTS(tts, chunk_duration_ms=200)
    return tts, streaming_tts


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    include_history: bool = True


class ChatResponse(BaseModel):
    response: str
    audio_base64: Optional[str] = None


class SynthesizeRequest(BaseModel):
    text: str


# API Endpoints

@app.get("/")
async def root():
    """Serve the web client."""
    return FileResponse("web_client/index.html")


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "llm_loaded": llm is not None,
        "tts_loaded": tts is not None,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - get text response.
    
    Example:
        curl -X POST http://localhost:8000/api/chat \\
            -H "Content-Type: application/json" \\
            -d '{"message": "Hello!"}'
    """
    try:
        llm_instance = get_llm()
        response_text = llm_instance.generate_response(
            request.message,
            include_history=request.include_history
        )
        
        return ChatResponse(
            response=response_text,
            audio_base64=None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    TTS synthesis endpoint - convert text to audio.
    
    Returns WAV audio file.
    
    Example:
        curl -X POST http://localhost:8000/api/synthesize \\
            -H "Content-Type: application/json" \\
            -d '{"text": "Hello world"}' \\
            --output response.wav
    """
    try:
        tts_instance, _ = get_tts()
        
        # Generate audio
        audio = tts_instance.synthesize(request.text)
        
        # Convert to numpy
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio, 24000, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=response.wav"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat-with-voice", response_model=ChatResponse)
async def chat_with_voice(request: ChatRequest):
    """
    Full conversation - text in, text + audio out.
    
    Returns JSON with response text and base64-encoded audio.
    
    Example:
        curl -X POST http://localhost:8000/api/chat-with-voice \\
            -H "Content-Type: application/json" \\
            -d '{"message": "Hello!"}' | jq
    """
    try:
        llm_instance = get_llm()
        tts_instance, _ = get_tts()
        
        # Generate text response
        response_text = llm_instance.generate_response(
            request.message,
            include_history=request.include_history
        )
        
        # Generate audio
        audio = tts_instance.synthesize(response_text)
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()
        
        # Encode audio as base64
        buffer = io.BytesIO()
        sf.write(buffer, audio, 24000, format='WAV')
        audio_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return ChatResponse(
            response=response_text,
            audio_base64=audio_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stream/{text}")
async def stream_audio(text: str):
    """
    Streaming audio endpoint.
    
    Returns audio chunks as they're generated.
    
    Example:
        curl http://localhost:8000/api/stream/Hello%20world --output response.wav
    """
    try:
        _, streaming_tts_instance = get_tts()
        
        def generate_chunks():
            """Generator for audio chunks."""
            for chunk in streaming_tts_instance.synthesize_stream(text):
                # Convert chunk to WAV bytes
                buffer = io.BytesIO()
                sf.write(buffer, chunk, 24000, format='WAV')
                yield buffer.getvalue()
        
        return StreamingResponse(
            generate_chunks(),
            media_type="audio/wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time voice conversation.
    
    Protocol:
        Client sends: {"type": "message", "text": "hello"}
        Server responds: {"type": "response", "text": "...", "audio_chunk": base64}
    """
    await websocket.accept()
    
    try:
        llm_instance = get_llm()
        _, streaming_tts_instance = get_tts()
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                text = data.get("text", "")
                
                # Generate response
                response_text = llm_instance.generate_response(text)
                
                # Send text response immediately
                await websocket.send_json({
                    "type": "response",
                    "text": response_text
                })
                
                # Stream audio chunks
                for chunk in streaming_tts_instance.synthesize_stream_sentences(response_text):
                    buffer = io.BytesIO()
                    sf.write(buffer, chunk, 24000, format='WAV')
                    audio_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": audio_b64
                    })
                
                # Signal completion
                await websocket.send_json({"type": "complete"})
                
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


# Mount static files for web client
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FERN Web Server...")
    print("   API docs: http://localhost:8000/docs")
    print("   Web client: http://localhost:8000")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

