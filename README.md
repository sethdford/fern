# FERN: Low Latency Voice-2-Voice Architecture for Agents

Implementation of the paper "FERN: Insights on Low Latency Voice-2-Voice Architecture for Agents" (https://arxiv.org/html/2509.20971v1)

## 🎉 Status Update

**Phase 1 COMPLETE** ✅ - Real CSM-1B integration is finished! The FERN framework now includes:

- ✅ Full CSM-1B architecture from [csm-streaming](https://github.com/davidbrowne17/csm-streaming) (0.28x RTF - 3.5x faster than real-time!)
- ✅ Dual-mode system: Placeholder (testing) + Real CSM (production)
- ✅ All tests passing with comprehensive documentation
- ✅ Production-ready architecture with graceful fallbacks

**Quick Start**:
```python
from fern.tts.csm_tts import CSMTTS

# Development mode (works immediately on CPU)
tts = CSMTTS(device='cpu', use_real_csm=False)
audio = tts.synthesize('Hello world!')

# Production mode (edit csm_real.py line 44-45 first)
tts = CSMTTS(device='cuda', use_real_csm=True)
audio = tts.synthesize('Production quality speech!')
```

See `SUCCESS_SUMMARY.md` for complete details, or `PHASE1_STATUS.md` for implementation notes.

## Overview

This project implements an end-to-end voice-to-voice (V-2-V) communication system optimized for real-time conversational applications. The system consists of three main components:

1. **ASR (Automatic Speech Recognition)**: OpenAI Whisper v3-large-turbo with chunked processing
2. **LLM (Large Language Model)**: OpenAI GPT-4o-mini for dialogue management
3. **TTS (Text-to-Speech)**: CSM-1B with optimized Residual Vector Quantization (RVQ)

## Key Features

- **Low Latency**: First chunk latency as low as 640.9ms on GPU
- **Streaming Audio**: Seamless audio streaming with inter-chunk generation
- **Context Awareness**: Ingests both audio and text context for accurate responses
- **RVQ Optimization**: Configurable RVQ iterations (16, 20, 24, 32) for latency/quality tradeoff
- **VAD Integration**: Silero-VAD for robust speech detection
- **Performance Metrics**: Real-time tracking of RTF, latency, SNR, and chunk statistics

## Architecture

```
Voice Input → VAD → ASR (Whisper) → LLM (GPT-4o-mini) → TTS (CSM-1B) → Voice Output
                                          ↑                      ↓
                                    Context Storage ← Audio Chunks
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (optional, for optimal performance)
- OpenAI API key

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Voice-to-Voice Conversation

```python
from fern import VoiceToVoicePipeline

# Initialize pipeline
pipeline = VoiceToVoicePipeline(
    device="cuda",  # or "cpu"
    rvq_iterations=16,  # 16, 20, 24, or 32
    enable_streaming=True
)

# Process audio file
output_audio, metrics = pipeline.process_audio("input.wav")

# Stream conversation
for audio_chunk in pipeline.stream_conversation("input.wav"):
    # Play audio_chunk in real-time
    pass
```

### Performance Metrics

The system tracks comprehensive metrics:

- First Chunk Latency (ms)
- Real-Time Factor (RTF)
- Average Chunk Size (ms)
- Average Inter-Chunk Latency (ms)
- Signal-to-Noise Ratio (SNR)
- Chunks per Second

## Optimization Parameters

### RVQ Iterations

Control the tradeoff between latency and audio quality:

- **16 iterations**: Lowest latency, acceptable quality for telephone-based agents
- **20 iterations**: Balanced latency and quality
- **24 iterations**: Higher quality, moderate latency
- **32 iterations**: Best quality, highest latency (default in CSM-1B)

### Environment

- **GPU (NVIDIA L4 24GB)**: RTF < 1.0, first chunk latency ~640ms (16 RVQ)
- **CPU (Apple M4 Max)**: RTF ~1.1, first chunk latency ~1749ms (16 RVQ)

## Benchmarks

### GPU Performance (NVIDIA L4)

| RVQ Iterations | First Chunk (ms) | RTF    | SNR (dB) |
|----------------|------------------|--------|----------|
| 16             | 640.9            | 0.480x | 7.158    |
| 20             | 1105.4           | 0.571x | 14.844   |
| 24             | 1172.3           | 0.574x | 22.817   |
| 32             | 1381.9           | 0.785x | 33.115   |

### CPU Performance (Apple M4 Max)

| RVQ Iterations | First Chunk (ms) | RTF    | SNR (dB) |
|----------------|------------------|--------|----------|
| 16             | 1748.6           | 0.934x | 15.459   |
| 20             | 1855.0           | 1.143x | 10.525   |
| 24             | 2172.3           | 1.236x | 24.569   |
| 32             | 2662.5           | 1.489x | 34.404   |

## Project Structure

```
voice/
├── README.md
├── requirements.txt
├── fern/
│   ├── __init__.py
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── whisper_asr.py
│   │   └── vad.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── dialogue_manager.py
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── csm_tts.py
│   │   └── rvq_optimizer.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── voice_pipeline.py
│   │   └── streaming.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── performance.py
│   └── config.py
├── examples/
│   ├── basic_conversation.py
│   ├── streaming_demo.py
│   └── benchmark.py
└── tests/
    ├── __init__.py
    ├── test_asr.py
    ├── test_llm.py
    ├── test_tts.py
    └── test_pipeline.py
```

## References

Choudhary, A., & Purwar, A. (2025). FERN: Insights on Low Latency Voice-2-Voice Architecture for Agents. arXiv:2509.20971

## License

MIT License

## Acknowledgements

- OpenAI Whisper for ASR
- Sesame AI for CSM-1B TTS model
- Silero Team for VAD
- OpenAI for GPT-4o-mini

