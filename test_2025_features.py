#!/usr/bin/env python3
"""
Test Suite for 2025 Voice Agent Improvements

Tests:
1. ConvFill Turn Detection
2. VoXtream Streaming TTS
3. Prosody Control
4. End-to-end integration

Usage:
    python test_2025_features.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

# Test utilities
def print_test_header(name):
    print("\n" + "=" * 70)
    print(f"  {name}")
    print("=" * 70)

def print_result(name, passed, latency_ms=None):
    status = "✓ PASS" if passed else "✗ FAIL"
    latency_str = f" ({latency_ms:.0f}ms)" if latency_ms else ""
    print(f"  {status}: {name}{latency_str}")


# Test 1: ConvFill Turn Detection
def test_convfill_turn_detection():
    print_test_header("Test 1: ConvFill Turn Detection")

    try:
        from fern.asr.convfill_turn import create_turn_detector

        # Detect device
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  Device: {device}\n")

        # Create detector
        detector = create_turn_detector(device=device, fast_mode=True)

        # Test cases
        test_cases = [
            {
                "text": "I think we should um...",
                "vad_silence": True,
                "expected": False,  # Incomplete (user hesitating)
                "name": "Hesitation detection"
            },
            {
                "text": "That sounds great!",
                "vad_silence": True,
                "expected": True,  # Complete
                "name": "Complete sentence"
            },
            {
                "text": "Can you help me with",
                "vad_silence": True,
                "expected": False,  # Incomplete
                "name": "Incomplete sentence"
            },
            {
                "text": "Yes, I understand.",
                "vad_silence": True,
                "expected": True,  # Complete
                "name": "Clear affirmation"
            },
            {
                "text": "Hello",
                "vad_silence": False,
                "expected": False,  # No VAD silence
                "name": "VAD gate (no silence)"
            },
        ]

        conversation_history = [
            "user: Hello!",
            "assistant: Hi! How can I help you today?",
        ]

        all_passed = True
        total_latency = 0

        for i, test in enumerate(test_cases, 1):
            result = detector.detect_turn_end(
                user_text=test['text'],
                vad_silence=test['vad_silence'],
                conversation_history=conversation_history
            )

            passed = result.is_complete == test['expected']
            print_result(
                test['name'],
                passed,
                result.latency_ms
            )

            if not passed:
                all_passed = False
                print(f"    Expected: {test['expected']}, Got: {result.is_complete}")
                print(f"    Confidence: {result.confidence:.2f}, Method: {result.method}")

            total_latency += result.latency_ms

        avg_latency = total_latency / len(test_cases)
        print(f"\n  Average latency: {avg_latency:.0f}ms")
        print(f"  Target: < 200ms (ConvFill)")

        latency_ok = avg_latency < 200
        print_result("Average latency check", latency_ok, avg_latency)

        return all_passed and latency_ok

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 2: VoXtream Streaming TTS
def test_voxtream_streaming():
    print_test_header("Test 2: VoXtream Streaming TTS")

    try:
        from fern.tts.voxtream_streaming import create_streaming_tts

        # Mock CSM model
        class MockCSM:
            def __init__(self):
                self.generator = lambda x: x

            def synthesize(self, text, speaker_embedding=None):
                # Mock: return audio proportional to text length
                duration_s = len(text.split()) * 0.3  # ~0.3s per word
                sample_rate = 24000
                samples = int(duration_s * sample_rate)
                return np.random.randn(samples).astype(np.float32) * 0.1

        mock_csm = MockCSM()

        # Create streaming TTS
        streaming = create_streaming_tts(mock_csm, mode="voxtream", device="cpu")

        test_text = "Hello world, this is a test of streaming synthesis!"
        print(f"  Text: \"{test_text}\"\n")

        chunk_count = 0
        start_time = time.time()
        initial_delay = None

        for audio_chunk in streaming.stream_audio(test_text):
            chunk_count += 1

            if chunk_count == 1:
                initial_delay = (time.time() - start_time) * 1000
                print_result("First chunk latency", initial_delay < 200, initial_delay)

        total_time = (time.time() - start_time) * 1000

        print(f"\n  Total chunks: {chunk_count}")
        print(f"  Total time: {total_time:.0f}ms")

        # Check streaming worked
        streaming_ok = chunk_count > 1
        print_result("Streaming (multiple chunks)", streaming_ok)

        # Check initial delay
        delay_ok = initial_delay and initial_delay < 200
        print_result("Initial delay < 200ms", delay_ok)

        return streaming_ok and delay_ok

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 3: Prosody Control
def test_prosody_control():
    print_test_header("Test 3: Prosody Control")

    try:
        from fern.tts.prosody_control import create_prosody_controller

        # Create controller
        prosody = create_prosody_controller(use_sentiment_model=False)

        test_cases = [
            {
                "text": "I'm SO excited!",
                "expected_emotion": "EXCITED",
                "should_have_emphasis": True,
                "should_have_pause": True,
                "name": "Excited with emphasis"
            },
            {
                "text": "Unfortunately, the project failed.",
                "expected_emotion": "SAD",
                "should_have_pause": True,
                "name": "Sad news"
            },
            {
                "text": "Hello, how are you?",
                "expected_emotion": "NEUTRAL",
                "should_have_pause": True,
                "name": "Neutral greeting"
            },
            {
                "text": "STOP! That's dangerous!",
                "expected_emotion": "ANGRY",
                "should_have_emphasis": True,
                "should_have_pause": True,
                "name": "Angry warning"
            },
        ]

        all_passed = True

        for i, test in enumerate(test_cases, 1):
            prosody_text = prosody.add_prosody(test['text'])

            # Check emotion
            has_emotion = test['expected_emotion'] in prosody_text
            if not has_emotion:
                all_passed = False

            # Check emphasis
            if test.get('should_have_emphasis'):
                has_emphasis = '[EMPHASIS]' in prosody_text
                if not has_emphasis:
                    all_passed = False

            # Check pauses
            if test.get('should_have_pause'):
                has_pause = '[PAUSE:' in prosody_text
                if not has_pause:
                    all_passed = False

            # Print result
            checks_passed = (
                has_emotion and
                (not test.get('should_have_emphasis') or '[EMPHASIS]' in prosody_text) and
                (not test.get('should_have_pause') or '[PAUSE:' in prosody_text)
            )

            print_result(test['name'], checks_passed)

            if not checks_passed:
                print(f"    Input: {test['text']}")
                print(f"    Output: {prosody_text}")

        return all_passed

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 4: Integration Test
def test_integration():
    print_test_header("Test 4: End-to-End Integration")

    print("  Checking imports...")

    try:
        from fern.asr.convfill_turn import create_turn_detector
        print("    ✓ ConvFill turn detection")

        from fern.tts.voxtream_streaming import create_streaming_tts
        print("    ✓ VoXtream streaming TTS")

        from fern.tts.prosody_control import create_prosody_controller
        print("    ✓ Prosody control")

        print("\n  All imports successful!")
        return True

    except ImportError as e:
        print(f"    ✗ Import failed: {e}")
        return False


# Main test runner
def main():
    print("\n" + "=" * 70)
    print("  FERN 2025 Features Test Suite")
    print("=" * 70)
    print("\n  Testing 2025 voice agent improvements:")
    print("    • ConvFill Turn Detection (Nov 2025)")
    print("    • VoXtream Streaming TTS (Sept 2025)")
    print("    • Prosody & Emotion Control (2025)")
    print()

    results = {}

    # Run tests
    results['integration'] = test_integration()
    results['turn_detection'] = test_convfill_turn_detection()
    results['streaming'] = test_voxtream_streaming()
    results['prosody'] = test_prosody_control()

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results.values() if r)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name.replace('_', ' ').title()}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print("\n  ⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
