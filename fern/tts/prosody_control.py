#!/usr/bin/env python3
"""
Prosody & Emotion Control (Chatterbox-Inspired, 2025)

Based on: Chatterbox (resemble-ai/chatterbox)
- First open-source TTS with emotion exaggeration control
- Sub-200ms latency
- Natural emphasis, pauses, and emotion

Features:
- Sentiment-aware emotion codes
- Emphasis detection (capitals, important words)
- Natural pause insertion (punctuation-aware)
- Prosody markers for TTS
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """Supported emotion types."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CALM = "calm"


@dataclass
class ProsodyConfig:
    """Configuration for prosody control."""
    enable_emotion: bool = True
    enable_emphasis: bool = True
    enable_pauses: bool = True
    pause_short_ms: int = 100   # For commas, semicolons
    pause_medium_ms: int = 200  # For periods, question marks
    pause_long_ms: int = 300    # For emphasis or dramatic effect
    sentiment_threshold: float = 0.7  # Confidence threshold for emotion
    use_sentiment_model: bool = False  # Set True to use RoBERTa (requires transformers)


class ProsodyController:
    """
    Add natural prosody and emotion to text for TTS.

    Chatterbox-inspired approach:
    - Analyze sentiment for emotion
    - Detect emphasis (caps, exclamation, important words)
    - Insert natural pauses (punctuation-based)
    - Generate prosody markers for TTS

    Usage:
        prosody = ProsodyController()

        # Add prosody to text
        text_with_prosody = prosody.add_prosody(
            "I'm SO excited to share this amazing news!"
        )
        # → "[EXCITED] I'm [EMPHASIS]SO[/EMPHASIS] excited to share this [PAUSE:100ms] amazing news[PAUSE:200ms]!"

        # Then pass to TTS
        audio = tts.synthesize(text_with_prosody)
    """

    def __init__(self, config: Optional[ProsodyConfig] = None):
        """
        Initialize prosody controller.

        Args:
            config: Prosody configuration (or use defaults)
        """
        self.config = config or ProsodyConfig()

        # Load sentiment model if requested
        self.sentiment_analyzer = None
        if self.config.use_sentiment_model:
            try:
                from transformers import pipeline
                logger.info("Loading sentiment analysis model...")
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if self._has_gpu() else -1
                )
                logger.info("✓ Sentiment model loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentiment model: {e}")
                logger.warning("Using rule-based emotion detection instead")

        logger.info("✓ Prosody controller ready")
        logger.info(f"  Emotion: {'enabled' if self.config.enable_emotion else 'disabled'}")
        logger.info(f"  Emphasis: {'enabled' if self.config.enable_emphasis else 'disabled'}")
        logger.info(f"  Pauses: {'enabled' if self.config.enable_pauses else 'disabled'}")

    def add_prosody(
        self,
        text: str,
        force_emotion: Optional[EmotionType] = None
    ) -> str:
        """
        Add prosody markers to text.

        Args:
            text: Input text
            force_emotion: Optionally force a specific emotion

        Returns:
            Text with prosody markers
        """
        # Detect emotion
        if self.config.enable_emotion:
            if force_emotion:
                emotion = force_emotion
            else:
                emotion = self._detect_emotion(text)

            # Prepend emotion marker
            text = f"[{emotion.value.upper()}] {text}"

        # Add emphasis markers
        if self.config.enable_emphasis:
            text = self._add_emphasis(text)

        # Add pause markers
        if self.config.enable_pauses:
            text = self._add_pauses(text)

        return text

    def _detect_emotion(self, text: str) -> EmotionType:
        """
        Detect emotion from text.

        Uses:
        1. Sentiment model (if available)
        2. Rule-based detection (fallback)
        """
        # Method 1: Sentiment model
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text)[0]
                label = result['label'].lower()
                score = result['score']

                if score > self.config.sentiment_threshold:
                    # Map sentiment to emotion
                    sentiment_map = {
                        'positive': EmotionType.HAPPY,
                        'negative': EmotionType.SAD,
                        'neutral': EmotionType.NEUTRAL,
                    }
                    return sentiment_map.get(label, EmotionType.NEUTRAL)

            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")

        # Method 2: Rule-based detection (fallback)
        return self._rule_based_emotion(text)

    def _rule_based_emotion(self, text: str) -> EmotionType:
        """
        Simple rule-based emotion detection.

        Checks for:
        - Multiple exclamations → EXCITED
        - Question marks → NEUTRAL (or SURPRISED if multiple)
        - Caps words → EXCITED or ANGRY
        - Sad words → SAD
        - Happy words → HAPPY
        """
        text_lower = text.lower()

        # Excited: multiple exclamations
        if text.count('!') >= 2:
            return EmotionType.EXCITED

        # Excited/Angry: lots of caps
        caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]
        if len(caps_words) >= 2:
            # Angry if negative words
            angry_words = ['hate', 'stupid', 'terrible', 'worst', 'awful']
            if any(word in text_lower for word in angry_words):
                return EmotionType.ANGRY
            else:
                return EmotionType.EXCITED

        # Surprised: multiple questions
        if text.count('?') >= 2:
            return EmotionType.SURPRISED

        # Sad: sad keywords
        sad_words = ['sad', 'sorry', 'unfortunately', 'terrible', 'bad', 'worse', 'failed']
        if any(word in text_lower for word in sad_words):
            return EmotionType.SAD

        # Happy: happy keywords
        happy_words = ['great', 'amazing', 'awesome', 'wonderful', 'excellent', 'love', 'happy', 'excited']
        if any(word in text_lower for word in happy_words):
            return EmotionType.HAPPY

        # Calm: calm keywords
        calm_words = ['please', 'thank you', 'appreciate', 'understand', 'okay']
        if any(word in text_lower for word in calm_words):
            return EmotionType.CALM

        # Default: neutral
        return EmotionType.NEUTRAL

    def _add_emphasis(self, text: str) -> str:
        """
        Add emphasis markers to important words.

        Emphasizes:
        - All-caps words (LIKE THIS)
        - Words with exclamation (amazing!)
        - Quoted words ("special")
        """
        # Emphasize all-caps words (2+ letters)
        text = re.sub(
            r'\b([A-Z]{2,})\b',
            r'[EMPHASIS]\1[/EMPHASIS]',
            text
        )

        # Emphasize words before exclamation
        text = re.sub(
            r'\b(\w+)(!+)',
            r'[EMPHASIS]\1[/EMPHASIS]\2',
            text
        )

        # Emphasize quoted words
        text = re.sub(
            r'"([^"]+)"',
            r'[EMPHASIS]"\1"[/EMPHASIS]',
            text
        )

        return text

    def _add_pauses(self, text: str) -> str:
        """
        Add natural pause markers.

        Pauses:
        - Short (100ms): Commas, semicolons
        - Medium (200ms): Periods, questions, exclamations
        - Long (300ms): Ellipsis (...), dramatic emphasis
        """
        # Long pause: ellipsis
        text = re.sub(
            r'\.\.\.',
            f'[PAUSE:{self.config.pause_long_ms}ms]',
            text
        )

        # Medium pause: sentence endings
        text = re.sub(
            r'([.!?])',
            rf'\1[PAUSE:{self.config.pause_medium_ms}ms]',
            text
        )

        # Short pause: commas, semicolons
        text = re.sub(
            r'([,;:])',
            rf'\1[PAUSE:{self.config.pause_short_ms}ms]',
            text
        )

        # Long pause before/after emphasis (for dramatic effect)
        text = re.sub(
            r'\[EMPHASIS\]',
            f'[PAUSE:{self.config.pause_short_ms}ms][EMPHASIS]',
            text
        )

        return text

    def parse_prosody_markers(self, text: str) -> List[Tuple[str, str, Optional[Dict]]]:
        """
        Parse prosody markers from text.

        Returns list of (type, content, params):
        - ('text', 'Hello', None)
        - ('emotion', 'HAPPY', None)
        - ('emphasis', 'amazing', None)
        - ('pause', None, {'duration_ms': 200})

        Useful for TTS systems that need structured prosody info.
        """
        segments = []
        current_pos = 0

        # Regex to match all markers
        marker_pattern = r'\[(HAPPY|SAD|EXCITED|ANGRY|SURPRISED|CALM|NEUTRAL|EMPHASIS|/EMPHASIS|PAUSE:\d+ms)\]'

        for match in re.finditer(marker_pattern, text):
            # Add text before marker
            if match.start() > current_pos:
                text_segment = text[current_pos:match.start()]
                if text_segment.strip():
                    segments.append(('text', text_segment, None))

            # Parse marker
            marker = match.group(1)

            if marker in ['HAPPY', 'SAD', 'EXCITED', 'ANGRY', 'SURPRISED', 'CALM', 'NEUTRAL']:
                segments.append(('emotion', marker, None))

            elif marker == 'EMPHASIS':
                segments.append(('emphasis_start', None, None))

            elif marker == '/EMPHASIS':
                segments.append(('emphasis_end', None, None))

            elif marker.startswith('PAUSE:'):
                duration_ms = int(marker.split(':')[1].replace('ms', ''))
                segments.append(('pause', None, {'duration_ms': duration_ms}))

            current_pos = match.end()

        # Add remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            if remaining_text.strip():
                segments.append(('text', remaining_text, None))

        return segments

    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        except:
            return False


class EmotionIntensityController:
    """
    Control emotion intensity (Chatterbox feature).

    Allows exaggerating or dampening emotions:
    - intensity = 0.0: No emotion (monotone)
    - intensity = 1.0: Normal emotion
    - intensity = 2.0: Exaggerated emotion (2x)
    """

    def __init__(self, default_intensity: float = 1.0):
        """
        Initialize emotion intensity controller.

        Args:
            default_intensity: Default intensity level (0.0 to 2.0)
        """
        self.default_intensity = default_intensity

    def adjust_emotion(
        self,
        text: str,
        intensity: Optional[float] = None
    ) -> str:
        """
        Adjust emotion intensity in prosody-marked text.

        Args:
            text: Text with prosody markers
            intensity: Intensity multiplier (0.0 to 2.0)

        Returns:
            Text with adjusted emotion intensity markers
        """
        if intensity is None:
            intensity = self.default_intensity

        # If intensity is 0, remove all emotion markers
        if intensity == 0.0:
            text = re.sub(r'\[(HAPPY|SAD|EXCITED|ANGRY|SURPRISED|CALM)\]', '[NEUTRAL]', text)
            return text

        # If intensity > 1.0, add intensity marker
        if intensity > 1.0:
            # Add intensity parameter to emotion markers
            text = re.sub(
                r'\[(HAPPY|SAD|EXCITED|ANGRY|SURPRISED|CALM)\]',
                rf'[\1:INTENSITY={intensity:.1f}]',
                text
            )

        return text


# Convenience factory function
def create_prosody_controller(
    use_sentiment_model: bool = False,
    enable_all: bool = True
) -> ProsodyController:
    """
    Create a prosody controller with sensible defaults.

    Args:
        use_sentiment_model: Use RoBERTa for sentiment (requires GPU)
        enable_all: Enable all prosody features

    Returns:
        Configured ProsodyController
    """
    config = ProsodyConfig(
        enable_emotion=enable_all,
        enable_emphasis=enable_all,
        enable_pauses=enable_all,
        use_sentiment_model=use_sentiment_model,
        sentiment_threshold=0.7,
        pause_short_ms=100,
        pause_medium_ms=200,
        pause_long_ms=300,
    )

    return ProsodyController(config)


if __name__ == "__main__":
    # Test prosody controller
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("  Prosody & Emotion Control Test")
    print("=" * 70)
    print()

    # Create controller
    prosody = create_prosody_controller(use_sentiment_model=False)

    # Test cases
    test_texts = [
        "I'm SO excited to share this amazing news!",
        "Unfortunately, the project failed.",
        "Can you help me with this, please?",
        "STOP! That's dangerous!",
        "Hello, how are you today?",
        "Wow... that's incredible!",
        "I absolutely LOVE this new feature!",
    ]

    print("Testing prosody markers...")
    print()

    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: \"{text}\"")

        # Add prosody
        prosody_text = prosody.add_prosody(text)
        print(f"  → {prosody_text}")

        # Parse markers
        segments = prosody.parse_prosody_markers(prosody_text)
        print(f"  → Segments: {len(segments)}")

        # Show emotion detected
        for seg_type, content, params in segments:
            if seg_type == 'emotion':
                print(f"     Emotion: {content}")
                break

        print()

    print("=" * 70)
    print("Test complete!")
    print("=" * 70)
    print()
    print("Prosody markers added:")
    print("  ✓ Emotion codes (HAPPY, SAD, EXCITED, etc.)")
    print("  ✓ Emphasis markers ([EMPHASIS]word[/EMPHASIS])")
    print("  ✓ Pause markers ([PAUSE:200ms])")
    print()
    print("Usage with TTS:")
    print("  prosody_text = prosody.add_prosody('Hello world!')")
    print("  audio = tts.synthesize(prosody_text)")
    print()
