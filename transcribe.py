"""
Transcription module using OpenAI Whisper.

Handles loading the Whisper model and transcribing audio chunks.
Optimized for M1 Macs with appropriate model settings.
"""

import sys
import os
from typing import Optional
import numpy as np
import whisper
from whisper import Whisper


# Cache for loaded model to avoid reloading
# Stores tuple of (model_size, model) to ensure correct model is returned
_model_cache: Optional[tuple[str, Whisper]] = None

# Common Whisper hallucinations when processing silence
_HALLUCINATION_PHRASES = {
    "you", "thank you", "thanks for watching", "bye", "goodbye",
    ".", "..", "...", "[BLANK_AUDIO]", "(music)", "(buzzing)",
    "you.", "thank you.", "thanks.", "bye.", "mm", "hmm",
    "uh", "um", "ah", "oh"
}


def _has_speech(audio_chunk: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Check if audio chunk contains speech based on energy/amplitude.

    Args:
        audio_chunk: Audio data as numpy array
        threshold: RMS energy threshold (default: 0.01)

    Returns:
        True if audio has sufficient energy, False if likely silence
    """
    # Flatten to 1D if needed
    if audio_chunk.ndim > 1:
        audio_chunk = audio_chunk.flatten()

    # Handle empty arrays
    if audio_chunk.size == 0:
        return False

    # Calculate RMS (root mean square) energy
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    return rms > threshold


def _is_hallucination(text: str) -> bool:
    """
    Check if transcribed text is likely a Whisper hallucination.

    Whisper tends to hallucinate common phrases when processing silence.

    Args:
        text: Transcribed text to check

    Returns:
        True if text is likely a hallucination, False otherwise
    """
    text_lower = text.lower().strip()

    # Empty or very short
    if len(text_lower) <= 2:
        return True

    # Exact match with known hallucinations
    if text_lower in _HALLUCINATION_PHRASES:
        return True

    # Only punctuation
    if all(c in ".,!?;: " for c in text_lower):
        return True

    return False


def load_model(model_size: str = "small", device: Optional[str] = None) -> Whisper:
    """
    Load a Whisper model with caching.

    The model is cached in memory so subsequent calls return the same instance.
    First load downloads the model (~500MB for small) if not already cached locally.

    Args:
        model_size: Size of Whisper model to load. Options:
                   - "tiny": Fastest, least accurate (~1GB RAM)
                   - "base": Fast, good accuracy (~1GB RAM)
                   - "small": Balanced (default, ~2GB RAM)
                   - "medium": Better accuracy (~5GB RAM)
                   - "large": Best accuracy (~10GB RAM)
        device: Device to run model on. None for auto-detection.
                - None: Auto-detect (uses MPS on M1/M2, CUDA on NVIDIA, CPU otherwise)
                - "cpu": Force CPU
                - "cuda": Force CUDA (NVIDIA GPUs)

    Returns:
        Loaded Whisper model instance

    Raises:
        RuntimeError: If model fails to load

    Example:
        >>> model = load_model("small")
        >>> # Model is now cached, subsequent calls are instant
        >>> model = load_model("small")
    """
    global _model_cache

    # Return cached model if it matches the requested size
    if _model_cache is not None and _model_cache[0] == model_size:
        return _model_cache[1]

    try:
        print(f"[Whisper] Loading '{model_size}' model (this may take a minute on first run)...")

        # Load model with auto device detection
        model = whisper.load_model(model_size, device=device)

        # Cache the model with its size
        _model_cache = (model_size, model)

        print(f"[Whisper] Model loaded successfully")
        return model

    except Exception as e:
        print(f"\n[Error] Failed to load Whisper model: {e}", file=sys.stderr)
        print("\nTroubleshooting tips:", file=sys.stderr)
        print("1. Ensure you have installed: pip install openai-whisper torch", file=sys.stderr)
        print("2. Check your internet connection (first run downloads the model)", file=sys.stderr)
        print("3. Try a smaller model size (e.g., 'tiny' or 'base')", file=sys.stderr)
        raise RuntimeError("Failed to load Whisper model") from e


def transcribe_chunk(
    model: Whisper,
    audio_chunk: np.ndarray,
    language: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Transcribe a single audio chunk using Whisper.

    Args:
        model: Loaded Whisper model (from load_model())
        audio_chunk: Audio data as numpy array (float32, 1D mono)
        language: Language code (e.g., "en", "es", "fr"). None for auto-detection.
        verbose: If True, print detailed transcription info

    Returns:
        Transcribed text (empty string if no speech detected)

    Note:
        This is a convenience wrapper around transcribe_chunk_with_details()
        that returns only the text. Audio normalization (stereo→mono, float32)
        is handled by the audio.capture_audio() function.

    Example:
        >>> model = load_model("small")
        >>> audio = np.array([...])  # Audio data
        >>> text = transcribe_chunk(model, audio)
        >>> print(text)
        "Hello, this is a test."
    """
    # Check if audio chunk has sufficient energy (not silence)
    if not _has_speech(audio_chunk):
        return ""

    result = transcribe_chunk_with_details(model, audio_chunk, language, verbose)
    text = result["text"].strip()

    # Filter out common Whisper hallucinations
    if _is_hallucination(text):
        return ""

    return text


def transcribe_chunk_with_details(
    model: Whisper,
    audio_chunk: np.ndarray,
    language: Optional[str] = None,
    verbose: bool = False
) -> dict:
    """
    Transcribe audio chunk and return detailed results.

    Similar to transcribe_chunk() but returns the full Whisper result dictionary
    including timestamps, language detection, and segments.

    Args:
        model: Loaded Whisper model
        audio_chunk: Audio data as numpy array (float32, 1D mono)
        language: Language code or None for auto-detection
        verbose: If True, print detailed transcription info

    Returns:
        Dictionary with keys:
            - text: Full transcribed text
            - segments: List of segments with timestamps
            - language: Detected/specified language

    Note:
        Audio should already be normalized (mono, float32) by audio.capture_audio().
        We keep defensive checks here for robustness if called from other contexts.

    Example:
        >>> result = transcribe_chunk_with_details(model, audio)
        >>> print(result["text"])
        "Hello world"
        >>> print(result["language"])
        "en"
        >>> for segment in result["segments"]:
        ...     print(f"[{segment['start']:.2f}s] {segment['text']}")
    """
    try:
        # Defensive: ensure audio is 1D (mono) and float32
        # This should already be done by audio.capture_audio(), but we keep
        # this for robustness if called from other contexts
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        audio_chunk = audio_chunk.astype(np.float32)

        # Suppress Whisper's progress bars and detection messages unless verbose
        if not verbose:
            # Redirect stderr to suppress progress bars
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            try:
                # Transcribe with Whisper
                # fp16=False is more compatible with M1 Macs
                result = model.transcribe(
                    audio_chunk,
                    language=language,
                    fp16=False,  # Better for M1 compatibility
                    verbose=None  # None suppresses all output including progress
                )
            finally:
                # Restore stderr
                sys.stderr.close()
                sys.stderr = old_stderr
        else:
            # If verbose mode, show all output
            result = model.transcribe(
                audio_chunk,
                language=language,
                fp16=False,
                verbose=True
            )

        return result

    except Exception as e:
        print(f"[Warning] Transcription failed: {e}", file=sys.stderr)
        return {"text": "", "segments": [], "language": language or "unknown"}
