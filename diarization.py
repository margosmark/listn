"""
Speaker diarization module using pyannote.audio.

Identifies different speakers in audio and assigns speaker labels.
Requires HuggingFace token for pretrained models.
"""

import sys
import warnings
from typing import Optional, List, Tuple
import numpy as np
from pathlib import Path

# Suppress torchcodec warning - we use in-memory audio, not file loading
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)


# Cache for loaded pipeline
_pipeline_cache: Optional[object] = None


def load_diarization_pipeline(hf_token: Optional[str] = None):
    """
    Load the pyannote speaker diarization pipeline.

    Args:
        hf_token: HuggingFace token for accessing pretrained models.
                 Can also be set via HF_TOKEN environment variable.

    Returns:
        Loaded diarization pipeline

    Raises:
        ImportError: If pyannote.audio is not installed
        ValueError: If HuggingFace token is not provided

    Example:
        >>> pipeline = load_diarization_pipeline("your_hf_token")
        >>> # Pipeline is cached for subsequent calls
    """
    global _pipeline_cache

    if _pipeline_cache is not None:
        return _pipeline_cache

    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        print("\n[Error] pyannote.audio not installed", file=sys.stderr)
        print("\nInstall with:", file=sys.stderr)
        print("  pip install pyannote.audio", file=sys.stderr)
        print("\nYou'll also need a HuggingFace token:", file=sys.stderr)
        print("  1. Create account at https://huggingface.co", file=sys.stderr)
        print("  2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1", file=sys.stderr)
        print("  3. Get token from https://huggingface.co/settings/tokens", file=sys.stderr)
        raise ImportError("pyannote.audio not installed")

    if hf_token is None:
        import os
        hf_token = os.getenv("HF_TOKEN")
        if hf_token is None:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN environment variable or pass hf_token parameter.\n"
                "Get token from: https://huggingface.co/settings/tokens"
            )

    print("[Diarization] Loading speaker diarization model...")
    print("[Diarization] This may take a minute on first run...")

    try:
        # Load the pretrained pipeline
        # Note: parameter name changed from use_auth_token to token in newer versions
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        except TypeError:
            # Fallback for older versions
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )

        # Use GPU if available (MPS on M1/M2, CUDA on NVIDIA)
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pipeline = pipeline.to(torch.device("mps"))

        _pipeline_cache = pipeline
        print("[Diarization] Model loaded successfully")
        return pipeline

    except Exception as e:
        print(f"\n[Error] Failed to load diarization model: {e}", file=sys.stderr)
        print("\nMake sure you:", file=sys.stderr)
        print("1. Have accepted the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1", file=sys.stderr)
        print("2. Have a valid HuggingFace token", file=sys.stderr)
        raise


def identify_speakers(
    pipeline,
    audio_chunk: np.ndarray,
    sample_rate: int = 16000
) -> List[Tuple[float, float, str]]:
    """
    Identify speakers in an audio chunk.

    Args:
        pipeline: Loaded diarization pipeline
        audio_chunk: Audio data as numpy array (float32, mono)
        sample_rate: Sample rate of the audio (default: 16000)

    Returns:
        List of tuples (start_time, end_time, speaker_label)
        Times are in seconds relative to chunk start.

    Example:
        >>> pipeline = load_diarization_pipeline("token")
        >>> speakers = identify_speakers(pipeline, audio_chunk)
        >>> for start, end, speaker in speakers:
        ...     print(f"{speaker}: {start:.2f}s - {end:.2f}s")
        SPEAKER_00: 0.50s - 2.30s
        SPEAKER_01: 2.50s - 4.20s
    """
    try:
        import torch

        # Debug: Check incoming audio
        # print(f"[Debug] Audio chunk shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}", file=sys.stderr)

        # Ensure audio is numpy array and correct dtype
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk)

        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # Ensure correct shape (channel, time) - pyannote expects channel first
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        elif audio_tensor.ndim == 2 and audio_tensor.shape[0] > audio_tensor.shape[1]:
            # If shape is (samples, channels) instead of (channels, samples), transpose
            audio_tensor = audio_tensor.T

        # Debug output
        # print(f"[Debug] Tensor shape: {audio_tensor.shape}, dtype: {audio_tensor.dtype}, is_tensor: {isinstance(audio_tensor, torch.Tensor)}", file=sys.stderr)

        # Create input dict with waveform and sample rate
        audio_dict = {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }

        # Run diarization directly on the tensor
        diarization_output = pipeline(audio_dict)

        # Extract speaker segments
        segments = []

        # Handle new API (DiarizeOutput object with speaker_diarization attribute)
        if hasattr(diarization_output, 'speaker_diarization'):
            annotation = diarization_output.speaker_diarization
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append((turn.start, turn.end, speaker))
        # Handle old API (direct Annotation object)
        elif hasattr(diarization_output, 'itertracks'):
            for turn, _, speaker in diarization_output.itertracks(yield_label=True):
                segments.append((turn.start, turn.end, speaker))

        return segments

    except Exception as e:
        print(f"[Warning] Speaker diarization failed: {e}", file=sys.stderr)
        return []


def get_speaker_at_time(
    segments: List[Tuple[float, float, str]],
    time: float
) -> Optional[str]:
    """
    Get the speaker label at a specific time.

    Args:
        segments: List of speaker segments from identify_speakers()
        time: Time in seconds to query

    Returns:
        Speaker label or None if no speaker at that time

    Example:
        >>> segments = [(0.0, 2.0, "SPEAKER_00"), (2.5, 4.0, "SPEAKER_01")]
        >>> get_speaker_at_time(segments, 1.5)
        'SPEAKER_00'
        >>> get_speaker_at_time(segments, 3.0)
        'SPEAKER_01'
    """
    for start, end, speaker in segments:
        if start <= time <= end:
            return speaker
    return None
