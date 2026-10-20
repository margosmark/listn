#!/usr/bin/env python3
"""
Post-process audio files for transcription with speaker diarization.

This script processes complete audio files (not real-time) for better
speaker separation and transcription quality. Use this after recording
interviews or meetings.

Usage:
    python process_audio.py input.wav --output transcript.txt
    python process_audio.py input.mp3 --enable-diarization --output interview.txt
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np

from transcribe import load_model, transcribe_chunk_with_details


def process_audio_file(
    audio_path: Path,
    output_path: Optional[Path] = None,
    model_size: str = "medium",
    language: Optional[str] = None,
    enable_diarization: bool = False,
    hf_token: Optional[str] = None
) -> None:
    """
    Process a complete audio file with transcription and optional diarization.

    Args:
        audio_path: Path to input audio file (wav, mp3, m4a, etc.)
        output_path: Path to output transcript file
        model_size: Whisper model size
        language: Language code (e.g., 'en')
        enable_diarization: Whether to identify speakers
        hf_token: HuggingFace token for diarization
    """
    import librosa

    print(f"[Setup] Loading audio file: {audio_path}")

    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"[Setup] Audio duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    # Load Whisper model
    print(f"[Setup] Loading Whisper '{model_size}' model...")
    whisper_model = load_model(model_size)
    print("[Setup] Model loaded successfully")

    # Load diarization pipeline if enabled
    diarization_pipeline = None
    if enable_diarization:
        try:
            from diarization import load_diarization_pipeline, identify_speakers
            import os

            token = hf_token or os.getenv("HF_TOKEN")
            if not token:
                print("[Warning] HF_TOKEN not set, continuing without diarization", file=sys.stderr)
                enable_diarization = False
            else:
                print("[Setup] Loading speaker diarization model...")
                diarization_pipeline = load_diarization_pipeline(token)
                print("[Setup] Diarization model loaded successfully")
        except Exception as e:
            print(f"[Warning] Could not load diarization: {e}", file=sys.stderr)
            enable_diarization = False

    # Run diarization on full audio first (if enabled)
    speaker_timeline = []
    if enable_diarization and diarization_pipeline:
        print("\n[Diarization] Analyzing speakers in full audio...")
        try:
            import torch

            # Convert to tensor for diarization
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            audio_dict = {
                "waveform": audio_tensor,
                "sample_rate": sr
            }

            diarization_output = diarization_pipeline(audio_dict)

            # Extract all speaker segments
            if hasattr(diarization_output, 'speaker_diarization'):
                annotation = diarization_output.speaker_diarization
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    speaker_timeline.append((turn.start, turn.end, speaker))

            print(f"[Diarization] Found {len(set(s[2] for s in speaker_timeline))} unique speakers")
            print(f"[Diarization] Total segments: {len(speaker_timeline)}")

        except Exception as e:
            print(f"[Warning] Diarization failed: {e}", file=sys.stderr)
            speaker_timeline = []

    # Transcribe audio
    print("\n[Transcription] Processing audio...")
    print("=" * 70)

    result = transcribe_chunk_with_details(
        model=whisper_model,
        audio_chunk=audio,
        language=language,
        verbose=False
    )

    # Open output file if specified
    output_file = None
    if output_path:
        output_file = open(output_path, "w", encoding="utf-8")

    # Process segments with speaker info
    for segment in result["segments"]:
        text = segment["text"].strip()
        if not text:
            continue

        start_time = segment["start"]
        end_time = segment["end"]

        # Find speaker for this segment
        speaker = None
        if speaker_timeline:
            segment_mid = (start_time + end_time) / 2
            for s_start, s_end, s_label in speaker_timeline:
                if s_start <= segment_mid <= s_end:
                    speaker = s_label
                    break

        # Format output
        timestamp = f"[{int(start_time//60):02d}:{int(start_time%60):02d}]"
        if speaker:
            line = f"{timestamp} [{speaker}] {text}"
        else:
            line = f"{timestamp} {text}"

        # Print and write
        print(line)
        if output_file:
            output_file.write(line + "\n")
            output_file.flush()

    if output_file:
        output_file.close()
        print("\n" + "=" * 70)
        print(f"[Done] Transcript saved to: {output_path}")

    print("\n[Complete] Processing finished successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process audio files with transcription and speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python process_audio.py interview.wav

  # With output file
  python process_audio.py interview.wav --output transcript.txt

  # With speaker diarization (best quality)
  python process_audio.py interview.wav --enable-diarization --output interview.txt

  # Specify language and model
  python process_audio.py meeting.mp3 --language en --model large --output meeting.txt

Note: For speaker diarization, set HF_TOKEN environment variable:
  export HF_TOKEN="your_token_here"
        """
    )

    parser.add_argument(
        "audio_file",
        type=Path,
        help="Input audio file (wav, mp3, m4a, etc.)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Output transcript file (optional, defaults to console only)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)"
    )

    parser.add_argument(
        "--language",
        type=str,
        metavar="CODE",
        help="Language code (e.g., 'en', 'es', 'fr'). Auto-detect if not specified."
    )

    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Enable speaker identification (requires HF_TOKEN)"
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        metavar="TOKEN",
        help="HuggingFace token for speaker diarization"
    )

    args = parser.parse_args()

    # Check if audio file exists
    if not args.audio_file.exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        return 1

    # Check for librosa
    try:
        import librosa
    except ImportError:
        print("\nError: librosa is required for audio file processing", file=sys.stderr)
        print("Install with: pip install librosa", file=sys.stderr)
        return 1

    # Process the audio file
    try:
        process_audio_file(
            audio_path=args.audio_file,
            output_path=args.output,
            model_size=args.model,
            language=args.language,
            enable_diarization=args.enable_diarization,
            hf_token=args.hf_token
        )
        return 0

    except KeyboardInterrupt:
        print("\n\n[Stopping] Processing interrupted by user")
        return 1

    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
