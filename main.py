#!/usr/bin/env python3
"""
listn - Real-time audio transcription CLI for macOS

Captures system audio (Zoom, speakers, etc.) and transcribes continuously using Whisper.
Requires BlackHole virtual audio device for system audio capture.

Usage:
    python main.py --list-devices              # List available audio devices
    python main.py --device 2                  # Transcribe from device 2
    python main.py --device 2 --output log.txt # Save to file
    python main.py --device 2 --model base     # Use different model size
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import our modules
from audio import list_devices, capture_audio, get_device_name
from transcribe import load_model, transcribe_chunk, transcribe_chunk_with_details


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configure command-line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription for macOS using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available audio devices
  python main.py --list-devices

  # Transcribe from device 2 (output to console only)
  python main.py --device 2

  # Transcribe and save to file
  python main.py --device 2 --output transcription.txt

  # Use a different Whisper model
  python main.py --device 2 --model base

  # Specify language (faster than auto-detection)
  python main.py --device 2 --language en

Setup:
  1. Install BlackHole: brew install blackhole-2ch
  2. Create Multi-Output Device in Audio MIDI Setup
  3. Set Multi-Output Device as system audio output
  4. Use --list-devices to find BlackHole device ID

For detailed setup instructions, see README.md
        """
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all available audio input devices and exit"
    )

    parser.add_argument(
        "--device",
        type=int,
        metavar="ID",
        help="Audio device ID to capture from (use --list-devices to find)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Output file path (appends if exists). If not specified, console only."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small). Larger = more accurate but slower."
    )

    parser.add_argument(
        "--language",
        type=str,
        metavar="CODE",
        help="Language code (e.g., 'en', 'es', 'fr'). Auto-detect if not specified."
    )

    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Audio chunk duration in seconds (default: 5.0)"
    )

    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Don't include timestamps in output"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed Whisper transcription output"
    )

    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Enable speaker identification (requires HF_TOKEN env variable)"
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        metavar="TOKEN",
        help="HuggingFace token for speaker diarization (or set HF_TOKEN env variable)"
    )

    return parser


def format_output(text: str, include_timestamp: bool = True, speaker: Optional[str] = None) -> str:
    """
    Format transcription output with optional timestamp and speaker label.

    Args:
        text: Transcribed text
        include_timestamp: Whether to include timestamp prefix
        speaker: Speaker label (e.g., "SPEAKER_00")

    Returns:
        Formatted output string
    """
    parts = []

    if include_timestamp:
        timestamp = datetime.now().strftime("%H:%M:%S")
        parts.append(f"[{timestamp}]")

    if speaker:
        parts.append(f"[{speaker}]")

    parts.append(text)

    return " ".join(parts)


def main() -> int:
    """Main entry point for the CLI."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Handle --list-devices
    if args.list_devices:
        list_devices()
        return 0

    # Require --device if not listing devices
    if args.device is None:
        parser.print_help()
        print("\nError: --device is required (use --list-devices to find device ID)", file=sys.stderr)
        return 1

    output_file = None
    diarization_pipeline = None

    try:
        # Load Whisper model (cached after first load)
        print(f"[Setup] Loading Whisper '{args.model}' model...")
        model = load_model(args.model)

        # Load diarization pipeline if enabled
        if args.enable_diarization:
            try:
                from diarization import load_diarization_pipeline, identify_speakers
                import os

                hf_token = args.hf_token or os.getenv("HF_TOKEN")
                diarization_pipeline = load_diarization_pipeline(hf_token)
                print(f"[Setup] Speaker diarization enabled")
            except ImportError as e:
                print(f"\n[Warning] Could not enable diarization: {e}", file=sys.stderr)
                print("[Warning] Continuing without speaker identification", file=sys.stderr)
                args.enable_diarization = False
            except Exception as e:
                print(f"\n[Warning] Failed to load diarization: {e}", file=sys.stderr)
                print("[Warning] Continuing without speaker identification", file=sys.stderr)
                args.enable_diarization = False

        # Get device name for display
        device_name = get_device_name(args.device)
        print(f"[Setup] Using audio device: {device_name} (ID: {args.device})")

        # Setup output file if specified
        if args.output:
            output_file = open(args.output, "a", encoding="utf-8")
            print(f"[Setup] Output file: {args.output}")
            print(f"[Setup] Appending to file (will create if doesn't exist)")

        print("\n" + "=" * 70)
        print("LISTENING... Press Ctrl+C to stop")
        if args.enable_diarization:
            print("Speaker identification: ENABLED")
        print("=" * 70 + "\n")

        # Start audio capture and transcription loop
        try:
            for audio_chunk in capture_audio(
                device_id=args.device,
                chunk_duration=args.chunk_duration,
                sample_rate=16000,
                channels=1
            ):
                # Handle diarization if enabled
                if args.enable_diarization and diarization_pipeline:
                    # Get detailed transcription with segments
                    result = transcribe_chunk_with_details(
                        model=model,
                        audio_chunk=audio_chunk,
                        language=args.language,
                        verbose=args.verbose
                    )

                    text = result["text"].strip()
                    if not text:
                        continue

                    # Get speaker segments for this chunk
                    speaker_segments = identify_speakers(
                        diarization_pipeline,
                        audio_chunk,
                        sample_rate=16000
                    )

                    # If we have speaker info, try to match transcription segments to speakers
                    if speaker_segments and "segments" in result and result["segments"]:
                        for segment in result["segments"]:
                            segment_text = segment["text"].strip()
                            if not segment_text:
                                continue

                            # Find speaker for this segment's midpoint
                            segment_midpoint = (segment["start"] + segment["end"]) / 2
                            speaker = None
                            for s_start, s_end, s_label in speaker_segments:
                                if s_start <= segment_midpoint <= s_end:
                                    speaker = s_label
                                    break

                            # Format and output
                            output_line = format_output(
                                segment_text,
                                include_timestamp=not args.no_timestamps,
                                speaker=speaker
                            )
                            print(output_line)

                            if output_file:
                                output_file.write(output_line + "\n")
                                output_file.flush()
                    else:
                        # No speaker segments, just output text
                        output_line = format_output(text, include_timestamp=not args.no_timestamps)
                        print(output_line)

                        if output_file:
                            output_file.write(output_line + "\n")
                            output_file.flush()

                else:
                    # No diarization, use simple transcription
                    text = transcribe_chunk(
                        model=model,
                        audio_chunk=audio_chunk,
                        language=args.language,
                        verbose=args.verbose
                    )

                    # Skip empty transcriptions (silence)
                    if not text:
                        continue

                    # Format output
                    output_line = format_output(text, include_timestamp=not args.no_timestamps)

                    # Print to console
                    print(output_line)

                    # Write to file if specified
                    if output_file:
                        output_file.write(output_line + "\n")
                        output_file.flush()  # Ensure immediate write

        except KeyboardInterrupt:
            print("\n\n[Stopping] Transcription interrupted by user")

        finally:
            # Clean up
            if output_file:
                output_file.close()
                print(f"[Done] Transcription saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
