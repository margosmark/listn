#!/usr/bin/env python3
"""
Record audio to file WHILE showing real-time transcription.

This hybrid approach gives you the best of both worlds:
1. Real-time transcription during the meeting (see it live)
2. Audio file saved for better post-processing later

Usage:
    python record_with_live_transcription.py --device 2
    python record_with_live_transcription.py --device 2 --output interview.wav
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import sounddevice as sd
import wave
import threading
import queue

from audio import list_audio_devices
from transcribe import load_model, transcribe_chunk


class LiveRecorder:
    """Records audio to file while transcribing in real-time."""

    def __init__(
        self,
        device_id: int,
        output_path: Path,
        model_size: str = "base",
        language: Optional[str] = None,
        chunk_duration: float = 5.0,
        sample_rate: int = 16000,
        include_timestamps: bool = True
    ):
        """
        Initialize live recorder.

        Args:
            device_id: Audio input device ID
            output_path: Path to output WAV file
            model_size: Whisper model size (base recommended for real-time)
            language: Language code or None for auto-detect
            chunk_duration: Duration of audio chunks in seconds
            sample_rate: Sample rate in Hz
            include_timestamps: Include timestamps in transcription
        """
        self.device_id = device_id
        self.output_path = output_path
        self.model_size = model_size
        self.language = language
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.include_timestamps = include_timestamps

        self.is_recording = False
        self.all_frames = []  # For saving to WAV
        self.chunk_buffer = []  # For real-time transcription
        self.chunk_samples = int(chunk_duration * sample_rate)

        # Load Whisper model
        print(f"[Setup] Loading Whisper '{model_size}' model...")
        self.model = load_model(model_size)
        print("[Setup] Model loaded\n")

    def audio_callback(self, indata, frames, time, status):
        """Called for each audio block from the device."""
        if status:
            print(f"[Warning] {status}", file=sys.stderr)

        # Store for WAV file
        self.all_frames.append(indata.copy())

        # Add to chunk buffer for transcription
        self.chunk_buffer.append(indata.copy())

    def transcription_worker(self):
        """Background thread that transcribes chunks."""
        while self.is_recording:
            # Check if we have enough audio for a chunk
            if len(self.chunk_buffer) > 0:
                total_samples = sum(len(frame) for frame in self.chunk_buffer)

                if total_samples >= self.chunk_samples:
                    # Get chunk
                    chunk_data = np.concatenate(self.chunk_buffer, axis=0)
                    chunk_to_process = chunk_data[:self.chunk_samples]

                    # Keep remaining samples for next chunk
                    remaining = chunk_data[self.chunk_samples:]
                    self.chunk_buffer = [remaining] if len(remaining) > 0 else []

                    # Flatten if stereo
                    if chunk_to_process.ndim > 1:
                        chunk_to_process = chunk_to_process.mean(axis=1)

                    # Transcribe
                    text = transcribe_chunk(
                        self.model,
                        chunk_to_process.astype(np.float32),
                        language=self.language,
                        verbose=False
                    )

                    # Display if not empty
                    if text:
                        if self.include_timestamps:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp}] {text}")
                        else:
                            print(text)
                        sys.stdout.flush()

            # Sleep briefly
            import time
            time.sleep(0.5)

    def start_recording(self):
        """Start recording and transcribing."""
        print(f"[Recording] Audio file: {self.output_path}")
        print(f"[Recording] Real-time transcription: {self.model_size} model")
        print(f"[Recording] Language: {self.language or 'auto-detect'}")
        print()

        # Query device to get supported channel count
        try:
            device_info = sd.query_devices(self.device_id)

            # Check if this is an input device
            max_input_channels = device_info.get('max_input_channels', 0)

            if max_input_channels == 0:
                raise ValueError(f"Device '{device_info['name']}' is not an input device (no input channels)")

            # For aggregate devices or multi-channel devices, use all available input channels
            # We'll convert to mono during transcription
            channels = max_input_channels

            print(f"[Recording] Device: {device_info['name']}")
            print(f"[Recording] Channels: {channels}")
        except Exception as e:
            print(f"[Error] Could not query device info: {e}")
            print(f"[Error] Make sure device ID {self.device_id} is correct")
            raise

        print("=" * 70)
        print("Press Ctrl+C to stop recording")
        print("=" * 70)
        print()

        self.is_recording = True
        self.all_frames = []
        self.chunk_buffer = []

        # Start transcription thread
        transcription_thread = threading.Thread(target=self.transcription_worker, daemon=True)
        transcription_thread.start()

        try:
            # Open audio stream with detected channel count
            with sd.InputStream(
                device=self.device_id,
                channels=channels,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                dtype=np.float32
            ):
                # Keep recording until interrupted
                while self.is_recording:
                    sd.sleep(1000)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("[Stopping] Recording stopped by user")
            self.is_recording = False

        # Wait for transcription thread to finish
        transcription_thread.join(timeout=2)

        # Save the recorded audio
        self._save_to_wav()

    def _save_to_wav(self):
        """Save recorded audio to WAV file."""
        if not self.all_frames:
            print("[Error] No audio data recorded", file=sys.stderr)
            return

        print()
        print("=" * 70)
        print("[Saving] Writing audio to file...")

        # Concatenate all frames
        audio_data = np.concatenate(self.all_frames, axis=0)

        # Convert stereo to mono if needed (average channels)
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            print(f"[Saving] Converting stereo to mono...")
            audio_data = audio_data.mean(axis=1)

        # Calculate duration
        duration = len(audio_data) / self.sample_rate
        mins = int(duration // 60)
        secs = duration % 60

        print(f"[Saving] Duration: {mins}:{secs:05.2f}")
        print(f"[Saving] Samples: {len(audio_data)}")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert float32 to int16 for WAV file
        audio_int16 = np.int16(audio_data * 32767)

        # Write WAV file (always mono)
        with wave.open(str(self.output_path), 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # Get file size
        file_size = self.output_path.stat().st_size / (1024 * 1024)  # MB

        print(f"[Saving] File size: {file_size:.2f} MB")
        print()
        print("=" * 70)
        print(f"✓ Audio saved to: {self.output_path}")
        print()
        print("Next: Post-process for better speaker separation:")
        print(f"  ./process_interview.sh {self.output_path} --enable-diarization --model medium")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Record audio with real-time transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script records audio to a file WHILE showing real-time transcription.

Benefits:
- See transcription live during the meeting
- Have audio file for better post-processing later
- Best of both worlds!

Examples:
  # List devices
  python record_with_live_transcription.py --list-devices

  # Record with default name (meeting_TIMESTAMP.wav)
  python record_with_live_transcription.py --device 2

  # Record with custom name
  python record_with_live_transcription.py --device 2 --output interview.wav

  # Specify language for faster transcription
  python record_with_live_transcription.py --device 2 --language en --output meeting.wav

After recording, post-process for better speaker separation:
  ./process_interview.sh meeting.wav --enable-diarization --model medium
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
        help="Audio device ID to record from (required unless --list-devices)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Output WAV file path (default: meeting_TIMESTAMP.wav)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size for real-time (default: base - fast and accurate)"
    )

    parser.add_argument(
        "--language",
        type=str,
        metavar="CODE",
        help="Language code (e.g., 'en', 'es', 'fr') for faster processing"
    )

    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Audio chunk duration in seconds (default: 5.0)"
    )

    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Don't include timestamps in real-time transcription"
    )

    args = parser.parse_args()

    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return 0

    # Validate device ID
    if args.device is None:
        print("Error: --device is required (use --list-devices to see options)", file=sys.stderr)
        return 1

    # Generate default output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"meeting_{timestamp}.wav")

    # Check if output file already exists
    if args.output.exists():
        print(f"Warning: Output file already exists: {args.output}", file=sys.stderr)
        response = input("Overwrite? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted")
            return 1

    # Create recorder and start
    try:
        recorder = LiveRecorder(
            device_id=args.device,
            output_path=args.output,
            model_size=args.model,
            language=args.language,
            chunk_duration=args.chunk_duration,
            include_timestamps=not args.no_timestamps
        )

        recorder.start_recording()
        return 0

    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
