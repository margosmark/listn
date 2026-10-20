#!/usr/bin/env python3
"""
Record audio from a device to a WAV file.

Simple script to capture system audio (via BlackHole) and save it for
later transcription with process_audio.py.

Usage:
    python record_audio.py --device 2 --output meeting.wav
    # Press Ctrl+C to stop recording
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import sounddevice as sd
import wave
from typing import Optional

from audio import list_audio_devices


class AudioRecorder:
    """Records audio from a device to a WAV file."""

    def __init__(
        self,
        device_id: int,
        output_path: Path,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        """
        Initialize audio recorder.

        Args:
            device_id: Audio input device ID
            output_path: Path to output WAV file
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
        """
        self.device_id = device_id
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = []
        self.is_recording = False

    def audio_callback(self, indata, frames, time, status):
        """Called for each audio block from the device."""
        if status:
            print(f"[Warning] {status}", file=sys.stderr)

        # Store audio data
        self.frames.append(indata.copy())

    def start_recording(self):
        """Start recording audio."""
        print(f"[Recording] Saving to: {self.output_path}")
        print(f"[Recording] Sample rate: {self.sample_rate} Hz")

        # Auto-detect channels from device
        try:
            device_info = sd.query_devices(self.device_id)
            max_channels = device_info.get('max_input_channels', 0)

            if max_channels == 0:
                raise ValueError(f"Device '{device_info['name']}' is not an input device")

            # Use device's native channel count (handles aggregate devices correctly)
            if self.channels != max_channels:
                print(f"[Recording] Device has {max_channels} input channels")
                if self.channels > max_channels:
                    print(f"[Recording] Requested {self.channels} channels, using {max_channels} instead")
                    self.channels = max_channels
                elif max_channels > 1:
                    print(f"[Recording] Recording {max_channels} channels, will convert to mono when saving")
                    self.channels = max_channels
        except Exception as e:
            print(f"[Error] Could not query device: {e}")
            raise

        print(f"[Recording] Channels: {self.channels}")
        print()
        print("Press Ctrl+C to stop recording")
        print("=" * 50)
        print()

        self.is_recording = True
        self.frames = []

        try:
            # Open audio stream
            with sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                dtype=np.float32
            ):
                # Show recording status
                start_time = datetime.now()

                while self.is_recording:
                    # Calculate duration
                    duration = (datetime.now() - start_time).total_seconds()
                    mins = int(duration // 60)
                    secs = int(duration % 60)

                    # Show recording indicator
                    print(f"\r🔴 Recording: {mins:02d}:{secs:02d}", end="", flush=True)

                    # Sleep briefly
                    sd.sleep(1000)  # 1 second

        except KeyboardInterrupt:
            print("\n\n[Stopping] Recording stopped by user")
            self.is_recording = False

        # Save the recorded audio
        self._save_to_wav()

    def _save_to_wav(self):
        """Save recorded audio to WAV file."""
        if not self.frames:
            print("[Error] No audio data recorded", file=sys.stderr)
            return

        print()
        print("=" * 50)
        print("[Saving] Writing audio to file...")

        # Concatenate all frames
        audio_data = np.concatenate(self.frames, axis=0)

        # Convert stereo to mono if needed (average channels)
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            print(f"[Saving] Converting stereo to mono...")
            audio_data = audio_data.mean(axis=1)
            save_channels = 1
        else:
            save_channels = self.channels

        # Calculate duration
        duration = len(audio_data) / self.sample_rate
        mins = int(duration // 60)
        secs = duration % 60

        print(f"[Saving] Duration: {mins}:{secs:05.2f}")
        print(f"[Saving] Samples: {len(audio_data)}")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert float32 to int16 for WAV file
        # WAV expects int16 in range [-32768, 32767]
        audio_int16 = np.int16(audio_data * 32767)

        # Write WAV file
        with wave.open(str(self.output_path), 'wb') as wf:
            wf.setnchannels(save_channels)
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # Get file size
        file_size = self.output_path.stat().st_size / (1024 * 1024)  # MB

        print(f"[Saving] File size: {file_size:.2f} MB")
        print()
        print("=" * 50)
        print(f"✓ Audio saved to: {self.output_path}")
        print()
        print("Next steps:")
        print(f"  1. Process with speaker diarization:")
        print(f"     ./process_interview.sh {self.output_path} --enable-diarization")
        print()
        print(f"  2. Or transcribe without speakers:")
        print(f"     python process_audio.py {self.output_path} --output transcript.txt")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Record audio from a device to a WAV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available devices
  python record_audio.py --list-devices

  # Record from BlackHole device
  python record_audio.py --device 2 --output meeting.wav

  # Record with custom sample rate
  python record_audio.py --device 2 --output interview.wav --sample-rate 44100

  # Record stereo
  python record_audio.py --device 2 --output stereo.wav --channels 2

After recording, process with:
  ./process_interview.sh meeting.wav --enable-diarization
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
        help="Output WAV file path (default: recording_TIMESTAMP.wav)"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        metavar="HZ",
        help="Sample rate in Hz (default: 16000)"
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of channels: 1 (mono) or 2 (stereo) (default: 1)"
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
        args.output = Path(f"recording_{timestamp}.wav")

    # Check if output file already exists
    if args.output.exists():
        print(f"Warning: Output file already exists: {args.output}", file=sys.stderr)
        response = input("Overwrite? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted")
            return 1

    # Create recorder and start recording
    try:
        recorder = AudioRecorder(
            device_id=args.device,
            output_path=args.output,
            sample_rate=args.sample_rate,
            channels=args.channels
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
