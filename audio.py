"""
Audio capture module for macOS system audio.

Uses sounddevice to capture audio from BlackHole or other loopback devices.
Provides audio chunks as a generator for continuous transcription.
"""

import sys
from queue import Queue, Empty
from typing import Generator, Optional
import sounddevice as sd
import numpy as np


def list_devices() -> None:
    """
    List all available audio input devices.

    Prints a formatted table of devices with their IDs, names, and channel counts.
    Useful for finding the BlackHole device ID.
    """
    print("\n=== Available Audio Devices ===\n")
    devices = sd.query_devices()

    print(f"{'ID':<4} {'Name':<40} {'Inputs':<8} {'Type':<15}")
    print("-" * 70)

    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            device_type = "Input"
            if "BlackHole" in device['name']:
                device_type = "BlackHole ✓"
            elif "Loopback" in device['name']:
                device_type = "Loopback ✓"

            print(f"{idx:<4} {device['name']:<40} {device['max_input_channels']:<8} {device_type:<15}")

    print("\nTip: Look for 'BlackHole' or 'Loopback' devices for system audio capture.")
    print("If you don't see BlackHole, install it with: brew install blackhole-2ch\n")


# Alias for compatibility with new scripts
list_audio_devices = list_devices


def get_device_name(device_id: int) -> str:
    """
    Get the name of an audio device by its ID.

    Args:
        device_id: The device ID to query

    Returns:
        The device name

    Raises:
        ValueError: If device_id is invalid
    """
    try:
        device = sd.query_devices(device_id)
        return device['name']
    except (ValueError, IndexError):
        raise ValueError(f"Invalid device ID: {device_id}")


def validate_device(device_id: int) -> bool:
    """
    Check if a device ID is valid and supports input.

    Args:
        device_id: The device ID to validate

    Returns:
        True if device is valid and has input channels
    """
    try:
        device = sd.query_devices(device_id)
        return device['max_input_channels'] > 0
    except (ValueError, IndexError):
        return False


def capture_audio(
    device_id: int,
    sample_rate: int = 16000,
    chunk_duration: float = 5.0,
    channels: int = 1
) -> Generator[np.ndarray, None, None]:
    """
    Continuously capture audio from the specified device in chunks.

    This is a generator function that yields numpy arrays of audio data.
    Each chunk contains `chunk_duration` seconds of audio sampled at `sample_rate` Hz.

    Args:
        device_id: The audio device ID to capture from (use list_devices() to find)
        sample_rate: Sample rate in Hz (default: 16000, optimal for Whisper)
        chunk_duration: Duration of each chunk in seconds (default: 5.0)
        channels: Number of audio channels (default: 1 for mono)

    Yields:
        numpy.ndarray: Audio chunks as 1D float32 arrays (mono) or 2D arrays (stereo)

    Raises:
        ValueError: If device_id is invalid or doesn't support input
        sd.PortAudioError: If audio stream fails to start

    Example:
        >>> for chunk in capture_audio(device_id=2, chunk_duration=5.0):
        ...     # Process each 5-second audio chunk
        ...     print(f"Received chunk with {len(chunk)} samples")
    """
    # Validate inputs
    if not validate_device(device_id):
        raise ValueError(f"Device {device_id} is not valid or doesn't support audio input")

    if chunk_duration <= 0:
        raise ValueError(f"chunk_duration must be positive, got {chunk_duration}")

    # Calculate samples per chunk
    chunk_samples = int(sample_rate * chunk_duration)

    # Queue to pass audio from callback thread to main thread
    audio_queue: Queue[np.ndarray] = Queue(maxsize=10)  # Buffer up to 10 chunks

    def audio_callback(indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """
        Callback function called by sounddevice for each audio block.

        Runs in a separate thread. Copies audio data to the queue for the main thread.
        """
        if status:
            print(f"[Audio Warning] {status}", file=sys.stderr)

        # Copy data to avoid issues with buffer reuse
        audio_queue.put(indata.copy())

    # Open audio input stream
    try:
        stream = sd.InputStream(
            device=device_id,
            channels=channels,
            samplerate=sample_rate,
            blocksize=chunk_samples,
            callback=audio_callback,
            dtype='float32'
        )

        with stream:
            print(f"[Audio] Capturing from device {device_id} ({get_device_name(device_id)})")
            print(f"[Audio] Sample rate: {sample_rate} Hz, Chunk duration: {chunk_duration}s, Channels: {channels}")

            # Continuously yield chunks from the queue
            while True:
                try:
                    # Get chunk with timeout to allow for graceful shutdown
                    chunk = audio_queue.get(timeout=1.0)

                    # Convert stereo to mono if needed by averaging channels
                    if channels > 1 and chunk.ndim > 1:
                        chunk = chunk.mean(axis=1)

                    yield chunk

                except Empty:
                    # Queue is empty, continue waiting
                    continue

    except sd.PortAudioError as e:
        print(f"\n[Error] Failed to open audio stream: {e}", file=sys.stderr)
        print("\nTroubleshooting tips:", file=sys.stderr)
        print("1. Check that the device ID is correct (use --list-devices)", file=sys.stderr)
        print("2. Ensure BlackHole or loopback device is installed", file=sys.stderr)
        print("3. Check Audio MIDI Setup for Multi-Output Device configuration", file=sys.stderr)
        raise
    except KeyboardInterrupt:
        print("\n[Audio] Stopping capture...")
        raise  # Re-raise to ensure proper cleanup
