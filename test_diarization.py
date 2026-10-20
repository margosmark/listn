#!/usr/bin/env python3
"""
Test script to debug pyannote.audio speaker diarization.
"""

import os
import numpy as np
import torch
from diarization import load_diarization_pipeline

# Create a dummy 5-second audio chunk (16kHz, mono)
sample_rate = 16000
duration = 5.0
num_samples = int(sample_rate * duration)

# Generate some random audio (simulating captured audio)
audio_chunk = np.random.randn(num_samples).astype(np.float32) * 0.1

print("=== Testing Speaker Diarization ===")
print(f"Audio shape: {audio_chunk.shape}")
print(f"Audio dtype: {audio_chunk.dtype}")
print(f"Audio range: [{audio_chunk.min():.3f}, {audio_chunk.max():.3f}]")
print()

# Load pipeline
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("Error: HF_TOKEN environment variable not set")
    exit(1)

print("Loading diarization pipeline...")
pipeline = load_diarization_pipeline(hf_token)
print("Pipeline loaded successfully")
print()

# Test 1: Direct tensor approach
print("Test 1: Passing audio as torch tensor")
try:
    audio_tensor = torch.from_numpy(audio_chunk).float()
    print(f"  Tensor shape before unsqueeze: {audio_tensor.shape}")
    print(f"  Tensor ndim: {audio_tensor.ndim}")

    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    print(f"  Tensor shape after unsqueeze: {audio_tensor.shape}")
    print(f"  Tensor dtype: {audio_tensor.dtype}")

    audio_dict = {
        "waveform": audio_tensor,
        "sample_rate": sample_rate
    }

    print(f"  Input dict keys: {audio_dict.keys()}")
    print(f"  Waveform type: {type(audio_dict['waveform'])}")
    print(f"  Waveform is torch.Tensor: {isinstance(audio_dict['waveform'], torch.Tensor)}")
    print()

    print("  Running pipeline...")
    diarization = pipeline(audio_dict)

    print(f"  Success! Result type: {type(diarization)}")
    print(f"  Has speaker_diarization: {hasattr(diarization, 'speaker_diarization')}")

    print("  Extracting segments...")
    segments = []

    # Check if it's a DiarizeOutput object (new API)
    if hasattr(diarization, 'speaker_diarization'):
        print("  Using new API (DiarizeOutput.speaker_diarization)")
        annotation = diarization.speaker_diarization
        print(f"  Annotation type: {type(annotation)}")

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
            print(f"    {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")

    # Check if it's an Annotation object (old API)
    elif hasattr(diarization, 'itertracks'):
        print("  Using old API (Annotation)")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
            print(f"    {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")

    if not segments:
        print("    (No speakers detected - this is expected for random noise)")

    print(f"\n  Total segments found: {len(segments)}")
    print("\n✓ Test 1 PASSED")

except Exception as e:
    print(f"\n✗ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Testing Complete ===")
