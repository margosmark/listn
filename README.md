# listn 🎧

Real-time audio transcription CLI for macOS. Captures system audio (Zoom, speakers, music, etc.) and transcribes continuously using OpenAI's Whisper model.

Perfect for:
- Transcribing Zoom/Teams/Meet calls
- Creating live captions for videos
- Recording meeting notes automatically
- Transcribing podcasts or audio content

## Features

- 🎤 **System audio capture** - Record from Zoom, speakers, or any application
- 🚀 **Real-time transcription** - See text appear as audio plays
- 🗣️ **Speaker identification** - Identify different speakers (optional)
- 🧠 **Local processing** - Everything runs on your Mac (M1 optimized)
- 📝 **Dual output** - Console display + text file
- 🌍 **Multi-language** - Auto-detect or specify language
- ⚡ **Efficient** - 5-second chunks balance latency and accuracy

## Prerequisites

- **macOS** (tested on M1/M2, should work on Intel)
- **Python 3.8+**
- **BlackHole** (virtual audio device for system audio capture)

## Installation

### 1. Install BlackHole (Virtual Audio Device)

BlackHole creates a virtual audio device that allows capturing system audio.

```bash
brew install blackhole-2ch
```

Or download from: https://existential.audio/blackhole/

### 2. Configure Multi-Output Device

This step routes audio to BOTH your speakers AND BlackHole (so you can hear and record).

1. Open **Audio MIDI Setup** (Applications → Utilities → Audio MIDI Setup)
2. Click the **+** button at the bottom left
3. Select **Create Multi-Output Device**
4. In the new device, check both:
   - ✅ **Built-in Output** (or your speakers/headphones)
   - ✅ **BlackHole 2ch**
5. (Optional) Click the gear icon → "Use This Device For Sound Output"

**Visual Guide:**

```
┌─────────────────────────────────────┐
│ Multi-Output Device                 │
├─────────────────────────────────────┤
│ ✅ Built-in Output                  │  ← You hear audio
│ ✅ BlackHole 2ch                    │  ← listn captures audio
└─────────────────────────────────────┘
```

### 3. Set System Audio Output

**Option A: System-wide (recommended for always-on transcription)**

1. Open **System Preferences → Sound → Output**
2. Select **Multi-Output Device**

**Option B: Per-application (for specific apps like Zoom)**

1. In Zoom/Teams/etc., go to audio settings
2. Set **Speaker** to **Multi-Output Device**

### 4. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** First run will download the Whisper model (~500MB for small model).

## Usage

**For interviews or meetings where you want the best speaker separation**, see [Post-Processing Audio Files](#post-processing-audio-files) below instead.

### Quick Start (Real-Time Transcription)

```bash
# 1. List available audio devices (find BlackHole device ID)
python main.py --list-devices

# 2. Start transcribing (replace '2' with your BlackHole device ID)
python main.py --device 2

# 3. Transcribe and save to file
python main.py --device 2 --output meeting_notes.txt
```

### Common Use Cases

**Transcribe Zoom call:**
```bash
python main.py --device 2 --output zoom_meeting.txt
```

**Transcribe in Spanish:**
```bash
python main.py --device 2 --language es
```

**Use faster model (less accurate but lower latency):**
```bash
python main.py --device 2 --model base
```

**Transcribe without timestamps:**
```bash
python main.py --device 2 --no-timestamps
```

**Transcribe interview with speaker identification:**
```bash
# Set HuggingFace token (one-time setup)
export HF_TOKEN="your_hf_token_here"

# Best settings for interviews: medium model + longer chunks
python main.py --device 2 --enable-diarization --language en --model medium --chunk-duration 15 --output interview.txt
```

### Speaker Identification (Diarization)

listn can identify different speakers in audio using pyannote.audio. This is useful for multi-person conversations, meetings, or interviews.

**Setup:**

1. Install the additional dependency:
   ```bash
   pip install pyannote.audio
   ```

2. Get a HuggingFace token:
   - Create account at https://huggingface.co
   - Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Get token from https://huggingface.co/settings/tokens

3. Set the token as an environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

**Basic Usage:**

```bash
# Basic speaker diarization
python main.py --device 2 --enable-diarization --output meeting.txt

# With language specified (faster)
python main.py --device 2 --enable-diarization --language en --output meeting.txt

# Recommended: Longer chunks for better speaker detection
python main.py --device 2 --enable-diarization --language en --chunk-duration 10 --output interview.txt

# Best quality: Medium model + longer chunks
python main.py --device 2 --enable-diarization --language en --model medium --chunk-duration 15 --output interview.txt
```

**Output format with speakers:**
```
[14:23:45] [SPEAKER_00] Hello everyone, welcome to the meeting.
[14:23:52] [SPEAKER_01] Thanks for having me.
[14:23:58] [SPEAKER_00] Let's start with the agenda for today.
[14:24:05] [SPEAKER_02] I have a question about the timeline.
```

**Tips for Better Speaker Detection:**

1. **Use longer chunks** - Increase `--chunk-duration` from 5 to 10-15 seconds:
   - Gives the model more context to identify different speakers
   - Better for interviews and multi-person meetings
   - Trade-off: Slightly higher latency

2. **Use a better model** - Try `--model medium` for more accurate diarization:
   - Better at distinguishing similar voices
   - Recommended for important meetings/interviews

3. **Audio source matters:**
   - **Aggregate Device** (BlackHole + your mic) - Best for interviews
   - **BlackHole only** - Works but pre-mixed audio is harder to separate

4. **Clear audio quality:**
   - Distinct voices are easier to separate
   - Minimize background noise
   - Good microphone quality helps

**Troubleshooting - Only seeing one speaker:**
- Try `--chunk-duration 15` for longer context
- Verify audio has clearly different speakers (test with YouTube interview)
- Use Aggregate Device instead of BlackHole-only if doing interviews
- Some Zoom/Teams audio is pre-mixed and harder to separate

**Notes:**
- First run will download the diarization model (~300MB)
- Speaker labels (SPEAKER_00, SPEAKER_01, etc.) are assigned automatically
- Same speaker will have the same label throughout the session
- 100% local processing - no cloud dependency after model download

### All Options

```bash
python main.py --help
```

```
Options:
  --list-devices          List all available audio input devices
  --device ID             Audio device ID to capture from (required)
  --output FILE           Output file path (optional, defaults to console only)
  --model SIZE            Whisper model: tiny, base, small, medium, large
                          (default: small)
  --language CODE         Language code (e.g., en, es, fr) for faster processing
  --chunk-duration SEC    Audio chunk duration in seconds (default: 5.0)
  --no-timestamps         Don't include timestamps in output
  --verbose               Show detailed Whisper output
  --enable-diarization    Enable speaker identification (requires HF_TOKEN)
  --hf-token TOKEN        HuggingFace token for speaker diarization
```

## Model Sizes

Choose based on your needs and Mac specs:

| Model  | Size  | RAM   | Speed     | Accuracy |
|--------|-------|-------|-----------|----------|
| tiny   | ~75MB | ~1GB  | Fastest   | Good     |
| base   | ~150MB| ~1GB  | Fast      | Better   |
| small  | ~500MB| ~2GB  | Moderate  | Great    | ← **Default**
| medium | ~1.5GB| ~5GB  | Slow      | Excellent|
| large  | ~3GB  | ~10GB | Slowest   | Best     |

**Recommendation:** Start with `small` (default). Use `base` if you need faster processing.

## Output Format

### Console Output
```
[14:23:45] Hello everyone, welcome to the meeting.
[14:23:52] Let's start with the agenda for today.
[14:23:58] First item is the project timeline.
```

### File Output
Same format as console, appended to the specified file. File is flushed after each line so you won't lose data if interrupted.

## Troubleshooting

### "Device not found" or "No audio captured"

1. **Verify BlackHole is installed:**
   ```bash
   python main.py --list-devices
   ```
   Look for "BlackHole 2ch" in the list.

2. **Check Multi-Output Device:**
   - Open Audio MIDI Setup
   - Ensure Multi-Output Device has both speakers and BlackHole checked

3. **Set correct audio output:**
   - System Preferences → Sound → Output
   - Or in your app (Zoom, Chrome, etc.)

### "Poor transcription quality"

1. **Use a larger model:**
   ```bash
   python main.py --device 2 --model medium
   ```

2. **Specify language (faster and more accurate):**
   ```bash
   python main.py --device 2 --language en
   ```

3. **Check audio quality:**
   - Ensure volume is reasonable (not too quiet)
   - Minimize background noise

### "Slow / Laggy transcription"

1. **Use a smaller model:**
   ```bash
   python main.py --device 2 --model base
   ```

2. **Increase chunk duration (less frequent but longer delay):**
   ```bash
   python main.py --device 2 --chunk-duration 10
   ```

3. **Close other resource-intensive apps**

### "Empty transcriptions"

- This is normal for silence/noise
- Whisper returns empty text when no clear speech is detected
- These are automatically filtered out

## How It Works

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   System     │───▶│  BlackHole   │───▶│    listn     │
│   Audio      │    │  (Virtual)   │    │  (Capture)   │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │   Whisper    │
                                        │ (Transcribe) │
                                        └──────┬───────┘
                                               │
                                               ▼
                                   ┌───────────────────────┐
                                   │  Console + Text File  │
                                   └───────────────────────┘
```

1. **Audio Routing:** System audio plays through Multi-Output Device (speakers + BlackHole)
2. **Capture:** listn captures audio from BlackHole in 5-second chunks
3. **Transcribe:** Each chunk is sent to Whisper (runs locally on M1 GPU)
4. **Output:** Transcribed text appears in console and/or file

## Architecture

Simple 4-module design:

- **main.py** - CLI orchestration, argument parsing, main loop
- **audio.py** - Audio capture using sounddevice + BlackHole
- **transcribe.py** - Whisper model loading and transcription
- **diarization.py** - Speaker identification using pyannote.audio (optional)

No frameworks, no databases, just Python and a few focused libraries.

## Performance

On M1 Mac:
- Model load time: ~5 seconds (first run only, then cached)
- Transcription: ~1-2 seconds for 5-second audio chunk
- Memory: ~2GB for small model
- CPU: Optimized for M1 (uses Metal Performance Shaders)

Real-time capable: Transcribes faster than real-time (2-3x speed).

## Post-Processing Audio Files

For **better speaker separation and transcription quality**, especially for interviews or meetings where you don't need real-time output, use the post-processing script instead.

### Why Post-Processing is Better

**Real-time transcription (`main.py`):**
- Processes audio in 5-15 second chunks
- Limited context for speaker identification
- Speaker labels may be inconsistent across chunks
- Good for: Live captions, real-time monitoring

**Post-processing (`process_audio.py`):**
- Analyzes the entire audio file at once
- Full conversation context for better speaker detection
- Consistent speaker labels throughout the recording
- More accurate speaker boundaries
- Good for: Interviews, meetings, podcasts (recorded audio)

### Recording Audio

**Finding Your Device ID:**

First, find your BlackHole device ID (varies by system):

```bash
python record_audio.py --list-devices
# or
python main.py --list-devices
```

Look for "BlackHole 2ch" in the list - that's the device ID you'll use (e.g., 2, 8, etc.)

**Recording Options:**

**Option 1: Live Transcription + Recording (Recommended)** ⭐

Record audio to file WHILE seeing real-time transcription:

```bash
# Quick start - auto-generate filename (meeting_TIMESTAMP.wav)
python record_with_live_transcription.py --device 8

# Custom filename
python record_with_live_transcription.py --device 8 --output interview.wav

# With language specified (faster real-time transcription)
python record_with_live_transcription.py --device 8 --language en

# Choose model for real-time transcription
python record_with_live_transcription.py --device 8 --model base    # Fast (default)
python record_with_live_transcription.py --device 8 --model small   # Better accuracy
python record_with_live_transcription.py --device 8 --model tiny    # Fastest

# Full options
python record_with_live_transcription.py --device 8 --model small --language en --output meeting.wav
```

**Available models for real-time:**
- `tiny` - Fastest, lowest accuracy (use if your Mac is slow)
- `base` - Fast and accurate (default, recommended)
- `small` - Better accuracy, a bit slower
- `medium` - Best accuracy, slower (may lag on older Macs)

Benefits:
- 👀 See transcription live during the meeting
- 💾 Audio file saved automatically
- 🎯 Post-process later for better speaker separation
- 📝 Two transcripts: real-time (quick) + post-processed (accurate)

After recording, post-process for better speaker diarization:
```bash
./process_interview.sh meeting_20260121_143022.wav --enable-diarization --model medium
```

**Option 2: Record-Then-Process Workflow**

Record first, process after:

```bash
# Make script executable (first time only)
chmod +x record_and_process.sh

# Auto-generate filename: meeting_TIMESTAMP.wav
./record_and_process.sh 8

# Custom filename
./record_and_process.sh 8 interview --enable-diarization --model medium
```

**Option 3: Record Only (No Transcription)**

Just record audio, process later:

```bash
python record_audio.py --device 8 --output meeting.wav
# Press Ctrl+C when done
```

**Option 4: Use External Tools**

Record with QuickTime, Audio Hijack, or ffmpeg:

```bash
# Using ffmpeg (if installed)
ffmpeg -f avfoundation -i ":BlackHole 2ch" -t 3600 meeting.wav
```

**All Options Reference:**

```bash
# Live transcription + recording options
python record_with_live_transcription.py --help

Options:
  --device ID             Audio device ID (required)
  --output FILE           Output WAV file (default: meeting_TIMESTAMP.wav)
  --model SIZE            Whisper model: tiny, base, small, medium (default: base)
  --language CODE         Language code (e.g., en, es, fr) for faster processing
  --chunk-duration SEC    Audio chunk duration in seconds (default: 5.0)
  --no-timestamps         Don't include timestamps in real-time transcription
```

### Usage

**Easiest Way - Use the Helper Script:**

The `process_interview.sh` script handles all the setup steps automatically:

```bash
# Make it executable (first time only)
chmod +x process_interview.sh

# Basic usage - automatically installs dependencies and sets up everything
./process_interview.sh interview.wav

# With speaker diarization (will prompt for HF_TOKEN if not set)
./process_interview.sh interview.wav --enable-diarization

# Full options
./process_interview.sh meeting.mp3 --enable-diarization --model medium --language en --output meeting.txt
```

The script will:
- ✓ Check if audio file exists
- ✓ Install librosa if needed
- ✓ Install pyannote.audio if diarization requested
- ✓ Prompt for HF_TOKEN if diarization enabled but token not set
- ✓ Generate output filename automatically if not specified
- ✓ Show progress and results

**Manual Usage:**

If you prefer to run `process_audio.py` directly:

**Basic transcription of an audio file:**
```bash
python process_audio.py interview.wav --output transcript.txt
```

**With speaker diarization (recommended for interviews):**
```bash
# Make sure HF_TOKEN is set
export HF_TOKEN="your_token_here"

# Process with speaker identification
python process_audio.py interview.wav --enable-diarization --output transcript.txt
```

**Best quality settings:**
```bash
# Medium model + speaker diarization
python process_audio.py meeting.mp3 --model medium --enable-diarization --language en --output meeting.txt
```

### Supported Audio Formats

The post-processing script uses librosa and supports:
- WAV, MP3, M4A, FLAC, OGG
- Any format supported by ffmpeg

**Install librosa** (if not already installed):
```bash
pip install librosa
```

### Complete Workflow Examples

**Recommended: Live Transcription + Post-Processing** ⭐
```bash
# Step 1: Record with live transcription (see it in real-time!)
python record_with_live_transcription.py --device 8 --language en
# You'll see transcription appear live during the meeting
# Audio saved as: meeting_20260121_143022.wav

# Step 2: Post-process for better speaker separation
./process_interview.sh meeting_20260121_143022.wav --enable-diarization --model medium
```

**Why this is best:**
- 👀 See transcription live - know what's happening during the meeting
- 💾 Audio file saved - can re-process later with better settings
- 🎯 Post-process gives better speaker labels (analyzes full conversation)
- 📝 Two transcripts: real-time (quick) + post-processed (accurate speakers)

**Option 1: Record-Then-Process (No live view)**
```bash
# Auto-generate filename with timestamp
./record_and_process.sh 8

# Or with custom name
./record_and_process.sh 8 interview --enable-diarization --model medium
```

**Option 2: Just record, process later**
```bash
# Record audio only
python record_audio.py --device 8 --output meeting.wav
# (Press Ctrl+C when done)

# Process anytime later
./process_interview.sh meeting.wav --enable-diarization --model medium
```

**Option 3: External tool + post-process**
```bash
# Record with QuickTime, Audio Hijack, etc. → Save as interview.wav

# Process with helper script
./process_interview.sh interview.wav --enable-diarization --model medium --language en
```

**Option 4: Manual processing (without helper script)**
```bash
# 1. Install dependencies
pip install librosa pyannote.audio

# 2. Set HF_TOKEN if using diarization
export HF_TOKEN="your_token_here"

# 3. Process audio file directly
python process_audio.py interview.wav --enable-diarization --language en --model medium --output interview_transcript.txt
```

### Post-Processing Options

```bash
python process_audio.py --help
```

```
Options:
  audio_file              Input audio file (wav, mp3, m4a, etc.)
  --output FILE           Output transcript file (optional)
  --model SIZE            Whisper model: tiny, base, small, medium, large
                         (default: medium - recommended for post-processing)
  --language CODE         Language code (e.g., en, es, fr) for faster processing
  --enable-diarization    Enable speaker identification (requires HF_TOKEN)
  --hf-token TOKEN        HuggingFace token for speaker diarization
```

### Output Format

With speaker diarization enabled:
```
[00:15] [SPEAKER_00] Hello everyone, welcome to the meeting.
[00:22] [SPEAKER_01] Thanks for having me.
[00:28] [SPEAKER_00] Let's start with the agenda for today.
[00:35] [SPEAKER_02] I have a question about the timeline.
```

## Future Enhancements

Possible additions (not implemented yet):

- [x] Voice activity detection (skip silent chunks) ✅
- [x] Speaker diarization (identify different speakers) ✅
- [x] Post-processing for better speaker separation ✅
- [ ] WebSocket streaming for real-time web display
- [ ] Multiple output formats (JSON, SRT subtitles)
- [ ] Cloud transcription API fallback

## License

MIT License - feel free to use and modify.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Transcription model
- [BlackHole](https://existential.audio/blackhole/) - Virtual audio driver
- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio capture library

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify your BlackHole setup with `--list-devices`
3. Try a different model size or chunk duration

---

**Happy transcribing! 🎧→📝**
