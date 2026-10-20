#!/usr/bin/env bash
#
# Helper script to process audio files with speaker diarization.
# Makes it easy to transcribe interviews, meetings, or podcasts.
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================"
echo "Audio Post-Processing Helper"
echo "================================"
echo ""

# Step 1: Check if audio file is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: No audio file specified${NC}"
    echo ""
    echo "Usage: ./process_interview.sh <audio_file> [options]"
    echo ""
    echo "Examples:"
    echo "  ./process_interview.sh interview.wav"
    echo "  ./process_interview.sh meeting.mp3 --output meeting_transcript.txt"
    echo "  ./process_interview.sh podcast.m4a --model large --language en"
    echo ""
    exit 1
fi

AUDIO_FILE="$1"
shift  # Remove first argument, rest are passed to process_audio.py

# Step 2: Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo -e "${RED}Error: Audio file not found: $AUDIO_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Audio file found: $AUDIO_FILE${NC}"
echo ""

# Step 3: Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠ Warning: Virtual environment not activated${NC}"
    echo "  Consider activating it: source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 4: Check if librosa is installed
echo "Checking dependencies..."
if ! python3 -c "import librosa" 2>/dev/null; then
    echo -e "${RED}✗ librosa not installed${NC}"
    echo ""
    echo "Installing librosa..."
    pip install librosa
    echo -e "${GREEN}✓ librosa installed${NC}"
else
    echo -e "${GREEN}✓ librosa is installed${NC}"
fi

# Step 5: Check if speaker diarization is requested
ENABLE_DIARIZATION=false
for arg in "$@"; do
    if [ "$arg" = "--enable-diarization" ]; then
        ENABLE_DIARIZATION=true
        break
    fi
done

# Step 6: Check HF_TOKEN if diarization enabled
if [ "$ENABLE_DIARIZATION" = true ]; then
    echo ""
    echo "Checking speaker diarization setup..."

    # Check if pyannote.audio is installed
    if ! python3 -c "import pyannote.audio" 2>/dev/null; then
        echo -e "${RED}✗ pyannote.audio not installed${NC}"
        echo ""
        echo "Installing pyannote.audio..."
        pip install pyannote.audio
        echo -e "${GREEN}✓ pyannote.audio installed${NC}"
    else
        echo -e "${GREEN}✓ pyannote.audio is installed${NC}"
    fi

    # Check HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}✗ HF_TOKEN not set${NC}"
        echo ""
        echo "Speaker diarization requires a HuggingFace token."
        echo ""
        echo "To get a token:"
        echo "  1. Create account at https://huggingface.co"
        echo "  2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
        echo "  3. Get token from https://huggingface.co/settings/tokens"
        echo ""
        read -p "Enter your HuggingFace token (or press Enter to skip diarization): " token
        if [ -n "$token" ]; then
            export HF_TOKEN="$token"
            echo -e "${GREEN}✓ HF_TOKEN set for this session${NC}"
            echo ""
            echo "To make this permanent, add to your ~/.bashrc or ~/.zshrc:"
            echo "  export HF_TOKEN=\"$token\""
        else
            echo -e "${YELLOW}⚠ Continuing without speaker diarization${NC}"
            # Remove --enable-diarization from arguments
            ARGS=()
            for arg in "$@"; do
                if [ "$arg" != "--enable-diarization" ]; then
                    ARGS+=("$arg")
                fi
            done
            set -- "${ARGS[@]}"
            ENABLE_DIARIZATION=false
        fi
    else
        echo -e "${GREEN}✓ HF_TOKEN is set${NC}"
    fi
fi

# Step 7: Determine output file
OUTPUT_FILE=""
HAS_OUTPUT=false
for ((i=1; i<=$#; i++)); do
    if [ "${!i}" = "--output" ]; then
        j=$((i+1))
        OUTPUT_FILE="${!j}"
        HAS_OUTPUT=true
        break
    fi
done

if [ "$HAS_OUTPUT" = false ]; then
    # Generate default output filename
    BASENAME=$(basename "$AUDIO_FILE")
    FILENAME="${BASENAME%.*}"
    OUTPUT_FILE="${FILENAME}_transcript.txt"
    echo ""
    echo -e "${YELLOW}No output file specified. Using: $OUTPUT_FILE${NC}"
    # Add --output to arguments
    set -- "$@" "--output" "$OUTPUT_FILE"
fi

# Step 8: Show processing configuration
echo ""
echo "================================"
echo "Processing Configuration"
echo "================================"
echo "Input:  $AUDIO_FILE"
echo "Output: $OUTPUT_FILE"

if [ "$ENABLE_DIARIZATION" = true ]; then
    echo "Speaker Diarization: ENABLED"
else
    echo "Speaker Diarization: DISABLED"
fi

# Check for model and language in args
MODEL="medium"  # default for post-processing
LANGUAGE="auto-detect"
for ((i=1; i<=$#; i++)); do
    if [ "${!i}" = "--model" ]; then
        j=$((i+1))
        MODEL="${!j}"
    fi
    if [ "${!i}" = "--language" ]; then
        j=$((i+1))
        LANGUAGE="${!j}"
    fi
done

echo "Model:    $MODEL"
echo "Language: $LANGUAGE"
echo "================================"
echo ""

# Step 9: Run the processing
echo "Starting transcription..."
echo ""

# Add --enable-diarization if not already present and it was enabled
if [ "$ENABLE_DIARIZATION" = true ]; then
    HAS_DIARIZATION_FLAG=false
    for arg in "$@"; do
        if [ "$arg" = "--enable-diarization" ]; then
            HAS_DIARIZATION_FLAG=true
            break
        fi
    done
    if [ "$HAS_DIARIZATION_FLAG" = false ]; then
        set -- "$@" "--enable-diarization"
    fi
fi

# Run the process_audio.py script
python3 process_audio.py "$AUDIO_FILE" "$@"

# Step 10: Success message
echo ""
echo "================================"
echo -e "${GREEN}✓ Processing Complete!${NC}"
echo "================================"
echo ""
echo "Transcript saved to: $OUTPUT_FILE"
echo ""

# Show file size
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(wc -l < "$OUTPUT_FILE")
    echo "Lines in transcript: $SIZE"
    echo ""
    echo "View transcript: cat $OUTPUT_FILE"
    echo "Or open in editor: open $OUTPUT_FILE"
fi

echo ""
