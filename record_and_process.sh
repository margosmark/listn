#!/usr/bin/env bash
#
# All-in-one script: Record audio then process with transcription + speaker diarization
#
# This script combines recording and post-processing for the complete workflow.
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "Record & Process Audio - Complete Workflow"
echo "========================================"
echo ""

# Check if device ID is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Available devices:${NC}"
    echo ""
    python3 record_audio.py --list-devices
    echo ""
    echo -e "${RED}Error: No device ID specified${NC}"
    echo ""
    echo "Usage: ./record_and_process.sh <device_id> [output_name] [options]"
    echo ""
    echo "Examples:"
    echo "  ./record_and_process.sh 2                          # Use device 2, auto-name file"
    echo "  ./record_and_process.sh 2 interview                # Save as interview.wav"
    echo "  ./record_and_process.sh 2 meeting --model medium   # With options"
    echo ""
    exit 1
fi

DEVICE_ID="$1"
shift

# Get output name (optional)
OUTPUT_NAME=""
if [ -n "$1" ] && [[ ! "$1" =~ ^-- ]]; then
    OUTPUT_NAME="$1"
    shift
fi

# Generate filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -z "$OUTPUT_NAME" ]; then
    AUDIO_FILE="meeting_${TIMESTAMP}.wav"
else
    # Add .wav extension if not present
    if [[ "$OUTPUT_NAME" != *.wav ]]; then
        AUDIO_FILE="${OUTPUT_NAME}.wav"
    else
        AUDIO_FILE="$OUTPUT_NAME"
    fi
fi

echo -e "${BLUE}Step 1: Recording Audio${NC}"
echo "========================================"
echo "Device ID: $DEVICE_ID"
echo "Output file: $AUDIO_FILE"
echo ""
echo "Recording will start in 3 seconds..."
echo "Press Ctrl+C when done recording"
echo ""
sleep 3

# Record audio
python3 record_audio.py --device "$DEVICE_ID" --output "$AUDIO_FILE"

# Check if recording was successful
if [ ! -f "$AUDIO_FILE" ]; then
    echo -e "${RED}Error: Recording failed - audio file not created${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Recording complete!${NC}"
echo ""
echo "========================================"
echo ""

# Ask user if they want to process now
echo -e "${BLUE}Step 2: Process & Transcribe${NC}"
echo "========================================"
echo ""
read -p "Process this audio now? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Recording saved. You can process it later with:"
    echo ""
    echo "  ./process_interview.sh $AUDIO_FILE --enable-diarization"
    echo ""
    exit 0
fi

echo ""
echo "Processing audio file..."
echo ""

# Process with the helper script, passing through any additional options
./process_interview.sh "$AUDIO_FILE" "$@"

echo ""
echo "========================================"
echo -e "${GREEN}✓ Complete Workflow Finished!${NC}"
echo "========================================"
echo ""
echo "Files created:"
echo "  Audio: $AUDIO_FILE"

# Try to find the transcript file
TRANSCRIPT="${AUDIO_FILE%.wav}_transcript.txt"
if [ -f "$TRANSCRIPT" ]; then
    echo "  Transcript: $TRANSCRIPT"
fi

echo ""
