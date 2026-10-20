#!/bin/bash
# Setup script for listn - Audio Transcription CLI

set -e  # Exit on error

echo "=========================================="
echo "  listn - Audio Transcription Setup"
echo "=========================================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This tool is designed for macOS only."
    exit 1
fi

# Check for Python 3.8+
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Install Python from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo ""
    echo "Warning: Homebrew is not installed."
    echo "Install from: https://brew.sh"
    echo ""
    read -p "Continue without BlackHole installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    SKIP_BLACKHOLE=true
fi

# Install BlackHole if not skipped
if [ "$SKIP_BLACKHOLE" != "true" ]; then
    echo ""
    echo "Checking for BlackHole audio driver..."

    if brew list blackhole-2ch &> /dev/null; then
        echo "✓ BlackHole is already installed"
    else
        echo "Installing BlackHole..."
        brew install blackhole-2ch
        echo "✓ BlackHole installed successfully"
    fi
fi

# Ask about speaker diarization
echo ""
read -p "Install speaker diarization support? (requires FFmpeg) (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    INSTALL_DIARIZATION=true

    echo ""
    echo "Checking for FFmpeg..."
    if brew list ffmpeg &> /dev/null; then
        echo "✓ FFmpeg is already installed"
    else
        echo "Installing FFmpeg..."
        brew install ffmpeg
        echo "✓ FFmpeg installed successfully"
    fi
else
    INSTALL_DIARIZATION=false
fi

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Installing Python dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install dependencies
pip install -r requirements.txt

# Install speaker diarization if selected
if [ "$INSTALL_DIARIZATION" = "true" ]; then
    echo ""
    echo "Installing speaker diarization dependencies..."
    pip install pyannote.audio soundfile
    echo "✓ Speaker diarization dependencies installed"
fi

echo "✓ Dependencies installed successfully"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Configure Audio (one-time setup):"
echo "   - Open 'Audio MIDI Setup' (Applications → Utilities)"
echo "   - Click '+' → Create Multi-Output Device"
echo "   - Check: Built-in Output + BlackHole 2ch"
echo "   - Set as system audio output (optional)"
echo ""
echo "2. Find your audio device:"
echo "   source venv/bin/activate"
echo "   python main.py --list-devices"
echo ""
echo "3. Start transcribing:"
echo "   python main.py --device <ID>"
echo ""

if [ "$INSTALL_DIARIZATION" = "true" ]; then
    echo "4. (Optional) Enable speaker identification:"
    echo "   - Get HuggingFace token:"
    echo "     * Sign up at https://huggingface.co"
    echo "     * Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "     * Get token from https://huggingface.co/settings/tokens"
    echo "   - Set token: export HF_TOKEN='your_token_here'"
    echo "   - Run: python main.py --device <ID> --enable-diarization"
    echo ""
fi

echo "See README.md for detailed instructions."
