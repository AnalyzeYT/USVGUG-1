#!/bin/bash
# One-click setup script for YouTube Video Creator on Colab

echo "Setting up YouTube Video Creator for Google Colab"
echo "==============================================="

# Install system dependencies
echo "Installing system dependencies..."
apt-get update > /dev/null
apt-get install -y ffmpeg libsndfile1 > /dev/null

# Fix ALSA issues by creating dummy config
echo "Configuring audio..."
mkdir -p /etc/modprobe.d/
echo "options snd-dummy enable=1 index=0" > /etc/modprobe.d/alsa.conf

# Configure MoviePy
echo "Configuring MoviePy..."
mkdir -p ~/.config/moviepy
echo '{"FFMPEG_BINARY": "ffmpeg", "IMAGEMAGICK_BINARY": "convert"}' > ~/.config/moviepy/moviepy.conf

# Install Python packages
echo "Installing Python requirements..."
pip install -q gradio edge-tts requests numpy Pillow moviepy whisperx torch torchvision torchaudio

# Prompt for API keys
echo -e "\nAPI Key Configuration"
echo -e "---------------------"

# Pexels API Key
read -p "Enter your Pexels API key (press Enter to skip): " PEXELS_API_KEY
if [ ! -z "$PEXELS_API_KEY" ]; then
    export PEXELS_API_KEY=$PEXELS_API_KEY
fi

# Tenor API Key  
read -p "Enter your Tenor API key (press Enter to skip): " TENOR_API_KEY
if [ ! -z "$TENOR_API_KEY" ]; then
    export TENOR_API_KEY=$TENOR_API_KEY
fi

echo -e "\nStarting the app..."
python app.py --device cpu --compute_type float32 --share 