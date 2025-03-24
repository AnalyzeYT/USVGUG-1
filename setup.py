"""
Setup script for YouTube Video Creator
Run this first in Google Colab to install dependencies and set up the environment
"""
import os
import sys
import subprocess
import time

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    
    # Install system packages
    print("Installing ffmpeg...")
    subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg", shell=True)
    
    # Install Python packages
    print("Installing Python packages...")
    packages = [
        "edge-tts==6.1.9",
        "requests>=2.28.2",
        "numpy>=1.23.5",
        "Pillow>=9.4.0",
        "moviepy>=1.0.3",
        "whisperx>=3.2.0", 
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "ipywidgets>=8.0.0",
        "gradio>=3.50.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package])
    
    print("All dependencies installed successfully!")

def configure_environment():
    """Configure the environment for Colab"""
    print("Configuring environment...")
    
    # Configure MoviePy
    print("Configuring MoviePy...")
    os.makedirs(os.path.expanduser("~/.config/moviepy"), exist_ok=True)
    with open(os.path.expanduser("~/.config/moviepy/moviepy.conf"), "w") as f:
        f.write('{"FFMPEG_BINARY": "ffmpeg", "IMAGEMAGICK_BINARY": "convert"}')
    
    # Create output directory
    print("Creating output directory...")
    os.makedirs("outputs", exist_ok=True)
    
    # Fix ALSA-related warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("Environment configured successfully!")

if __name__ == "__main__":
    start_time = time.time()
    print("Setting up YouTube Video Creator environment...")
    
    install_dependencies()
    configure_environment()
    
    end_time = time.time()
    print(f"Setup completed in {end_time - start_time:.2f} seconds!")
    print("\nYou can now run the YouTube Video Creator by running app.py") 