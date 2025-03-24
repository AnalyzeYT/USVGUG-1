"""
Simple script to run the YouTube video creator app in Google Colab
"""

import os
import sys
import subprocess
import pkg_resources

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    
    # Install ffmpeg and audio dependencies
    subprocess.run("apt-get update && apt-get install -y ffmpeg", shell=True)
    
    # Install Python packages
    required = {'gradio', 'edge-tts', 'requests', 'numpy', 'Pillow', 
                'moviepy', 'whisperx', 'torch', 'torchvision', 'torchaudio'}
    
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if missing:
        print(f"Missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    
    print("Requirements installed.")

def configure_moviepy():
    """Configure MoviePy for Colab"""
    print("Configuring MoviePy...")
    os.makedirs(os.path.expanduser("~/.config/moviepy"), exist_ok=True)
    
    with open(os.path.expanduser("~/.config/moviepy/moviepy.conf"), "w") as f:
        f.write('{"FFMPEG_BINARY": "ffmpeg", "IMAGEMAGICK_BINARY": "convert"}')
    
    print("MoviePy configured.")

def set_api_keys():
    """Set API keys from user input"""
    print("\nAPI Key Configuration")
    print("---------------------")
    
    # Pexels API Key
    pexels_key = os.environ.get("PEXELS_API_KEY", "")
    if not pexels_key:
        pexels_key = input("Enter your Pexels API key (press Enter to skip): ")
        if pexels_key:
            os.environ["PEXELS_API_KEY"] = pexels_key
    
    # Tenor API Key
    tenor_key = os.environ.get("TENOR_API_KEY", "")
    if not tenor_key:
        tenor_key = input("Enter your Tenor API key (press Enter to skip): ")
        if tenor_key:
            os.environ["TENOR_API_KEY"] = tenor_key
    
    print("API keys configured.")

def run_app():
    """Run the app with public URL"""
    print("\nStarting the app...")
    subprocess.run(["python", "app.py", "--share"])

if __name__ == "__main__":
    print("Setting up YouTube Video Creator for Google Colab")
    print("===============================================")
    
    install_requirements()
    configure_moviepy()
    set_api_keys()
    run_app() 