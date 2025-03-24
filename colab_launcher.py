"""
Launcher script for YouTube Video Creator in Google Colab with Gradio Interface
Simply run this script in a Colab cell to download and set up everything

This version uses a Gradio web interface for a user-friendly experience.

Usage in Colab:
!curl -s https://raw.githubusercontent.com/YOUR_USERNAME/youtube-video-creator/main/colab_launcher.py | python
"""
import os
import sys
import subprocess
import urllib.request

def download_file(url, filename):
    """Download a file from a URL to a local file"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def main():
    """Main function to download and set up the YouTube Video Creator"""
    print("=" * 60)
    print("Setting up YouTube Video Creator with Gradio for Google Colab".center(60))
    print("=" * 60)
    
    # List of files to download
    # Replace these URLs with your actual GitHub/cloud storage URLs
    files = {
        "setup.py": "REPLACE_WITH_ACTUAL_URL/setup.py",
        "app.py": "REPLACE_WITH_ACTUAL_URL/app.py"
    }
    
    # Download each file
    any_failed = False
    for filename, url in files.items():
        if not download_file(url, filename):
            any_failed = True
    
    if any_failed:
        print("\nSome files failed to download. Please check the errors above.")
        print("You can manually upload the files from github.com/YOUR_USERNAME/youtube-video-creator")
        return
    
    # Run the setup script
    print("\nRunning setup script...")
    try:
        subprocess.run([sys.executable, "setup.py"])
        print("Setup completed successfully!")
        
        # Run the Gradio app
        print("\nStarting YouTube Video Creator with Gradio...\n")
        print("NOTE: This will start a web server with a Gradio interface.")
        print("When it's ready, you'll see a URL you can click to access the interface.")
        subprocess.run([sys.executable, "app.py", "--device", "cpu", "--compute_type", "float32"])
    except Exception as e:
        print(f"Error during setup: {e}")
        print("\nYou can try running the scripts manually:")
        print("1. %run setup.py")
        print("2. !python app.py --device cpu --compute_type float32")

if __name__ == "__main__":
    main() 