"""
Launcher script for YouTube Video Creator in Google Colab
Simply run this script in a Colab cell to download and set up everything

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
    print("Setting up YouTube Video Creator for Google Colab".center(60))
    print("=" * 60)
    
    # List of files to download
    # Replace these URLs with your actual GitHub/cloud storage URLs
    files = {
        "setup.py": "REPLACE_WITH_ACTUAL_URL/setup.py",
        "video_creator.py": "REPLACE_WITH_ACTUAL_URL/video_creator.py",
        "run_app.py": "REPLACE_WITH_ACTUAL_URL/run_app.py"
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
        
        # Run the main application
        print("\nStarting YouTube Video Creator...\n")
        subprocess.run([sys.executable, "run_app.py"])
    except Exception as e:
        print(f"Error during setup: {e}")
        print("\nYou can try running the scripts manually:")
        print("1. %run setup.py")
        print("2. %run run_app.py")

if __name__ == "__main__":
    main() 