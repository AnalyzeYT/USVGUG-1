"""
YouTube Video Creator - Direct Python Implementation
No Gradio UI - For Google Colab
"""

import os
import re
import time
import random
import requests
import numpy as np
import edge_tts
import asyncio
import tempfile
import shutil
from PIL import Image
import warnings
from IPython.display import display, HTML, Audio, Video
from moviepy.editor import *
import torch
import whisperx
import ipywidgets as widgets
from typing import List, Dict, Any, Tuple, Optional

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
VOICE = "en-US-AnaNeural"  # Default voice for Edge TTS
OUTPUT_DIR = "outputs"     # Directory to store temporary and output files
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Configure MoviePy for Colab
import subprocess
try:
    subprocess.run(['mkdir', '-p', os.path.expanduser('~/.config/moviepy')])
    with open(os.path.expanduser('~/.config/moviepy/moviepy.conf'), 'w') as f:
        f.write('{"FFMPEG_BINARY": "ffmpeg", "IMAGEMAGICK_BINARY": "convert"}')
except Exception as e:
    print(f"Warning: Could not configure MoviePy: {e}")

# Load WhisperX model
def load_whisper_model(device="cpu", compute_type="float32"):
    """Load WhisperX model with appropriate settings for Colab"""
    print(f"Loading WhisperX model (device: {device}, compute type: {compute_type})...")
    try:
        model = whisperx.load_model("base", device, compute_type=compute_type)
        print("WhisperX model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading WhisperX model: {e}")
        print("Attempting fallback configuration...")
        try:
            # Fallback to CPU
            device = "cpu"
            compute_type = "float32"
            model = whisperx.load_model("base", device, compute_type=compute_type)
            print("WhisperX model loaded with fallback configuration")
            return model
        except Exception as e:
            print(f"Critical error loading WhisperX model: {e}")
            raise

# Try to load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float32"  # Always use float32 for Colab compatibility
whisper_model = load_whisper_model(device, compute_type)

def clean_text(text):
    """Clean text to remove special characters and normalize spacing"""
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def text_to_speech(text, voice=VOICE):
    """Convert text to speech using Edge TTS"""
    output_file = os.path.join(OUTPUT_DIR, "speech.mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return output_file

def extract_timestamps(audio_file):
    """Extract word-level timestamps using WhisperX"""
    audio = whisperx.load_audio(audio_file)
    result = whisper_model.transcribe(audio, language="en")
    
    # Get word-level timestamps
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    
    words_with_timestamps = []
    for segment in result["segments"]:
        for word in segment["words"]:
            words_with_timestamps.append({
                "word": word["word"],
                "start": word["start"],
                "end": word["end"]
            })
    
    return words_with_timestamps

def fetch_pexels_videos(query, api_key, per_page=5):
    """Fetch videos from Pexels API based on query"""
    if not api_key:
        print("Pexels API key is required")
        return []
    
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}"
    headers = {"Authorization": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            videos = []
            for video in data.get("videos", []):
                # Get the medium size video file
                video_files = video.get("video_files", [])
                for file in video_files:
                    if file["quality"] == "sd" and file["width"] < 900:
                        videos.append({
                            "url": file["link"],
                            "width": file["width"],
                            "height": file["height"]
                        })
                        break
            print(f"Found {len(videos)} videos for query: {query}")
            return videos
        else:
            print(f"Pexels API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching Pexels videos: {e}")
        return []

def fetch_tenor_gifs(query, api_key, limit=5):
    """Fetch GIFs from Tenor API based on query"""
    if not api_key:
        print("Tenor API key is required")
        return []
    
    url = f"https://tenor.googleapis.com/v2/search?q={query}&key={api_key}&limit={limit}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            gifs = []
            for result in data.get("results", []):
                media_formats = result.get("media_formats", {})
                if "gif" in media_formats:
                    gifs.append({
                        "url": media_formats["gif"]["url"],
                        "width": media_formats["gif"]["dims"][0],
                        "height": media_formats["gif"]["dims"][1]
                    })
            print(f"Found {len(gifs)} GIFs for query: {query}")
            return gifs
        else:
            print(f"Tenor API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching Tenor GIFs: {e}")
        return []

def download_media(url, output_path):
    """Download media from URL to specified path"""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return True
        else:
            print(f"Failed to download media: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading media: {e}")
        return False

def get_keywords_from_text(text, num_keywords=5):
    """Extract keywords from text for media search"""
    # Simple implementation - split by spaces and get unique words
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    # Remove common words
    common_words = {"the", "and", "that", "have", "for", "not", "with", "you", "this", "but"}
    words = [word for word in words if word not in common_words]
    
    # Count frequency
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    # Sort by frequency
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in sorted_words[:num_keywords]]
    
    return keywords

def group_words_into_phrases(word_timestamps):
    """Group words into phrases for better media sync"""
    phrases = []
    current_phrase = []
    
    for word_data in word_timestamps:
        current_phrase.append(word_data)
        if word_data["word"].strip() in {",", ".", "!", "?", ";", ":"} or len(current_phrase) >= 5:
            if current_phrase:
                start_time = current_phrase[0]["start"]
                end_time = current_phrase[-1]["end"]
                phrase_text = " ".join([w["word"] for w in current_phrase])
                phrases.append({
                    "text": phrase_text,
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time
                })
            current_phrase = []
    
    # Add any remaining words
    if current_phrase:
        start_time = current_phrase[0]["start"]
        end_time = current_phrase[-1]["end"]
        phrase_text = " ".join([w["word"] for w in current_phrase])
        phrases.append({
            "text": phrase_text,
            "start": start_time,
            "end": end_time,
            "duration": end_time - start_time
        })
    
    return phrases

def loop_clip(clip, target_duration):
    """Loop a clip to reach the target duration"""
    n_loops = int(np.ceil(target_duration / clip.duration))
    return concatenate_videoclips([clip] * n_loops).subclip(0, target_duration)

def add_subtitle_to_clip(clip, text, start_time, end_time):
    """Add subtitle to video clip"""
    txt_clip = TextClip(text, fontsize=30, color='white', font='Arial', 
                       bg_color='black', stroke_color='black', stroke_width=1,
                       method='caption', size=(clip.w, None))
    txt_clip = txt_clip.set_duration(end_time - start_time)
    txt_clip = txt_clip.set_position(('center', 'bottom'))
    
    return CompositeVideoClip([clip, txt_clip])

def create_video(story, genre, title, pexels_api_key, tenor_api_key, include_subtitles=True, resolution="720p"):
    """Main function to create video from text"""
    print(f"Creating video with genre '{genre}' and title '{title}'")
    print(f"Story length: {len(story)} characters")
    
    # Create temp directory
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Clean the input text
        story = clean_text(story)
        
        # Step 1: Convert text to speech
        print("Converting text to speech...")
        speech_file = asyncio.run(text_to_speech(story))
        display(Audio(speech_file))
        print("Speech generated successfully.")
        
        # Step 2: Extract timestamps
        print("Extracting word timestamps...")
        word_timestamps = extract_timestamps(speech_file)
        print(f"Extracted timestamps for {len(word_timestamps)} words")
        
        # Step 3: Generate keywords based on story and genre
        keywords = get_keywords_from_text(story)
        if genre:
            keywords.append(genre)
        print(f"Generated keywords: {keywords}")
        
        # Step 4: Fetch videos and GIFs
        print("Fetching media...")
        
        # Step A: Fetch videos from Pexels
        videos = []
        for keyword in keywords:
            videos.extend(fetch_pexels_videos(f"{keyword} {genre}".strip(), pexels_api_key, per_page=3))
        
        # Step B: Fetch GIFs from Tenor as backup
        gifs = []
        for keyword in keywords:
            gifs.extend(fetch_tenor_gifs(f"{keyword} {genre}".strip(), tenor_api_key, limit=3))
        
        # Combine and shuffle media sources
        media_sources = videos + gifs
        random.shuffle(media_sources)
        
        if not media_sources:
            return "Error: No media found. Please check API keys or try different keywords."
        
        print(f"Found {len(media_sources)} media items")
        
        # Step 5: Download media files
        print("Downloading media files...")
        media_files = []
        for i, media in enumerate(media_sources):
            extension = "mp4" if "video" in media.get("url", "") else "gif"
            output_path = os.path.join(temp_dir, f"media_{i}.{extension}")
            if download_media(media["url"], output_path):
                media_files.append({
                    "path": output_path,
                    "type": extension,
                    "width": media.get("width", 720),
                    "height": media.get("height", 480)
                })
        
        print(f"Downloaded {len(media_files)} media files")
        
        # Step 6: Set video resolution
        if resolution == "720p":
            target_width, target_height = 1280, 720
        elif resolution == "1080p":
            target_width, target_height = 1920, 1080
        else:  # Default to 480p
            target_width, target_height = 854, 480
        
        # Step 7: Create clips from media files
        print("Creating video clips...")
        audio_clip = AudioFileClip(speech_file)
        audio_duration = audio_clip.duration
        
        # Group words into phrases for better media sync
        phrases = group_words_into_phrases(word_timestamps)
        print(f"Created {len(phrases)} phrases for synchronization")
        
        # Create video clips
        video_clips = []
        for i, phrase in enumerate(phrases):
            # Select media file (cycling through available media)
            media_index = i % len(media_files)
            media = media_files[media_index]
            
            print(f"Processing phrase {i+1}/{len(phrases)}: '{phrase['text'][:30]}...'")
            
            if media["type"] == "mp4":
                clip = VideoFileClip(media["path"])
            else:  # GIF
                clip = VideoFileClip(media["path"])
            
            # Resize to target resolution
            clip = clip.resize(width=target_width, height=target_height)
            
            # Set duration and position
            clip = clip.subclip(0, min(phrase["duration"], clip.duration))
            if clip.duration < phrase["duration"]:
                # Loop the clip if it's shorter than needed
                clip = loop_clip(clip, phrase["duration"])
            
            # Set start time
            clip = clip.set_start(phrase["start"])
            
            # Add subtitles if requested
            if include_subtitles:
                clip = add_subtitle_to_clip(clip, phrase["text"], phrase["start"], phrase["end"])
            
            video_clips.append(clip)
        
        print("Combining clips into final video...")
        # Step 8: Combine clips into final video
        final_clip = CompositeVideoClip(video_clips, size=(target_width, target_height))
        final_clip = final_clip.set_duration(audio_duration)
        
        # Add audio
        final_clip = final_clip.set_audio(audio_clip)
        
        # Add title at the beginning
        if title:
            print(f"Adding title: '{title}'")
            title_clip = TextClip(title, fontsize=70, color='white', bg_color='black',
                                size=(target_width, target_height), method='caption')
            title_clip = title_clip.set_duration(3)
            final_clip = CompositeVideoClip([title_clip, final_clip.set_start(3)])
            final_clip = final_clip.set_duration(audio_duration + 3)
        
        # Export video
        output_file = os.path.join(OUTPUT_DIR, f"video_{int(time.time())}.mp4")
        print(f"Rendering final video to {output_file}...")
        final_clip.write_videofile(output_file, fps=24, codec='libx264')
        
        # Clean up
        shutil.rmtree(temp_dir)
        print("Video creation completed!")
        
        # Return path to final video
        return output_file
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"Error creating video: {e}")
        return f"Error: {str(e)}"

# Simple test to verify the module works
if __name__ == "__main__":
    print("YouTube Video Creator module loaded successfully.") 