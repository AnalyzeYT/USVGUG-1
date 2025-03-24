import os
import re
import time
import random
import requests
import gradio as gr
import numpy as np
import edge_tts
import asyncio
import tempfile
import shutil
from PIL import Image
import platform
import warnings
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='YouTube Video Creator')
parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use for model (cuda/cpu)')
parser.add_argument('--compute_type', type=str, choices=['float16', 'float32'], help='Compute type for model')
parser.add_argument('--share', action='store_true', help='Create a public URL')
args = parser.parse_args()

# Suppress ALSA warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab environment")
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    # Set up MoviePy for Colab
    import subprocess
    try:
        subprocess.run(['mkdir', '-p', os.path.expanduser('~/.config/moviepy')])
        with open(os.path.expanduser('~/.config/moviepy/moviepy.conf'), 'w') as f:
            f.write('{"FFMPEG_BINARY": "ffmpeg", "IMAGEMAGICK_BINARY": "convert"}')
    except Exception as e:
        print(f"Warning: Could not configure MoviePy: {e}")

# Now import MoviePy after configuring it
from moviepy.editor import *
import whisperx
import torch

# API Keys
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")  # Get from environment variable
TENOR_API_KEY = os.environ.get("TENOR_API_KEY", "")   # Get from environment variable

# Constants
VOICE = "en-US-AnaNeural"  # Default voice for Edge TTS
OUTPUT_DIR = "outputs"     # Directory to store temporary and output files
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize WhisperX for timestamp extraction
device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
# Use compute_type float32 for Colab compatibility or if specified by args
compute_type = args.compute_type if args.compute_type else ("float32" if IN_COLAB or device == "cpu" else "float16")
print(f"Using device: {device}, compute_type: {compute_type}")

try:
    whisper_model = whisperx.load_model("base", device, compute_type=compute_type)
    print("WhisperX model loaded successfully")
except Exception as e:
    print(f"Error loading WhisperX model: {e}")
    print("Attempting fallback configuration...")
    try:
        # Fallback to CPU if needed
        device = "cpu"
        compute_type = "float32"
        whisper_model = whisperx.load_model("base", device, compute_type=compute_type)
        print("WhisperX model loaded with fallback configuration")
    except Exception as e:
        print(f"Critical error loading WhisperX model: {e}")
        raise

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

def fetch_pexels_videos(query, per_page=10):
    """Fetch videos from Pexels API based on query"""
    if not PEXELS_API_KEY:
        print("PEXELS_API_KEY not found in environment variables")
        return []
    
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}"
    headers = {"Authorization": PEXELS_API_KEY}
    
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
            return videos
        else:
            print(f"Pexels API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching Pexels videos: {e}")
        return []

def fetch_tenor_gifs(query, limit=10):
    """Fetch GIFs from Tenor API based on query"""
    if not TENOR_API_KEY:
        print("TENOR_API_KEY not found in environment variables")
        return []
    
    url = f"https://tenor.googleapis.com/v2/search?q={query}&key={TENOR_API_KEY}&limit={limit}"
    
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

def create_video(story, genre, title, include_subtitles=True, resolution="720p"):
    """Main function to create video from text"""
    # Create temp directory
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Clean the input text
        story = clean_text(story)
        
        # Step 1: Convert text to speech
        speech_file = asyncio.run(text_to_speech(story))
        
        # Step 2: Extract timestamps
        word_timestamps = extract_timestamps(speech_file)
        
        # Step 3: Generate keywords based on story and genre
        keywords = get_keywords_from_text(story)
        if genre:
            keywords.append(genre)
        
        # Step A: Fetch videos from Pexels
        videos = []
        for keyword in keywords:
            videos.extend(fetch_pexels_videos(f"{keyword} {genre}".strip(), per_page=3))
        
        # Step B: Fetch GIFs from Tenor as backup
        gifs = []
        for keyword in keywords:
            gifs.extend(fetch_tenor_gifs(f"{keyword} {genre}".strip(), limit=3))
        
        # Combine and shuffle media sources
        media_sources = videos + gifs
        random.shuffle(media_sources)
        
        if not media_sources:
            return "Error: No media found. Please check API keys or try different keywords."
        
        # Step 4: Download media files
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
        
        # Step 5: Set video resolution
        if resolution == "720p":
            target_width, target_height = 1280, 720
        elif resolution == "1080p":
            target_width, target_height = 1920, 1080
        else:  # Default to 480p
            target_width, target_height = 854, 480
        
        # Step 6: Create clips from media files
        audio_clip = AudioFileClip(speech_file)
        audio_duration = audio_clip.duration
        
        # Group words into phrases for better media sync
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
        
        # Create video clips
        video_clips = []
        for i, phrase in enumerate(phrases):
            # Select media file (cycling through available media)
            media_index = i % len(media_files)
            media = media_files[media_index]
            
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
        
        # Step 7: Combine clips into final video
        final_clip = CompositeVideoClip(video_clips, size=(target_width, target_height))
        final_clip = final_clip.set_duration(audio_duration)
        
        # Add audio
        final_clip = final_clip.set_audio(audio_clip)
        
        # Add title at the beginning
        if title:
            title_clip = TextClip(title, fontsize=70, color='white', bg_color='black',
                                  size=(target_width, target_height), method='caption')
            title_clip = title_clip.set_duration(3)
            final_clip = CompositeVideoClip([title_clip, final_clip.set_start(3)])
            final_clip = final_clip.set_duration(audio_duration + 3)
        
        # Export video
        output_file = os.path.join(OUTPUT_DIR, f"video_{int(time.time())}.mp4")
        final_clip.write_videofile(output_file, fps=24, codec='libx264')
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return output_file
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"Error creating video: {e}")
        return f"Error: {str(e)}"

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

# Gradio Interface
def create_video_interface(story, genre, title, include_subtitles, resolution):
    """Gradio interface function"""
    if not story:
        return "Please provide a story"
    
    if not title:
        title = "My Video"  # Default title
    
    # Create the video
    result = create_video(story, genre, title, include_subtitles, resolution)
    
    if isinstance(result, str) and result.startswith("Error"):
        return result
    else:
        return result

# Create Gradio interface
with gr.Blocks(title="YouTube Video Creator") as demo:
    gr.Markdown("# One-Click YouTube Video Creator")
    gr.Markdown("Upload or paste a story, select a genre and title, and generate a complete video.")
    
    with gr.Row():
        with gr.Column():
            story_input = gr.Textbox(label="Your Story", lines=10, placeholder="Paste your story here...")
            genre_input = gr.Dropdown(
                label="Genre", 
                choices=["Nature", "Technology", "Education", "Entertainment", "Business", "Travel", "Sports", "Food"],
                value="Nature"
            )
            title_input = gr.Textbox(label="Video Title", placeholder="Enter a catchy title...")
            
            with gr.Row():
                subtitles_input = gr.Checkbox(label="Include Subtitles", value=True)
                resolution_input = gr.Radio(
                    label="Resolution", 
                    choices=["480p", "720p", "1080p"],
                    value="720p"
                )
            
            create_btn = gr.Button("Create Video", variant="primary")
        
        with gr.Column():
            output = gr.Video(label="Generated Video")
    
    create_btn.click(
        fn=create_video_interface,
        inputs=[story_input, genre_input, title_input, subtitles_input, resolution_input],
        outputs=output
    )
    
    gr.Markdown("""
    ## How it works
    
    1. Your story is converted to speech using Edge TTS
    2. WhisperX extracts timestamps for each word
    3. Relevant images and videos are fetched based on your story and genre
    4. Media clips are synchronized with the audio to create a complete video
    
    ## Notes
    
    - Processing may take a few minutes depending on story length
    - For best results, provide a clear and detailed story
    - You'll need to set up API keys for Pexels and Tenor in your Hugging Face Space
    """)

if __name__ == "__main__":
    demo.launch(share=args.share) 