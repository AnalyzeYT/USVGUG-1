"""
YouTube Video Creator Module
This module contains all functions needed to create a complete video from a story
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
import warnings
from PIL import Image
import moviepy.editor as mp
import torch
import whisperx

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Constants
VOICE_NAME = "en-US-ChristopherNeural"
VOICE_RATE = "+0%"
VOICE_VOLUME = "+0%"
OUTPUT_DIR = "outputs"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ TITLE TO SCRIPT GENERATION ------------------

def expand_title_to_script(title, genre, length=500):
    """
    Generate a script from a title and genre
    Args:
        title: The title to expand
        genre: The genre of the video
        length: Approximate length of script in characters
    Returns:
        A script based on the title
    """
    # For now, this is a placeholder function.
    # In a real implementation, you would use an AI model or API to generate a script
    print(f"Title expansion is not implemented. Please provide your own script.")
    return None

# ------------------ SCRIPT TO AUDIO CONVERSION ------------------

def clean_text(text):
    """Clean text for TTS processing"""
    # Replace newlines with spaces
    text = re.sub(r'\n+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
    return text.strip()

async def _generate_audio(text, output_file, voice_name=VOICE_NAME, rate=VOICE_RATE, volume=VOICE_VOLUME):
    """Generate audio from text using edge-tts"""
    communicate = edge_tts.Communicate(text, voice_name, rate=rate, volume=volume)
    await communicate.save(output_file)

def text_to_speech(text, output_file):
    """
    Convert text to speech using Edge TTS
    Args:
        text: The text to convert to speech
        output_file: Output audio file path
    Returns:
        Path to the generated audio file
    """
    text = clean_text(text)
    
    try:
        # Run the async function
        asyncio.run(_generate_audio(text, output_file))
        return output_file
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

# ------------------ AUDIO TO TRANSCRIPT CSV ------------------

def load_whisperx_model(device="cuda", compute_type="float16"):
    """
    Load the WhisperX model for timestamp extraction
    Args:
        device: The device to run the model on ('cuda' or 'cpu')
        compute_type: The computation type ('float16' or 'float32')
    Returns:
        Loaded WhisperX model
    """
    try:
        # Check if CUDA is available when device is set to cuda
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
            compute_type = "float32"  # Use float32 for CPU
        
        # For CPU, always use float32
        if device == "cpu":
            compute_type = "float32"
        
        print(f"Loading WhisperX model on {device} with {compute_type}...")
        model = whisperx.load_model("tiny", device, compute_type=compute_type)
        print("WhisperX model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading WhisperX model: {e}")
        # Fallback to CPU with float32
        try:
            print("Attempting to load model on CPU with float32...")
            model = whisperx.load_model("tiny", "cpu", compute_type="float32")
            print("WhisperX model loaded successfully on CPU")
            return model
        except Exception as e2:
            print(f"Critical error loading WhisperX model: {e2}")
            return None

def extract_timestamps(audio_file, device="cuda", compute_type="float16"):
    """
    Extract word-level timestamps from audio
    Args:
        audio_file: Path to the audio file
        device: Device to run the model on
        compute_type: Computation type for the model
    Returns:
        Dataframe with word-level timestamps
    """
    try:
        # Load model
        model = load_whisperx_model(device, compute_type)
        if model is None:
            return None
        
        # Transcribe audio
        print("Transcribing audio with WhisperX...")
        result = model.transcribe(audio_file)
        
        # Use alignment model to get word-level timestamps
        print("Aligning words for timestamps...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_file, device)
        
        # Extract word information
        word_timestamps = []
        for segment in result["segments"]:
            for word in segment["words"]:
                word_timestamps.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"]
                })
        
        return word_timestamps
    except Exception as e:
        print(f"Error extracting timestamps: {e}")
        return None

# ------------------ TRANSCRIPT TO VIDEO/GIF MATCHING ------------------

def extract_keywords(text, genre):
    """
    Extract keywords from text for media search
    Args:
        text: The text to extract keywords from
        genre: The genre to include in keywords
    Returns:
        List of keywords
    """
    # Simple keyword extraction (can be improved with NLP)
    # Remove common words and keep only meaningful ones
    common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of", 
                  "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", 
                  "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
                  "can", "could", "may", "might", "must", "ought", "i", "you", "he", "she", 
                  "it", "we", "they", "me", "him", "her", "us", "them"}
    
    # Split text into words and filter
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if word not in common_words and len(word) > 3]
    
    # Add genre as a keyword
    if genre and genre.lower() not in [k.lower() for k in keywords]:
        keywords.append(genre.lower())
    
    # Deduplicate and return
    return list(set(keywords))

def group_words_into_phrases(word_timestamps, max_phrase_duration=5.0):
    """
    Group words into phrases for media search
    Args:
        word_timestamps: List of word timestamps
        max_phrase_duration: Maximum duration of a phrase in seconds
    Returns:
        List of phrases with start and end times
    """
    phrases = []
    current_phrase = []
    current_start = None
    current_end = None
    current_text = ""
    
    for word_info in word_timestamps:
        word = word_info["word"].strip()
        start = word_info["start"]
        end = word_info["end"]
        
        # Start a new phrase
        if not current_phrase:
            current_phrase = [word_info]
            current_start = start
            current_end = end
            current_text = word
            continue
        
        # If adding this word would exceed the max duration, start a new phrase
        if end - current_start > max_phrase_duration:
            phrases.append({
                "text": current_text,
                "start": current_start,
                "end": current_end,
                "words": current_phrase
            })
            current_phrase = [word_info]
            current_start = start
            current_end = end
            current_text = word
        else:
            # Otherwise, add to current phrase
            current_phrase.append(word_info)
            current_end = end
            current_text += " " + word
    
    # Add the last phrase
    if current_phrase:
        phrases.append({
            "text": current_text,
            "start": current_start,
            "end": current_end,
            "words": current_phrase
        })
    
    return phrases

def fetch_pexels_videos(query, api_key, resolution="720p", per_page=10, orientation="landscape"):
    """
    Fetch videos from Pexels API based on keywords
    Args:
        query: Search query
        api_key: Pexels API key
        resolution: Desired resolution
        per_page: Number of results to fetch
        orientation: Video orientation
    Returns:
        List of video URLs
    """
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}&orientation={orientation}"
    headers = {"Authorization": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        videos = []
        for video in data.get("videos", []):
            # Get video file based on resolution
            video_files = video.get("video_files", [])
            for file in video_files:
                if resolution in file.get("quality", "").lower():
                    videos.append({
                        "url": file.get("link"),
                        "width": file.get("width"),
                        "height": file.get("height"),
                        "duration": video.get("duration")
                    })
                    break
        
        return videos
    except Exception as e:
        print(f"Error fetching Pexels videos: {e}")
        return []

def fetch_tenor_gifs(query, api_key, limit=10):
    """
    Fetch GIFs from Tenor API based on keywords
    Args:
        query: Search query
        api_key: Tenor API key
        limit: Number of results to fetch
    Returns:
        List of GIF URLs
    """
    url = f"https://g.tenor.com/v1/search?q={query}&key={api_key}&limit={limit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        gifs = []
        for result in data.get("results", []):
            media = result.get("media", [{}])[0]
            gif = media.get("gif", {})
            gifs.append({
                "url": gif.get("url"),
                "width": gif.get("dims", [0, 0])[0],
                "height": gif.get("dims", [0, 0])[1]
            })
        
        return gifs
    except Exception as e:
        print(f"Error fetching Tenor GIFs: {e}")
        return []

def download_media(url, output_file):
    """
    Download media from URL
    Args:
        url: URL of the media
        output_file: Output file path
    Returns:
        Path to the downloaded file
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        return output_file
    except Exception as e:
        print(f"Error downloading media: {e}")
        return None

# ------------------ GIF TO VIDEO CONVERSION ------------------

def gif_to_video(gif_path, output_path, duration=None, resolution=None):
    """
    Convert GIF to video
    Args:
        gif_path: Path to the GIF file
        output_path: Path for the output video
        duration: Duration of the video
        resolution: Target resolution (width, height)
    Returns:
        Path to the output video
    """
    try:
        # Load the GIF as clip
        clip = mp.VideoFileClip(gif_path)
        
        # Set the duration
        if duration:
            clip = clip.loop(duration=duration)
        
        # Resize if resolution is specified
        if resolution:
            clip = clip.resize(resolution)
        
        # Write to file
        clip.write_videofile(output_path, codec='libx264', audio=False, fps=24)
        clip.close()
        
        return output_path
    except Exception as e:
        print(f"Error converting GIF to video: {e}")
        return None

def loop_clip_to_duration(clip, target_duration):
    """
    Loop a clip to reach the target duration
    Args:
        clip: MoviePy VideoClip
        target_duration: Target duration in seconds
    Returns:
        Looped clip
    """
    if clip.duration >= target_duration:
        return clip.subclip(0, target_duration)
    
    # Calculate how many times we need to loop
    loops = int(np.ceil(target_duration / clip.duration))
    return mp.concatenate_videoclips([clip] * loops).subclip(0, target_duration)

# ------------------ SUBTITLE GENERATION ------------------

def add_subtitles(clip, word_timestamps, font_size=40, color='white', stroke_color='black', stroke_width=2):
    """
    Add subtitles to a video clip
    Args:
        clip: MoviePy VideoClip
        word_timestamps: List of word timestamps
        font_size: Font size for subtitles
        color: Font color
        stroke_color: Stroke color
        stroke_width: Stroke width
    Returns:
        Clip with subtitles
    """
    subtitle_clips = []
    
    for word_info in word_timestamps:
        word = word_info["word"].strip()
        start = word_info["start"]
        end = word_info["end"]
        duration = end - start
        
        # Create TextClip for the word
        txt_clip = mp.TextClip(word, fontsize=font_size, color=color, stroke_color=stroke_color, 
                             stroke_width=stroke_width, font='Arial-Bold', method='caption')
        
        # Set position to bottom center
        txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(duration).set_start(start)
        
        subtitle_clips.append(txt_clip)
    
    # Composite all subtitle clips over the main clip
    return mp.CompositeVideoClip([clip] + subtitle_clips)

# ------------------ FINAL VIDEO CREATION ------------------

def create_video(story, genre, title, pexels_api_key, tenor_api_key, include_subtitles=True, 
               resolution="720p", progress_callback=None):
    """
    Create a complete video from a story
    Args:
        story: The story text
        genre: The genre of the video
        title: The title of the video
        pexels_api_key: Pexels API key
        tenor_api_key: Tenor API key
        include_subtitles: Whether to include subtitles
        resolution: Video resolution (480p, 720p, 1080p)
        progress_callback: Callback function for progress updates
    Returns:
        Path to the output video file
    """
    start_time = time.time()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Set resolution parameters
        res_map = {
            "480p": (854, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080)
        }
        video_resolution = res_map.get(resolution, (1280, 720))
        
        # Callback for progress
        def update_progress(message):
            if progress_callback:
                progress_callback(message)
            else:
                print(message)
        
        # Step 1: Convert text to speech
        update_progress("Step 1/5: Converting text to speech...")
        audio_file = os.path.join(temp_dir, "audio.mp3")
        speech_result = text_to_speech(story, audio_file)
        
        if not speech_result:
            return "Error: Failed to generate speech audio"
        
        # Step 2: Extract word-level timestamps
        update_progress("Step 2/5: Extracting word-level timestamps...")
        word_timestamps = extract_timestamps(audio_file)
        
        if not word_timestamps:
            return "Error: Failed to extract timestamps"
        
        # Step 3: Group words into phrases for media search
        update_progress("Step 3/5: Analyzing text for media matching...")
        phrases = group_words_into_phrases(word_timestamps)
        
        # Load audio clip to get duration
        audio_clip = mp.AudioFileClip(audio_file)
        total_duration = audio_clip.duration
        
        # Step 4: Fetch and prepare media for each phrase
        update_progress("Step 4/5: Fetching and preparing media...")
        video_clips = []
        
        for i, phrase in enumerate(phrases):
            update_progress(f"  Processing phrase {i+1}/{len(phrases)}...")
            phrase_duration = phrase["end"] - phrase["start"]
            
            # Extract keywords from the phrase
            keywords = extract_keywords(phrase["text"], genre)
            search_query = " ".join(keywords[:3]) if keywords else genre
            
            # Try to get a video first, fallback to GIF
            videos = fetch_pexels_videos(search_query, pexels_api_key, resolution=resolution)
            
            if videos:
                # Use a random video from results
                video_info = random.choice(videos)
                video_path = os.path.join(temp_dir, f"video_{i}.mp4")
                
                if download_media(video_info["url"], video_path):
                    # Load the video and set duration
                    video_clip = mp.VideoFileClip(video_path)
                    video_clip = loop_clip_to_duration(video_clip, phrase_duration)
                    
                    # Resize to target resolution
                    video_clip = video_clip.resize(video_resolution)
                    
                    # Set start time
                    video_clip = video_clip.set_start(phrase["start"])
                    
                    video_clips.append(video_clip)
            else:
                # Fallback to GIFs
                gifs = fetch_tenor_gifs(search_query, tenor_api_key)
                
                if gifs:
                    gif_info = random.choice(gifs)
                    gif_path = os.path.join(temp_dir, f"gif_{i}.gif")
                    
                    if download_media(gif_info["url"], gif_path):
                        # Convert GIF to video
                        gif_video_path = os.path.join(temp_dir, f"gif_video_{i}.mp4")
                        if gif_to_video(gif_path, gif_video_path, duration=phrase_duration):
                            # Load the video
                            gif_clip = mp.VideoFileClip(gif_video_path)
                            
                            # Resize to target resolution
                            gif_clip = gif_clip.resize(video_resolution)
                            
                            # Set start time
                            gif_clip = gif_clip.set_start(phrase["start"])
                            
                            video_clips.append(gif_clip)
        
        # Step 5: Combine everything into the final video
        update_progress("Step 5/5: Creating final video...")
        
        # Create a black background clip for the entire duration
        background = mp.ColorClip(video_resolution, color=(0, 0, 0), duration=total_duration)
        
        # Combine all video clips on top of the background
        if video_clips:
            final_clip = mp.CompositeVideoClip([background] + video_clips)
        else:
            final_clip = background
        
        # Add subtitles if requested
        if include_subtitles:
            update_progress("  Adding subtitles...")
            final_clip = add_subtitles(final_clip, word_timestamps)
        
        # Add audio
        final_clip = final_clip.set_audio(audio_clip)
        
        # Create output filename based on title
        safe_title = re.sub(r'[^a-zA-Z0-9_]', '_', title).lower()
        timestamp = int(time.time())
        output_filename = f"{safe_title}_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Write the final video
        update_progress("  Rendering final video...")
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)
        
        # Cleanup
        for clip in video_clips:
            clip.close()
        audio_clip.close()
        final_clip.close()
        
        end_time = time.time()
        update_progress(f"Video created successfully in {end_time - start_time:.2f} seconds!")
        
        return output_path
    
    except Exception as e:
        return f"Error: {str(e)}"
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

# Simple test to verify the module works
if __name__ == "__main__":
    print("YouTube Video Creator module loaded successfully.") 