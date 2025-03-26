import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from typing import Optional, Dict, List
import pandas as pd
import os
from pathlib import Path
import ffmpeg

class VideoEditor:
    def __init__(self):
        """Initialize VideoEditor with default settings"""
        self.output_dir = "final_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Default video settings
        self.width = 1920
        self.height = 1080
        self.fps = 30
    
    def load_media_files(
        self,
        csv_path: str,
        media_dir: str,
        audio_path: str,
        intro_path: Optional[str] = None,
        outro_path: Optional[str] = None,
        bg_music_path: Optional[str] = None
    ) -> Dict:
        """Load and validate all media files"""
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file not found: {csv_path}")
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
            
        media_files = {
            "csv": pd.read_csv(csv_path),
            "audio": AudioFileClip(audio_path),
            "clips": {},
            "intro": None,
            "outro": None,
            "bg_music": None
        }
        
        # Load optional components
        if intro_path and os.path.exists(intro_path):
            media_files["intro"] = VideoFileClip(intro_path)
            print("✅ Loaded intro video")
            
        if outro_path and os.path.exists(outro_path):
            media_files["outro"] = VideoFileClip(outro_path)
            print("✅ Loaded outro video")
            
        if bg_music_path and os.path.exists(bg_music_path):
            media_files["bg_music"] = AudioFileClip(bg_music_path)
            print("✅ Loaded background music")
        
        # Load all word clips
        print("🎬 Loading word clips...")
        for _, row in media_files["csv"].iterrows():
            s_no = row["s.no"]
            duration = row["duration"]
            
            # Try both mp4 and gif extensions
            for ext in [".mp4", ".gif"]:
                clip_path = os.path.join(media_dir, f"{s_no}{ext}")
                if os.path.exists(clip_path):
                    clip = VideoFileClip(clip_path)
                    
                    # Loop clip if shorter than required duration
                    if clip.duration < duration:
                        clip = clip.loop(duration=duration)
                    else:
                        clip = clip.subclip(0, duration)
                    
                    # Resize to match output dimensions
                    clip = clip.resize((self.width, self.height))
                    media_files["clips"][s_no] = clip
                    break
        
        print(f"✅ Loaded {len(media_files['clips'])} word clips")
        return media_files
    
    def create_final_video(
        self,
        media_files: Dict,
        bg_music_volume: float = 0.1,
        output_filename: str = "final_video.mp4"
    ) -> str:
        """Create the final video with all components"""
        try:
            print("\n🔍 Verifying input files:")
            print(f"Number of clips: {len(media_files['clips'])}")
            print(f"Audio file: {media_files['audio']}")
            if media_files['bg_music']:
                print(f"Background music: {media_files['bg_music']}")

            # Check first clip
            first_clip = next(iter(media_files['clips'].values()))
            print(f"\n📊 Sample clip info:")
            print(f"Path: {first_clip['path']}")
            print(f"Duration: {first_clip['duration']}")
            
            clips: List[VideoFileClip] = []
            current_time = 0
            
            # Add intro if available
            if media_files["intro"]:
                intro = media_files["intro"].resize((self.width, self.height))
                clips.append(intro)
                current_time += intro.duration
            
            # Add word clips in sequence
            for _, row in media_files["csv"].iterrows():
                s_no = row["s.no"]
                if s_no in media_files["clips"]:
                    clip = media_files["clips"][s_no].set_start(current_time)
                    clips.append(clip)
                    current_time += clip.duration
            
            # Add outro if available
            if media_files["outro"]:
                outro = media_files["outro"].resize((self.width, self.height))
                outro = outro.set_start(current_time)
                clips.append(outro)
                current_time += outro.duration
            
            # Combine all video clips
            print("🎬 Combining video clips...")
            final_video = mp.CompositeVideoClip(clips)
            
            # Add main audio
            print("🎵 Adding main audio...")
            final_video = final_video.set_audio(media_files["audio"])
            
            # Add background music if available
            if media_files["bg_music"]:
                print("🎵 Adding background music...")
                # Loop bg music if shorter than video
                bg_music = media_files["bg_music"]
                if bg_music.duration < final_video.duration:
                    bg_music = bg_music.loop(duration=final_video.duration)
                else:
                    bg_music = bg_music.subclip(0, final_video.duration)
                
                # Set volume and mix with main audio
                bg_music = bg_music.volumex(bg_music_volume)
                final_audio = mp.CompositeAudioClip([
                    final_video.audio,
                    bg_music
                ])
                final_video = final_video.set_audio(final_audio)
            
            # Write final video
            output_path = os.path.join(self.output_dir, output_filename)
            print("💾 Creating final video...")
            try:
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            except ffmpeg.Error as e:
                print("⚠️ FFMPEG Error Details:")
                print("stdout:", e.stdout.decode('utf8'))
                print("stderr:", e.stderr.decode('utf8'))
                raise Exception("FFMPEG processing failed. See error details above.")
            
            print(f"✅ Final video saved to: {output_path}")
            return output_path
            
        finally:
            # Clean up clips
            for clip in clips:
                clip.close()
            media_files["audio"].close()
            if media_files["bg_music"]:
                media_files["bg_music"].close()

#@title Video Editor Settings {display-mode: "form"}

#@markdown ### Input Files
CSV_PATH = "timestamps_adjusted.csv" #@param {type:"string"}
MEDIA_DIR = "downloaded_media" #@param {type:"string"}
AUDIO_PATH = "generated_audio/test1.mp3" #@param {type:"string"}

#@markdown ### Optional Components
INTRO_PATH = "" #@param {type:"string"}
OUTRO_PATH = "" #@param {type:"string"}
BG_MUSIC_PATH = "" #@param {type:"string"}

#@markdown ### Output Settings
OUTPUT_FILENAME = "final_video.mp4" #@param {type:"string"}
BG_MUSIC_VOLUME = 0.1 #@param {type:"number"}
#@markdown Volume of background music (0.0 to 1.0)

def run_video_editor():
    """Run the video editor with the configured settings"""
    try:
        # Initialize editor
        editor = VideoEditor()
        
        # Load all media files
        print("📂 Loading media files...")
        media_files = editor.load_media_files(
            csv_path=CSV_PATH,
            media_dir=MEDIA_DIR,
            audio_path=AUDIO_PATH,
            intro_path=INTRO_PATH if INTRO_PATH else None,
            outro_path=OUTRO_PATH if OUTRO_PATH else None,
            bg_music_path=BG_MUSIC_PATH if BG_MUSIC_PATH else None
        )
        
        # Create final video
        output_path = editor.create_final_video(
            media_files=media_files,
            bg_music_volume=BG_MUSIC_VOLUME,
            output_filename=OUTPUT_FILENAME
        )
        
        print("\n🎉 Video creation completed!")
        print(f"📂 Final video saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    run_video_editor()