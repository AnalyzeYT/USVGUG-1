import requests
import pandas as pd
import os
from typing import Optional, Dict, Set, Literal
from urllib.parse import urlparse
from pathlib import Path
import json

class MediaFetcher:
    def __init__(self, pexels_api_key: str = "", tenor_api_key: str = ""):
        """Initialize MediaFetcher with API keys"""
        self.pexels_api_key = pexels_api_key
        self.tenor_api_key = tenor_api_key
        self.used_media: Set[str] = set()  # Track used media to avoid duplicates
        
        # Create necessary directories
        self.media_dir = "downloaded_media"
        os.makedirs(self.media_dir, exist_ok=True)
    
    def adjust_timestamps(self, csv_path: str, min_duration: float) -> pd.DataFrame:
        """
        Adjust timestamps in CSV file to ensure continuous timing and filter short durations
        
        Args:
            csv_path: Path to the CSV file
            min_duration: Minimum duration threshold in seconds
        """
        print("📊 Adjusting timestamps...")
        df = pd.read_csv(csv_path)
        initial_count = len(df)
        
        # Adjust start times and durations
        for i in range(1, len(df)):
            df.loc[i, 'start'] = df.loc[i-1, 'end']
            df.loc[i, 'duration'] = df.loc[i, 'end'] - df.loc[i, 'start']
        
        # Filter out words with duration less than minimum threshold
        df_filtered = df[df['duration'] >= min_duration].copy()
        df_filtered.reset_index(drop=True, inplace=True)
        
        # Renumber the s.no column
        df_filtered['s.no'] = range(1, len(df_filtered) + 1)
        
        # Save adjusted and filtered CSV
        adjusted_csv = csv_path.replace('.csv', '_adjusted.csv')
        df_filtered.to_csv(adjusted_csv, index=False)
        
        removed_count = initial_count - len(df_filtered)
        print(f"✅ Adjusted timestamps saved to: {adjusted_csv}")
        print(f"ℹ️ Removed {removed_count} words with duration < {min_duration} seconds")
        
        return df_filtered
    
    def search_pexels(self, query: str) -> Optional[str]:
        """Search Pexels API for a video"""
        if not self.pexels_api_key:
            raise ValueError("Pexels API key not provided")
        
        headers = {"Authorization": self.pexels_api_key}
        url = f"https://api.pexels.com/videos/search?query={query}&per_page=5"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("videos"):
                return None
            
            # Find a video that hasn't been used yet
            for video in data["videos"]:
                for video_file in video["video_files"]:
                    if (
                        video_file["link"] not in self.used_media
                        and video_file["file_type"] == "video/mp4"
                        and video_file["width"] <= 1920  # Limit to HD or smaller
                    ):
                        return video_file["link"]
            
            return None
        except Exception as e:
            print(f"⚠️ Pexels API error: {str(e)}")
            return None
    
    def search_tenor(self, query: str) -> Optional[str]:
        """Search Tenor API for a GIF"""
        if not self.tenor_api_key:
            raise ValueError("Tenor API key not provided")
        
        url = f"https://tenor.googleapis.com/v2/search?q={query}&key={self.tenor_api_key}&limit=5"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("results"):
                return None
            
            # Find a GIF that hasn't been used yet
            for result in data["results"]:
                gif_url = result["media_formats"]["gif"]["url"]
                if gif_url not in self.used_media:
                    return gif_url
            
            return None
        except Exception as e:
            print(f"⚠️ Tenor API error: {str(e)}")
            return None
    
    def download_media(self, url: str, output_path: str) -> bool:
        """Download media file from URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.used_media.add(url)  # Mark as used
            return True
        except Exception as e:
            print(f"⚠️ Download error: {str(e)}")
            return False
    
    def process_csv(
        self,
        csv_path: str,
        media_type: Literal["pexels", "tenor"],
        api_key: str,
        min_duration: float
    ) -> Dict[int, str]:
        """
        Process CSV file and download media for each word
        
        Args:
            csv_path: Path to the CSV file
            media_type: Type of media to fetch ("pexels" or "tenor")
            api_key: API key for the selected service
            min_duration: Minimum duration threshold in seconds
        
        Returns:
            Dictionary mapping s.no to media file paths
        """
        # Set the appropriate API key
        if media_type == "pexels":
            self.pexels_api_key = api_key
        else:
            self.tenor_api_key = api_key
        
        # First adjust the timestamps and filter short durations
        df = self.adjust_timestamps(csv_path, min_duration)
        
        # Track processed words and their media paths
        word_to_media: Dict[str, str] = {}
        sno_to_media: Dict[int, str] = {}
        
        print(f"🎬 Downloading media from {media_type.title()}...")
        for _, row in df.iterrows():
            word = row['word'].strip().lower()
            s_no = row['s.no']
            
            # Skip if we already have media for this word
            if word in word_to_media:
                sno_to_media[s_no] = word_to_media[word]
                continue
            
            # Search and download new media
            media_url = (
                self.search_pexels(word) if media_type == "pexels"
                else self.search_tenor(word)
            )
            
            if media_url:
                extension = ".mp4" if media_type == "pexels" else ".gif"
                output_path = os.path.join(
                    self.media_dir,
                    f"{s_no}{extension}"
                )
                
                if self.download_media(media_url, output_path):
                    word_to_media[word] = output_path
                    sno_to_media[s_no] = output_path
                    print(f"✅ Downloaded media for word '{word}' (#{s_no})")
                else:
                    print(f"❌ Failed to download media for word '{word}' (#{s_no})")
            else:
                print(f"⚠️ No media found for word '{word}' (#{s_no})")
        
        # Save mapping to JSON for reference
        mapping_file = os.path.join(self.media_dir, "media_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump({
                "word_to_media": word_to_media,
                "sno_to_media": {str(k): v for k, v in sno_to_media.items()},
                "settings": {
                    "min_duration": min_duration,
                    "media_type": media_type
                }
            }, f, indent=2)
        
        print(f"\n📊 Summary:")
        print(f"✅ Processed {len(df)} words")
        print(f"✅ Downloaded {len(word_to_media)} unique media files")
        print(f"💾 Media mapping saved to: {mapping_file}")
        
        return sno_to_media

# Example usage:
if __name__ == "__main__":
    #@title Media Fetcher Settings {display-mode: "form"}
    
    #@markdown ### API Keys (Required)
    PEXELS_API_KEY = "" #@param {type:"string"}
    TENOR_API_KEY = "" #@param {type:"string"}
    
    #@markdown ### Input/Output Settings
    CSV_PATH = "timestamps.csv" #@param {type:"string"}
    MEDIA_TYPE = "pexels" #@param ["pexels", "tenor"]
    
    #@markdown ### Processing Settings
    MIN_DURATION = 0.3 #@param {type:"number"}
    #@markdown Minimum duration (in seconds) for each word. Words with shorter duration will be removed.
    
    #@markdown ### Run Media Fetcher
    def run_media_fetcher():
        """Run the media fetcher with the configured settings"""
        try:
            # Validate inputs
            if MEDIA_TYPE == "pexels" and not PEXELS_API_KEY:
                raise ValueError("❌ Pexels API key is required for video downloads")
            if MEDIA_TYPE == "tenor" and not TENOR_API_KEY:
                raise ValueError("❌ Tenor API key is required for GIF downloads")
            if not os.path.exists(CSV_PATH):
                raise ValueError(f"❌ CSV file not found: {CSV_PATH}")
            if MIN_DURATION <= 0:
                raise ValueError("❌ Minimum duration must be greater than 0 seconds")
            
            # Initialize fetcher
            fetcher = MediaFetcher()
            
            # Process CSV and download media
            api_key = PEXELS_API_KEY if MEDIA_TYPE == "pexels" else TENOR_API_KEY
            media_files = fetcher.process_csv(
                csv_path=CSV_PATH,
                media_type=MEDIA_TYPE,
                api_key=api_key,
                min_duration=MIN_DURATION
            )
            
            print("\n🎉 Media fetching completed!")
            print(f"📂 Media files are saved in the 'downloaded_media' directory")
            print(f"📝 Check 'media_mapping.json' for the mapping details")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Run the fetcher
    run_media_fetcher() 