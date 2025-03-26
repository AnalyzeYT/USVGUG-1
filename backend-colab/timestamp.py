import whisperx
import torch
import pandas as pd
import os
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
import glob

# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class TimestampGenerator:
    def __init__(self):
        # Common words to remove (can be extended)
        self.common_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'can', 'cannot', "can't", 'could', 'may', 'might',
            'must', 'need', 'ought', 'shall', 'should', 'would', 'i', 'you',
            'your', 'yours', 'yourself', 'yourselves'
        }
        
        # Initialize device
        self.device = self._setup_device()
        print(f"🖥️ Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Load WhisperX model (will be loaded when needed)
        self.model = None
        self.alignment_model = None
    
    def _setup_device(self) -> str:
        """Setup and return the best available device"""
        if torch.cuda.is_available():
            # Set CUDA device
            torch.cuda.set_device(0)
            # Clear GPU cache
            torch.cuda.empty_cache()
            return "cuda"
        return "cpu"
    
    def load_models(self):
        """Load WhisperX models if not already loaded"""
        try:
            if self.model is None:
                print("📥 Loading Whisper base model...")
                # Force float32 computation type
                self.model = whisperx.load_model(
                    "base",
                    self.device,
                    compute_type="float32"  # Changed from default float16
                )
                print("✅ Whisper model loaded successfully")
            
            if self.alignment_model is None:
                print("📥 Loading alignment model...")
                self.alignment_model, self.metadata = whisperx.load_align_model(
                    language_code="en",
                    device=self.device
                )
                print("✅ Alignment model loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")
    
    def add_common_words(self, words: List[str]):
        """Add more words to the common words set"""
        self.common_words.update(set(word.lower() for word in words))
    
    def remove_common_words(self, words: List[Dict]) -> List[Dict]:
        """Remove common words from the word list while preserving timestamps"""
        return [word for word in words if word['word'].lower() not in self.common_words]
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing"""
        # Pattern matching for temporary files
        patterns = ["=*.*", "~*.*"]
        
        for pattern in patterns:
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    print(f"🧹 Cleaned up temporary file: {file}")
                except Exception as e:
                    print(f"⚠️ Could not remove {file}: {str(e)}")
    
    def process_audio(
        self,
        audio_path: str,
        output_csv: Optional[str] = None,
        remove_common: bool = False,
        batch_size: int = 16
    ) -> pd.DataFrame:
        """
        Process audio file and generate word-level timestamps
        
        Args:
            audio_path: Path to the audio file
            output_csv: Path to save the CSV file (optional)
            remove_common: Whether to remove common words
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with word timestamps
        """
        try:
            # Load models if needed
            self.load_models()
            
            print("🎯 Transcribing audio...")
            # Transcribe audio with language specified
            result = self.model.transcribe(
                audio_path,
                batch_size=batch_size,
                language="en"  # Specify English language
            )
            print("✅ Transcription complete")
            
            print("🎯 Aligning timestamps...")
            # Align whisper output
            result = whisperx.align(
                result["segments"],
                self.alignment_model,
                self.metadata,
                audio_path,
                self.device
            )
            print("✅ Alignment complete")
            
            # Extract word-level data
            print("🎯 Processing words...")
            words_data = []
            
            for segment in tqdm(result["segments"], desc="Processing segments"):
                for word in segment["words"]:
                    word_data = {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "title": ""  # Initialize with empty title
                    }
                    words_data.append(word_data)
            
            # Remove common words if requested
            if remove_common:
                print("🎯 Removing common words...")
                initial_count = len(words_data)
                words_data = self.remove_common_words(words_data)
                removed_count = initial_count - len(words_data)
                print(f"✅ Removed {removed_count} common words")
            
            # Create DataFrame
            df = pd.DataFrame(words_data)
            
            # Add required columns
            df['duration'] = df['end'] - df['start']
            df.insert(0, 's.no', range(1, len(df) + 1))
            
            # Reorder columns
            df = df[['s.no', 'title', 'word', 'start', 'end', 'duration']]
            
            # Save to CSV if output path provided
            if output_csv:
                print(f"💾 Saving to {output_csv}...")
                df.to_csv(output_csv, index=False)
                print("✅ CSV file saved")
            
            # Clean up temporary files after processing
            self.cleanup_temp_files()
            
            return df
        
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")

#@title Run Timestamp Generation {display-mode: "form"}

#@markdown ### Input Settings
audio_path = "generated_audio/test1.mp3" #@param {type:"string"}
output_csv = "timestamps.csv" #@param {type:"string"}
remove_common_words = True #@param {type:"boolean"}

#@markdown ### Additional Common Words (optional)
additional_words = "" #@param {type:"string"} 

def run_timestamp_generation():
    """Run the timestamp generation with the selected parameters"""
    try:
        # Initialize generator
        generator = TimestampGenerator()
        
        # Add additional common words if provided
        if additional_words:
            new_words = [w.strip() for w in additional_words.split(',')]
            generator.add_common_words(new_words)
            print(f"✅ Added {len(new_words)} custom words to common words list")
        
        # Process the audio
        df = generator.process_audio(
            audio_path=audio_path,
            output_csv=output_csv,
            remove_common=remove_common_words
        )
        
        # Display results
        print("\n📊 Results Summary:")
        print(f"✅ Timestamps generated successfully!")
        print(f"📂 CSV file saved to: {output_csv}")
        print(f"📝 Total words processed: {len(df)}")
        print("\n📋 First few entries:")
        print(df.head().to_string())
        
        # Display memory usage if using GPU
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"\n💾 GPU Memory Usage:")
            print(f"   Allocated: {memory_allocated:.2f} MB")
            print(f"   Reserved: {memory_reserved:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Run the timestamp generation
if __name__ == "__main__":
    run_timestamp_generation() 