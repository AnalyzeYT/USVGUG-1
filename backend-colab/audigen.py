import os
import PyPDF2
import edge_tts
import asyncio
from pathlib import Path
import tempfile
import nest_asyncio

# Enable nested event loops (required for Jupyter/Colab)
nest_asyncio.apply()

#@title Audio Generator Class and Functions
class AudioGenerator:
    def __init__(self):
        self.voice = "en-US-JennyNeural" #@param ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-AU-NatashaNeural", "en-AU-WilliamNeural"]
        self.output_dir = "generated_audio" #@param {type:"string"}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
    
    def read_text_file(self, file_path):
        """Read text from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")
    
    def clean_text(self, text):
        """Clean and prepare text for TTS"""
        # Remove extra whitespace and normalize line endings
        text = " ".join(text.split())
        return text
    
    async def generate_audio(self, text, output_filename=None, voice=None):
        """Generate audio from text using Edge TTS"""
        try:
            if voice:
                self.voice = voice
            
            # Create a temporary file if no output filename is provided
            if output_filename is None:
                temp_dir = tempfile.gettempdir()
                output_filename = os.path.join(temp_dir, "generated_speech.mp3")
            else:
                output_filename = os.path.join(self.output_dir, output_filename)
            
            # Initialize Edge TTS communicate
            tts = edge_tts.Communicate(text, self.voice)
            
            # Generate audio
            await tts.save(output_filename)
            
            return output_filename
        
        except Exception as e:
            raise Exception(f"Error generating audio: {str(e)}")
    
    async def process_input_async(self, input_source, input_type="text", output_filename=None, voice=None):
        """Process input and generate audio based on input type (async version)"""
        try:
            # Extract text based on input type
            if input_type == "pdf":
                text = self.extract_text_from_pdf(input_source)
            elif input_type == "file":
                text = self.read_text_file(input_source)
            else:  # text input
                text = input_source
            
            # Clean the text
            text = self.clean_text(text)
            
            # Generate audio
            audio_path = await self.generate_audio(text, output_filename, voice)
            
            return {
                "success": True,
                "audio_path": audio_path,
                "text_length": len(text),
                "duration": None  # TODO: Add audio duration calculation
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_input(self, input_source, input_type="text", output_filename=None, voice=None):
        """Synchronous wrapper for process_input_async"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.process_input_async(input_source, input_type, output_filename, voice)
        )

#@title Run Text-to-Speech Conversion {display-mode: "form"}

#@markdown ### Input Settings
input_text = "Hello! This is a test of the text to speech system." #@param {type:"string"}
input_type = "text" #@param ["text", "pdf", "file"] {allow-input: true}
output_filename = "test1.mp3" #@param {type:"string"}

#@markdown ### Voice Settings
selected_voice = "en-US-JennyNeural" #@param ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-AU-NatashaNeural", "en-AU-WilliamNeural"]

#@markdown ### File Paths (only needed for PDF or Text file input)
pdf_path = "sample.pdf" #@param {type:"string"}
text_path = "sample.txt" #@param {type:"string"}

def run_conversion():
    """Run the text-to-speech conversion with the selected parameters"""
    generator = AudioGenerator()
    
    # Set the input source based on input type
    if input_type == "text":
        source = input_text
    elif input_type == "pdf":
        source = pdf_path
    else:
        source = text_path
    
    # Process the input
    result = generator.process_input(
        input_source=source,
        input_type=input_type,
        output_filename=output_filename,
        voice=selected_voice
    )
    
    # Print the result
    if result["success"]:
        print(f"✅ Audio generated successfully!")
        print(f"📂 Output file: {result['audio_path']}")
        print(f"📝 Text length: {result['text_length']} characters")
        
        # Display the audio player if in Colab
        from IPython.display import Audio, display
        display(Audio(result["audio_path"]))
    else:
        print(f"❌ Error: {result['error']}")

# Run the conversion
if __name__ == "__main__":
    run_conversion()