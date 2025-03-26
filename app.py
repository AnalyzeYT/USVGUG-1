import gradio as gr
import os
from typing import Optional
import tempfile
import shutil

def generate_video(
    content_type: str,
    text_content: Optional[str] = None,
    pdf_file: Optional[gr.File] = None,
    text_file: Optional[gr.File] = None,
    music_source: str = "auto",
    video_style: str = "stock",
    voice_type: str = "female",
    speech_speed: float = 1.0,
    subtitles: bool = True,
    include_intro: bool = False,
    include_outro: bool = False,
    screen_size: str = "16:9"
):
    """
    Generate a video based on the provided parameters.
    This is a placeholder function that simulates video generation.
    In a real implementation, you would integrate with actual video generation services.
    """
    # Simulate video generation
    output_path = "output_video.mp4"
    
    # Here you would implement the actual video generation logic
    # For now, we'll just return a message
    return f"Video generation started with the following parameters:\n" \
           f"Content Type: {content_type}\n" \
           f"Voice Type: {voice_type}\n" \
           f"Speech Speed: {speech_speed}\n" \
           f"Screen Size: {screen_size}"

def process_content(content_type, text_input, pdf_file, text_file, voice_type, speech_speed, 
                   screen_size, music_type, subtitles, video_style, include_intro, include_outro,
                   pexels_key, openai_key, elevenlabs_key):
    """Handle the video generation process"""
    if content_type == "Upload PDF" and pdf_file is None:
        return "Please upload a PDF file"
    elif content_type == "Text File" and text_file is None:
        return "Please upload a text file"
    elif content_type == "Paste Text" and not text_input:
        return "Please enter some text"
    
    # Here you would implement the actual video generation logic
    return f"Processing video with {content_type} input..."

def show_upload_options(choice):
    """Control visibility of input options based on content type selection"""
    is_text = choice == "Paste Text"
    is_pdf = choice == "Upload PDF"
    is_file = choice == "Text File"
    
    # Return individual visibility values instead of a dictionary
    return (
        gr.update(visible=is_text),  # text_area visibility
        gr.update(visible=is_pdf),   # pdf_upload visibility
        gr.update(visible=is_file)   # text_file_upload visibility
    )

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # AI Video Generator
        Transform your text into engaging videos in minutes
        """
    )
    
    with gr.Column(variant="panel"):
        gr.Markdown("### Create Your Video")
        
        # Content Source Section
        with gr.Group():
            gr.Markdown("📄 Content Source")
            content_type = gr.Radio(
                choices=["Paste Text", "Upload PDF", "Text File"],
                value="Paste Text",
                label="Select Input Type"
            )
            
            # Input options
            with gr.Group():
                text_area = gr.Textbox(
                    placeholder="Enter your content here...",
                    lines=5,
                    label="Text Input",
                    visible=True
                )
                
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    visible=False
                )
                
                text_file_upload = gr.File(
                    label="Upload Text File",
                    file_types=[".txt", ".doc", ".docx"],
                    visible=False
                )

        # Screen Size Section
        with gr.Group():
            gr.Markdown("📺 Screen Size")
            screen_size = gr.Radio(
                choices=["16:9", "4:3", "9:16"],
                value="16:9",
                label="Video Format",
                info="16:9 for YouTube, 9:16 for mobile"
            )

        # Background Music Section
        with gr.Group():
            gr.Markdown("🎵 Background Music")
            music_type = gr.Radio(
                choices=["Auto-select music", "Upload custom music", "No music"],
                value="Auto-select music",
                label="Music Options"
            )

        # Voice Settings Section
        with gr.Group():
            gr.Markdown("🎙️ Voice Settings")
            with gr.Row():
                voice_type = gr.Dropdown(
                    choices=["Female", "Male"],
                    value="Female",
                    label="Voice Type"
                )
                speech_speed = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Speech Speed"
                )

        # Subtitles Option
        with gr.Group():
            gr.Markdown("📝 Subtitles")
            subtitles = gr.Checkbox(
                value=True,
                label="Generate Subtitles",
                info="Add subtitles to your video"
            )

        # Video Style Section
        with gr.Group():
            gr.Markdown("🎬 Video Style")
            video_style = gr.Radio(
                choices=["Stock Footage", "Static Background"],
                value="Stock Footage",
                label="Visual Style"
            )

        # Intro/Outro Options
        with gr.Group():
            gr.Markdown("🎥 Intro/Outro")
            include_intro = gr.Checkbox(label="Add Intro", info="Include opening sequence")
            include_outro = gr.Checkbox(label="Add Outro", info="Include closing sequence")

        # API Configuration
        with gr.Group():
            gr.Markdown("🔑 API Configuration")
            pexels_key = gr.Textbox(
                label="Pexels API Key",
                placeholder="Enter Pexels API key",
                type="password"
            )
            openai_key = gr.Textbox(
                label="OpenAI API Key",
                placeholder="Enter OpenAI API key",
                type="password"
            )
            elevenlabs_key = gr.Textbox(
                label="ElevenLabs API Key",
                placeholder="Enter ElevenLabs API key",
                type="password"
            )

        # Generate Button and Status
        generate_btn = gr.Button("Generate Video", variant="primary")
        output_text = gr.Textbox(label="Status")

    # Event handlers
    content_type.change(
        fn=show_upload_options,
        inputs=[content_type],
        outputs=[text_area, pdf_upload, text_file_upload]
    )
    
    generate_btn.click(
        fn=process_content,
        inputs=[
            content_type, text_area, pdf_upload, text_file_upload,
            voice_type, speech_speed, screen_size, music_type,
            subtitles, video_style, include_intro, include_outro,
            pexels_key, openai_key, elevenlabs_key
        ],
        outputs=[output_text]
    )

if __name__ == "__main__":
    demo.launch() 