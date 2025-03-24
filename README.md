# YouTube Video Creator - Gradio Web Interface

This is a Python-based video creator that creates complete YouTube-ready videos from stories. It uses Gradio to provide a user-friendly web interface.

## How to Use in Google Colab

1. **Create a new Colab notebook**

2. **Upload the Python files**
   - Upload `setup.py` and `app.py` to your Colab session
   - You can upload files using the file upload button in the left sidebar

3. **Run the following in a Colab cell:**
   ```python
   # Run the setup script
   %run setup.py
   
   # Run the Gradio app (add --share flag to create a public URL)
   !python app.py --device cpu --compute_type float32
   ```

4. **Use the web interface**
   - Enter your story in the text box (or use the sample)
   - Select a genre from the dropdown
   - Enter a title for your video
   - Choose subtitle and resolution options
   - Enter your Pexels and Tenor API keys
   - Click "Create Video"

5. **When finished, the video will be displayed in the interface and available for download**

## Command-line Arguments

- `--device`: Choose `cpu` or `cuda` (GPU) for processing (default: `cpu`)
- `--compute_type`: Choose `float32` or `float16` for model computation (default: `float32`)
- `--share`: Create a public URL to access the interface from anywhere

## Files Included

- **setup.py**: Installs all required dependencies and configures the environment
- **app.py**: Contains both the video creation functionality and Gradio web interface

## Requirements

- Pexels API key (get from [Pexels API](https://www.pexels.com/api/))
- Tenor API key (get from [Tenor API](https://developers.google.com/tenor/guides/quickstart))

## Features

- Text-to-speech conversion using Edge TTS
- Word-level timestamp extraction with WhisperX
- Automatic media collection from Pexels and Tenor
- Video creation with MoviePy
- User-friendly web interface with Gradio

## Troubleshooting

- If you encounter any issues with audio generation, try rerunning the setup.py script
- If WhisperX model loading fails, ensure you're using CPU with float32 in Colab
- For errors related to media, check your API keys and internet connection
- If you encounter any security warnings about the web interface, you can add the `--share` flag to create a Gradio public URL 