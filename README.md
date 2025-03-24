# YouTube Video Creator for Google Colab

This is a Python-based video creator that creates complete YouTube-ready videos from stories. It works in Google Colab without requiring Gradio or notebook files.

## How to Use in Google Colab

1. **Create a new Colab notebook**

2. **Upload the Python files**
   - Upload `setup.py`, `video_creator.py`, and `run_app.py` to your Colab session
   - You can upload files using the file upload button in the left sidebar

3. **Run the following in a Colab cell:**
   ```python
   # Run the setup script
   %run setup.py
   
   # Run the main application
   %run run_app.py
   ```

4. **Follow the prompts to enter:**
   - Your story (or use the sample)
   - Genre selection
   - Video title
   - Subtitle preference
   - Resolution
   - API keys for Pexels and Tenor

5. **When finished, the video will be displayed in Colab and available for download**

## Files Included

- **setup.py**: Installs all required dependencies and configures the environment
- **video_creator.py**: Contains the core functionality for creating videos
- **run_app.py**: Simple command-line interface to run in Colab

## Requirements

- Pexels API key (get from [Pexels API](https://www.pexels.com/api/))
- Tenor API key (get from [Tenor API](https://developers.google.com/tenor/guides/quickstart))

## Features

- Text-to-speech conversion using Edge TTS
- Word-level timestamp extraction with WhisperX
- Automatic media collection from Pexels and Tenor
- Video creation with MoviePy
- Simple command-line interface for Google Colab

## Troubleshooting

- If you encounter any issues with audio generation, try rerunning the setup.py script
- If WhisperX model loading fails, try restarting the Colab runtime
- For errors related to media, check your API keys and internet connection 