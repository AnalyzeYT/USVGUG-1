# YouTube Video Creator - Google Colab Implementation

This project allows you to create complete YouTube-ready videos by simply uploading or pasting a story. The app processes text into audio, generates timestamps, and synchronizes relevant media to create a complete video.

## Quick Start - Google Colab

1. Open the notebook in Google Colab by clicking this badge:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/youtube-video-creator/blob/main/youtube_video_creator.ipynb)

2. Run all cells in order

3. Enter your API keys (get them from [Pexels API](https://www.pexels.com/api/) and [Tenor API](https://developers.google.com/tenor/guides/quickstart))

4. Enter your story, select options, and click "Create Video"

## Features

- **Text-to-Speech**: Convert your story to natural-sounding speech using Edge TTS
- **Word-Level Timestamps**: Extract precise timing for each word using WhisperX
- **Automatic Media Selection**: Fetch relevant videos and GIFs based on your story and selected genre
- **Custom Options**: Choose resolution, enable/disable subtitles, and set a custom title
- **Pure Python Implementation**: No web framework dependencies, runs directly in Colab

## How It Works

1. Your story is converted to speech using Edge TTS
2. WhisperX extracts timestamps for each word
3. Keywords are extracted from your story and combined with the genre to search for relevant media
4. Videos and GIFs are fetched from Pexels and Tenor APIs
5. Media clips are synchronized with the audio based on word timestamps
6. The final video is rendered with MoviePy

## Requirements

- Python 3.7+
- Libraries listed in requirements.txt
- API keys for Pexels and Tenor

## Manual Installation (If not using Colab)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/youtube-video-creator.git
cd youtube-video-creator

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook youtube_video_creator.ipynb
```

## Notes

- Processing time depends on story length and available computing resources
- For best results, provide clear, detailed stories (1-3 paragraphs recommended)
- The app works best with GPU acceleration enabled
- Media quality depends on available content from Pexels and Tenor for your keywords
- This version is optimized for Google Colab and doesn't require Gradio or a web interface

## Troubleshooting

- If you encounter ALSA errors or model loading errors in Colab, try restarting the runtime
- If WhisperX fails to load, the app will automatically fall back to CPU mode with float32 computation
- For "CUDA out of memory" errors, try decreasing the resolution to 480p or shortening your story 