# One-Click YouTube Video Creator

This Gradio app allows users to create complete YouTube-ready videos by simply uploading or pasting a story. The app handles all aspects of video creation including text-to-speech conversion, media selection, and synchronization.

## Features

- **Text-to-Speech**: Convert your story to natural-sounding speech using Edge TTS
- **Word-Level Timestamps**: Extract precise timing for each word using WhisperX
- **Automatic Media Selection**: Fetch relevant videos and GIFs based on your story and selected genre
- **Custom Options**: Choose resolution, enable/disable subtitles, and set a custom title
- **One-Click Generation**: Generate a complete, synchronized video ready for YouTube

## Setup Options

### Option 1: Hugging Face Spaces

#### 1. Clone the repository on Hugging Face Spaces

Create a new Gradio Space on Hugging Face and clone this repository.

#### 2. Set up API Keys

Add the following secrets to your Hugging Face Space:

- `PEXELS_API_KEY`: Get from [Pexels API](https://www.pexels.com/api/)
- `TENOR_API_KEY`: Get from [Tenor API](https://developers.google.com/tenor/guides/quickstart)

#### 3. Deploy

Deploy your Hugging Face Space with a GPU for faster processing.

### Option 2: Google Colab

You can run this app directly in Google Colab for development and testing.

#### Quickest method: Single command setup

In a Colab notebook, just run these commands to set up and start the app:

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/youtube-video-creator
%cd youtube-video-creator

# Run the setup script (will prompt for API keys)
!bash setup_colab.sh
```

Or if you've already downloaded all the files to Colab:

```python
# Make the setup script executable and run it
!chmod +x setup_colab.sh
!./setup_colab.sh
```

The script will:
- Install all dependencies
- Configure the environment for Colab
- Prompt for API keys
- Start the app with a public URL

#### Alternative: Step-by-step setup

#### 1. Open the notebook in Colab

Click this button to open the setup notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/youtube-video-creator/blob/main/colab_setup.ipynb)

Or upload `colab_setup.ipynb` to your Google Drive and open it in Colab.

#### 2. Run the setup cells

Follow the instructions in the notebook to:
- Clone the repository
- Install dependencies
- Set up API keys
- Run the app

## Usage

1. **Input Story**: Paste or type your story in the text input area
2. **Select Genre**: Choose a genre that matches your story's theme
3. **Set Title**: Enter a catchy title for your video
4. **Customize Options**: Adjust resolution and subtitle settings as needed
5. **Generate**: Click the "Create Video" button and wait for processing to complete
6. **Download**: Once generated, you can download your video directly from the interface

## How It Works

1. Your story is converted to speech using Edge TTS
2. WhisperX extracts timestamps for each word
3. Keywords are extracted from your story and combined with the genre to search for relevant media
4. Videos and GIFs are fetched from Pexels and Tenor APIs
5. Media clips are synchronized with the audio based on word timestamps
6. The final video is rendered with MoviePy

## Troubleshooting

### Google Colab Issues

If you encounter ALSA errors or model loading errors in Colab:

1. Make sure you're using the modified app.py provided in this repository
2. Try running with a GPU runtime for better performance
3. If WhisperX still fails, try the fallback CPU approach:
   ```python
   !python app.py --device cpu
   ```

### API Key Issues

If no media is being fetched:
1. Check that your API keys are correctly set
2. Verify your API keys are active by testing them with a simple request
3. Check your console for any API error messages

## Notes

- Processing time depends on story length and available computing resources
- For best results, provide clear, detailed stories (1-3 paragraphs recommended)
- The app works best with GPU acceleration enabled
- Media quality depends on available content from Pexels and Tenor for your keywords

## Future Enhancements

- Background music selection
- Voice type selection
- Direct YouTube upload capability 
- Custom transitions between media clips
- More granular control over media selection 