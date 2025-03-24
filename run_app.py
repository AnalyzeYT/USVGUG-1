"""
Run script for YouTube Video Creator
This can be executed directly in Google Colab
"""
import os
import time
import sys
from google.colab import files
from IPython.display import display, Audio, Video, HTML

# First run setup script if it exists
if os.path.exists('setup.py'):
    print("Running setup script...")
    import setup
    setup.install_dependencies()
    setup.configure_environment()

# Import the video creator module
import video_creator

# Function to get user input
def get_user_input():
    """Get user input for video creation"""
    # Story input
    print("\n--- Enter Your Story ---")
    story = input("Enter your story text (or type 'sample' for a sample story): ")
    
    if story.lower() == 'sample':
        story = """The small fishing village awakened as the morning sun painted the horizon in shades of amber and gold. 
Fishermen prepared their boats, checking nets and supplies for the day ahead. The sea, calm today, whispered 
promises of a good catch. Children ran along the shore, collecting shells and chasing seagulls. In the heart of 
the village, the market sprung to life with vendors arranging fresh produce and morning delicacies. The village 
had thrived this way for generations, finding balance between tradition and the rhythms of nature."""
        print("\nUsing sample story:")
        print(story)
    
    # Genre selection
    print("\n--- Select a Genre ---")
    genres = ['Nature', 'Technology', 'Education', 'Entertainment', 'Business', 'Travel', 'Sports', 'Food']
    for i, genre in enumerate(genres, 1):
        print(f"{i}. {genre}")
    
    genre_choice = input("Enter the number for your genre (1-8): ")
    try:
        genre_idx = int(genre_choice) - 1
        if 0 <= genre_idx < len(genres):
            genre = genres[genre_idx]
        else:
            genre = "Nature"  # Default
            print("Invalid selection. Using 'Nature' as default.")
    except ValueError:
        genre = "Nature"  # Default
        print("Invalid input. Using 'Nature' as default.")
    
    # Title input
    title = input("\nEnter a title for your video: ")
    
    # Subtitle option
    subtitle_choice = input("\nInclude subtitles? (y/n): ").lower()
    include_subtitles = subtitle_choice.startswith('y')
    
    # Resolution selection
    print("\n--- Select Resolution ---")
    print("1. 480p (faster rendering)")
    print("2. 720p (recommended)")
    print("3. 1080p (slower rendering)")
    
    res_choice = input("Enter your choice (1-3): ")
    resolutions = ["480p", "720p", "1080p"]
    try:
        res_idx = int(res_choice) - 1
        if 0 <= res_idx < len(resolutions):
            resolution = resolutions[res_idx]
        else:
            resolution = "720p"  # Default
            print("Invalid selection. Using 720p as default.")
    except ValueError:
        resolution = "720p"  # Default
        print("Invalid input. Using 720p as default.")
    
    # API Keys
    pexels_api_key = input("\nEnter your Pexels API Key: ")
    tenor_api_key = input("Enter your Tenor API Key: ")
    
    return {
        "story": story,
        "genre": genre,
        "title": title,
        "include_subtitles": include_subtitles,
        "resolution": resolution,
        "pexels_api_key": pexels_api_key,
        "tenor_api_key": tenor_api_key
    }

def progress_callback(message):
    """Simple progress callback function"""
    print(message)

def main():
    """Main function to run the app"""
    print("=" * 60)
    print("YouTube Video Creator".center(60))
    print("=" * 60)
    
    # Get user inputs
    inputs = get_user_input()
    
    # Validate inputs
    if not inputs["story"].strip():
        print("Error: Story cannot be empty.")
        return
    
    if not inputs["pexels_api_key"].strip() or not inputs["tenor_api_key"].strip():
        print("Error: Both API keys are required.")
        return
    
    # Create the video
    print("\n--- Creating Your Video ---\n")
    start_time = time.time()
    
    result = video_creator.create_video(
        story=inputs["story"],
        genre=inputs["genre"],
        title=inputs["title"],
        pexels_api_key=inputs["pexels_api_key"],
        tenor_api_key=inputs["tenor_api_key"],
        include_subtitles=inputs["include_subtitles"],
        resolution=inputs["resolution"],
        progress_callback=progress_callback
    )
    
    end_time = time.time()
    
    # Check result
    if isinstance(result, str) and result.startswith("Error"):
        print(f"\nError: {result}")
    else:
        print(f"\nVideo created successfully in {end_time - start_time:.2f} seconds!")
        print(f"Output file: {result}")
        
        # Display the video in Colab
        print("\nDisplaying video (if in Colab):")
        try:
            display(Video(result, width=640))
        except NameError:
            print("Not running in IPython/Colab environment. Cannot display video.")
            
        # Provide download link
        try:
            display(HTML(f"<a href='files/{result}' download>Download video</a>"))
            print("\nIf download link doesn't work, use this command to download:")
            print(f"files.download('{result}')")
        except NameError:
            print(f"Video saved to {result}")

if __name__ == "__main__":
    main() 