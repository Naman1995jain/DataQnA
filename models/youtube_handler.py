import re
import streamlit as st
from typing import Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

class YouTubeHandler:
    """
    Handles YouTube video operations including validation, fetching, and processing content.
    Extracts transcript and metadata from YouTube videos for analysis.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def validate_youtube_url(self, url: str) -> Optional[str]:
        """
        Validates if the provided string is a valid YouTube URL.
        Returns the video ID if valid, None otherwise.
        """
        # Regular expressions for different YouTube URL formats
        youtube_regex = (
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        
        match = re.match(youtube_regex, url)
        if match:
            return match.group(6)  # Return the video ID
        else:
            raise ValueError("Invalid YouTube URL format. Please provide a valid YouTube video URL.")
    
    def fetch_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Fetches metadata for a YouTube video including title, description, etc.
        """
        metadata = {
            "video_id": video_id,
            "title": "",
            "description": "",
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        }
        
        try:
            # Fetch the YouTube video page
            url = f"https://www.youtube.com/watch?v={video_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                # Remove " - YouTube" from the title
                title = title_tag.string.replace(" - YouTube", "") if title_tag.string else "YouTube Video"
                metadata["title"] = title
            
            # Try to extract description (this is more complex due to YouTube's dynamic content)
            # This is a simplified approach and might not always work
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc:
                metadata["description"] = meta_desc.get("content", "")
            
            return metadata
            
        except Exception as e:
            st.error(f"Error fetching video metadata: {str(e)}")
            return metadata
    
    def fetch_video_transcript(self, video_id: str) -> Optional[str]:
        """
        Fetches the transcript of a YouTube video.
        Returns the transcript text or None if unavailable.
        """
        try:
            # Get transcript using youtube_transcript_api
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine all transcript segments into a single text
            transcript_text = " ".join([item['text'] for item in transcript_list])
            
            return transcript_text
            
        except Exception as e:
            st.error(f"Error fetching video transcript: {str(e)}")
            return None
    
    def fetch_youtube_content(self, url: str) -> Optional[str]:
        """
        Main method to fetch and process YouTube video content.
        Returns a formatted string with video metadata and transcript.
        """
        try:
            # Validate URL and get video ID
            video_id = self.validate_youtube_url(url)
            if not video_id:
                return None
            
            # Fetch video metadata
            metadata = self.fetch_video_metadata(video_id)
            
            # Fetch video transcript
            transcript = self.fetch_video_transcript(video_id)
            
            if not transcript:
                return f"YouTube Video: {metadata['title']}\n\nDescription: {metadata['description']}\n\nTranscript: Not available for this video."
            
            # Format the content
            content = f"YouTube Video: {metadata['title']}\n\nDescription: {metadata['description']}\n\nTranscript:\n{transcript}"
            
            return content
            
        except Exception as e:
            st.error(f"Error processing YouTube video: {str(e)}")
            return None
            
    def get_youtube_metadata(self, url: str) -> Dict[str, Any]:
        """
        Get metadata for a YouTube video including title, description, and thumbnail.
        Returns a dictionary with video metadata.
        """
        try:
            # Validate URL and get video ID
            video_id = self.validate_youtube_url(url)
            if not video_id:
                return {"title": "Unknown Video", "description": "", "url": url}
            
            # Fetch video metadata
            metadata = self.fetch_video_metadata(video_id)
            metadata["url"] = url
            
            return metadata
        except Exception as e:
            st.error(f"Error getting YouTube metadata: {str(e)}")
            return {"title": "Unknown Video", "description": "", "url": url}