import requests
from bs4 import BeautifulSoup
import streamlit as st
from typing import Optional, Dict, Any
import re
import validators

class WebsiteHandler:
    """
    Handles website URL operations including validation, fetching, and processing content.
    Extracts text content from websites for analysis.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def validate_url(self, url: str) -> bool:
        """
        Validates if the provided string is a valid URL.
        Raises ValueError if validation fails.
        """
        if not validators.url(url):
            raise ValueError("Invalid URL format. Please provide a valid website URL.")
        return True
    
    def fetch_website_content(self, url: str) -> Optional[str]:
        """
        Fetches and processes website content from the given URL.
        Returns the extracted text content or None if an error occurs.
        """
        try:
            # Validate URL before proceeding
            self.validate_url(url)
            
            # Fetch website content
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
                script_or_style.extract()
            
            # Get text content
            text = soup.get_text(separator='\n')
            
            # Clean up text (remove excessive whitespace)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Extract title if available
            title = soup.title.string if soup.title else "Website Content"
            
            return f"Website: {title}\n\nContent:\n{text}"
        
        except requests.RequestException as e:
            st.error(f"Error fetching website: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error processing website content: {str(e)}")
            return None
    
    def get_website_metadata(self, url: str) -> Dict[str, Any]:
        """
        Extracts metadata from the website such as title, description, and keywords.
        """
        metadata = {
            "url": url,
            "title": "",
            "description": "",
            "keywords": ""
        }
        
        try:
            # Validate URL before proceeding
            self.validate_url(url)
            
            # Fetch website content
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            if soup.title:
                metadata["title"] = soup.title.string
            
            # Extract meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc:
                metadata["description"] = meta_desc.get("content", "")
            
            # Extract meta keywords
            meta_keywords = soup.find("meta", attrs={"name": "keywords"})
            if meta_keywords:
                metadata["keywords"] = meta_keywords.get("content", "")
            
            return metadata
        
        except Exception:
            # Return basic metadata if there's an error
            return metadata