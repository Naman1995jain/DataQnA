import uuid
import streamlit as st
from typing import Dict, Any, Optional, List, Union

class SessionManager:
    """
    Manages application session state and data persistence.
    Handles initialization, storage, and retrieval of session data.
    """
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """
        Initialize session state variables if they don't exist.
        """
        if "id" not in st.session_state:
            st.session_state.id = uuid.uuid4()
            st.session_state.file_cache = {}
            st.session_state.uploaded_files = {}  # Dictionary to store multiple files
            st.session_state.website_urls = {}  # Dictionary to store website URLs
            st.session_state.youtube_videos = {}  # Dictionary to store YouTube videos
            st.session_state.data_messages = []
            st.session_state.doc_messages = []
            st.session_state.web_messages = []  # Messages for website chat
            st.session_state.youtube_messages = []  # Messages for YouTube chat
            st.session_state.response_times = []
            st.session_state.token_counts = []
            st.session_state.ratings = []
            st.session_state.source_feedback = []
            st.session_state.current_file = None
            st.session_state.current_website = None  # Currently selected website for chat
            st.session_state.current_youtube = None  # Currently selected YouTube video for chat
            st.session_state.selected_file = None  # Currently selected file for chat
        
        # Initialize chat history if it doesn't exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = {
                "data": [],  # For data analysis chat
                "document": [],  # For document chat
                "website": [],  # For website chat
                "youtube": [],  # For YouTube chat
                "current_file": None,  # Track the current file
                "current_website": None,  # Track the current website
                "current_youtube": None  # Track the current YouTube video
            }
        
        # Initialize performance monitor if it doesn't exist
        if "performance_monitor" not in st.session_state:
            from models.performance_monitor import PerformanceMonitor
            st.session_state.performance_monitor = PerformanceMonitor()
    
    def get_session_id(self) -> str:
        """
        Get the current session ID.
        """
        return str(st.session_state.id)
    
    def add_file_to_cache(self, file_key: str, query_engine):
        """
        Add a query engine to the file cache.
        """
        st.session_state.file_cache[file_key] = query_engine
    
    def get_file_from_cache(self, file_key: str):
        """
        Get a query engine from the file cache.
        """
        return st.session_state.file_cache.get(file_key)
    
    def add_uploaded_file(self, file_key: str, file):
        """
        Add an uploaded file to the session state.
        """
        st.session_state.uploaded_files[file_key] = file
    
    def remove_uploaded_file(self, file_key: str):
        """
        Remove an uploaded file from the session state.
        """
        if file_key in st.session_state.uploaded_files:
            del st.session_state.uploaded_files[file_key]
            
            # If the current file is being removed, reset it
            if st.session_state.current_file and st.session_state.current_file.name == st.session_state.uploaded_files[file_key].name:
                st.session_state.current_file = None
    
    def set_current_file(self, file):
        """
        Set the current file for chat.
        """
        st.session_state.current_file = file
    
    def get_current_file(self):
        """
        Get the current file for chat.
        """
        return st.session_state.current_file
    
    def add_website_url(self, url_key: str, url: str, content: str):
        """
        Add a website URL to the session state.
        """
        st.session_state.website_urls[url_key] = {
            "url": url,
            "content": content
        }
    
    def get_website_url(self, url_key: str) -> Optional[Dict[str, str]]:
        """
        Get a website URL from the session state.
        """
        return st.session_state.website_urls.get(url_key)
    
    def remove_website_url(self, url_key: str):
        """
        Remove a website URL from the session state.
        """
        if url_key in st.session_state.website_urls:
            del st.session_state.website_urls[url_key]
            
            # If the current website is being removed, reset it
            if st.session_state.current_website and st.session_state.current_website.get("key") == url_key:
                st.session_state.current_website = None
    
    def set_current_website(self, website: Dict[str, str]):
        """
        Set the current website for chat.
        """
        st.session_state.current_website = website
    
    def get_current_website(self) -> Optional[Dict[str, str]]:
        """
        Get the current website for chat.
        """
        return st.session_state.current_website
        
    def add_youtube_url(self, url_key: str, url: str, content: str):
        """
        Add a YouTube URL to the session state.
        """
        st.session_state.youtube_videos[url_key] = {
            "url": url,
            "content": content
        }
    
    def get_youtube_url(self, url_key: str) -> Optional[Dict[str, str]]:
        """
        Get a YouTube URL from the session state.
        """
        return st.session_state.youtube_videos.get(url_key)
    
    def remove_youtube_url(self, url_key: str):
        """
        Remove a YouTube URL from the session state.
        """
        if url_key in st.session_state.youtube_videos:
            del st.session_state.youtube_videos[url_key]
            
            # If the current YouTube video is being removed, reset it
            if st.session_state.current_youtube and st.session_state.current_youtube.get("key") == url_key:
                st.session_state.current_youtube = None
    
    def set_current_youtube(self, youtube: Dict[str, str]):
        """
        Set the current YouTube video for chat.
        """
        st.session_state.current_youtube = youtube
    
    def get_current_youtube(self) -> Optional[Dict[str, str]]:
        """
        Get the current YouTube video for chat.
        """
        return st.session_state.current_youtube
    
    def add_message(self, message: Dict[str, Any], chat_type: str = "data"):
        """
        Add a message to the chat history.
        """
        if chat_type == "data":
            st.session_state.data_messages.append(message)
        elif chat_type == "document":
            st.session_state.doc_messages.append(message)
        elif chat_type == "website":
            st.session_state.web_messages.append(message)
        else:  # youtube
            st.session_state.youtube_messages.append(message)
    
    def get_messages(self, chat_type: str = "data") -> List[Dict[str, Any]]:
        """
        Get messages from the chat history.
        """
        if chat_type == "data":
            return st.session_state.data_messages
        elif chat_type == "document":
            return st.session_state.doc_messages
        elif chat_type == "website":
            return st.session_state.web_messages
        else:  # youtube
            return st.session_state.youtube_messages
    
    def get_chat_history_for_file(self, file_name: str) -> Dict[str, List]:
        """
        Get chat history for a specific file.
        """
        data_messages = [msg for msg in st.session_state.data_messages if msg.get("file_name") == file_name]
        doc_messages = [msg for msg in st.session_state.doc_messages if msg.get("file_name") == file_name]
        return {
            "data": data_messages,
            "document": doc_messages
        }
    
    def download_chat_history(self, file_name: str) -> str:
        """
        Generate downloadable chat history for a file.
        """
        history = self.get_chat_history_for_file(file_name)
        
        # Combine all messages and format them
        formatted_history = []
        for chat_type, messages in history.items():
            if messages:
                formatted_history.append(f"\n=== {chat_type.upper()} CHAT ===\n")
                for msg in messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    formatted_history.append(f"\n[{role}]:\n{msg['content']}\n")
        
        return "\n".join(formatted_history)
    
    def log_response_time(self, response_time: float):
        """
        Log a response time to the session state.
        """
        st.session_state.response_times.append(response_time)
        st.session_state.performance_monitor.log_response_time(response_time)
    
    def log_token_count(self, token_count: int):
        """
        Log a token count to the session state.
        """
        if token_count > 0:
            st.session_state.token_counts.append(token_count)
            st.session_state.performance_monitor.log_token_count(token_count)
    
    def log_user_rating(self, query: str, rating: int):
        """
        Log a user rating to the session state.
        """
        st.session_state.ratings.append({
            "query": query, 
            "rating": rating
        })
        st.session_state.performance_monitor.log_user_rating(query, rating)
    
    def log_source_feedback(self, query: str, rating: str):
        """
        Log source feedback to the session state.
        """
        st.session_state.source_feedback.append({
            "query": query, 
            "rating": rating
        })
        st.session_state.performance_monitor.log_source_feedback(query, rating)
    
    def get_performance_monitor(self):
        """
        Get the performance monitor from the session state.
        """
        return st.session_state.performance_monitor