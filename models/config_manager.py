from dataclasses import dataclass
from typing import Dict, List, Optional
import os

@dataclass
class ModelConfig:
    name: str
    api_key_env: str
    max_tokens: int
    temperature: float

class ConfigManager:
    """
    Manages application configuration settings.
    Centralizes all configuration parameters and provides validation methods.
    """
    def __init__(self):
        # Application settings
        self.title = "DataQnA"
        self.cache_dir = ".cache"
        
        # Model configurations
        self.models = {
            "mistral": ModelConfig(
                name="mistralai/mistral-small-3.1-24b-instruct:free",
                api_key_env="OPENROUTER_API_KEY",
                max_tokens=40000,
                temperature=0.7
            )
        }
        self.default_model = "mistralai/mistral-small-3.1-24b-instruct:free"
        self.default_embedding = "BAAI/bge-large-en-v1.5"
        
        # File handling settings
        self.supported_structured_data = ["xlsx", "xls", "csv", "json"]
        self.supported_documents = ["pdf", "gdoc"]  # Added gdoc for Google Documents
        self.supported_websites = ["url"]  # For website URLs
        self.supported_youtube = ["youtube"]  # For YouTube videos
        self.supported_file_types = self.supported_structured_data + self.supported_documents
        self.max_file_size_mb = 300
        
        # Token limits
        self.token_limit = 9000
    
    def validate_environment(self) -> List[str]:
        """
        Validates that all required environment variables are set.
        Returns a list of missing environment variables.
        """
        missing_vars = []
        for model in self.models.values():
            if not os.getenv(model.api_key_env):
                missing_vars.append(model.api_key_env)
        return missing_vars
    
    def get_api_key(self, model_key: str = "mistral") -> Optional[str]:
        """
        Gets the API key for the specified model from environment variables.
        """
        if model_key not in self.models:
            return None
        
        return os.getenv(self.models[model_key].api_key_env)
    
    def is_valid_file_type(self, file_extension: str) -> bool:
        """
        Checks if the given file extension is supported.
        """
        # Remove the dot if present
        if file_extension.startswith('.'):
            file_extension = file_extension[1:]
        
        return file_extension.lower() in self.supported_file_types
    
    def is_structured_data(self, file_extension: str) -> bool:
        """
        Checks if the given file extension is for structured data.
        """
        # Remove the dot if present
        if file_extension.startswith('.'):
            file_extension = file_extension[1:]
        
        return file_extension.lower() in self.supported_structured_data
    
    def is_document(self, file_extension: str) -> bool:
        """
        Checks if the given file extension is for a document.
        """
        # Remove the dot if present
        if file_extension.startswith('.'):
            file_extension = file_extension[1:]
        
        return file_extension.lower() in self.supported_documents