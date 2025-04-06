from dataclasses import dataclass
from typing import Dict, List
import os
#
@dataclass
class ModelConfig:
    name: str
    api_key_env: str
    max_tokens: int
    temperature: float

@dataclass
class AppConfig:
    title: str = "AI Data Analyst"
    models: Dict[str, ModelConfig] = None
    supported_file_types: List[str] = None
    max_file_size_mb: int = 500
    cache_dir: str = ".cache"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    
    def __post_init__(self):
        self.models = {
            "mistral": ModelConfig(
                name="mistralai/mistral-small-3.1-24b-instruct:free",
                api_key_env="OPENROUTER_API_KEY",
                max_tokens=40000,
                temperature=0.2
            )
        }
        self.supported_file_types = ["xlsx", "xls", "csv"]

    def validate_environment(self) -> List[str]:
        missing_vars = []
        for model in self.models.values():
            if not os.getenv(model.api_key_env):
                missing_vars.append(model.api_key_env)
        return missing_vars

config = AppConfig()
