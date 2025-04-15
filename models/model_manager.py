import os
from typing import Optional, Dict, Any, List
import time

from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

import streamlit as st

class ModelManager:
    """
    Manages LLM and embedding models, including loading, configuration, and query execution.
    Handles API key validation, model initialization, and token counting.
    """
    def __init__(self, api_key: str, model_name: str, embedding_model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.llm = None
        self.embed_model = None
        self.token_counter = None
    
    def load_llm(self):
        """
        Creates an LLM instance using the provided OpenRouter API key.
        Validates the API key with a test request.
        """
        try:
            self.llm = OpenRouter(
                api_key=self.api_key,
                model=self.model_name,
                request_timeout=10000.0,
                headers={"HTTP-Referer": "https://localhost:8501"}  # Add referer for API validation
            )
            # Test the API key with a simple request
            self.llm.complete("test")
            return self.llm
        except Exception as e:
            if "401" in str(e):
                raise ValueError("OpenRouter API key authentication failed. Please check your API key.")
            else:
                raise e
    
    def load_embedding_model(self):
        """
        Loads the embedding model for document indexing.
        """
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name, 
                trust_remote_code=True
            )
            return self.embed_model
        except Exception as e:
            raise ValueError(f"Failed to load embedding model: {str(e)}")
    
    def setup_token_counter(self):
        """
        Sets up a token counter for tracking token usage.
        """
        self.token_counter = TokenCountingHandler()
        Settings.callback_manager = CallbackManager([self.token_counter])
        return self.token_counter
    
    def create_index_from_documents(self, documents: List[Document]):
        """
        Creates a vector store index from the provided documents.
        """
        if not self.embed_model:
            self.load_embedding_model()
        
        Settings.embed_model = self.embed_model
        node_parser = MarkdownNodeParser()
        index = VectorStoreIndex.from_documents(
            documents=documents, 
            transformations=[node_parser], 
            show_progress=True
        )
        return index
    
    def create_query_engine(self, index):
        """
        Creates a query engine from the provided index.
        """
        if not self.llm:
            self.load_llm()
        
        Settings.llm = self.llm
        query_engine = index.as_query_engine(streaming=True)
        return query_engine
    
    def execute_query(self, query_engine, prompt: str):
        """
        Executes a query using the provided query engine.
        Returns the streaming response and tracks performance metrics.
        """
        # Setup token counter for this query
        token_counter = self.setup_token_counter()
        
        # Initialize metrics
        start_time = time.time()
        
        # Execute query
        streaming_response = query_engine.query(prompt)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Get token count
        token_count = 0
        if hasattr(token_counter, 'total_llm_token_count'):
            token_count = token_counter.total_llm_token_count
        
        return {
            "streaming_response": streaming_response,
            "response_time": response_time,
            "token_count": token_count
        }