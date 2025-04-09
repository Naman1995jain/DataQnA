import os
import gc
import tempfile
import uuid
import pandas as pd
import io
import time
import altair as alt
import json
import re
from dotenv import load_dotenv
import plotly.express as px
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from pathlib import Path
import base64
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract
from typing import Optional, List

import streamlit as st

# Import visualization utilities
from visualization_utils import generate_visualization, parse_visualization_request, get_data_summary

# Load environment variables
load_dotenv()

# Get API key from environment variables
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")
elif not openrouter_api_key.startswith('sk-or-v1-'):
    raise ValueError("Invalid OpenRouter API key format. Key should start with 'sk-or-v1-'")

# -----------------------------
# Session & State Management
# -----------------------------
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.uploaded_files = {}  # Dictionary to store multiple files
    st.session_state.data_messages = []
    st.session_state.doc_messages = []
    st.session_state.response_times = []
    st.session_state.token_counts = []
    st.session_state.ratings = []
    st.session_state.source_feedback = []
    st.session_state.current_file = None
    st.session_state.selected_file = None  # Currently selected file for chat

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {
        "data": [],  # For data analysis chat
        "document": [],  # For document chat
        "current_file": None  # Track the current file
    }

session_id = st.session_state.id

# Add configuration class
class Config:
    DEFAULT_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"
    DEFAULT_EMBEDDING = "BAAI/bge-large-en-v1.5"
    SUPPORTED_FILE_TYPES = ["xlsx", "xls", "csv", "json", "pdf"]
    MAX_FILE_SIZE_MB = 300
    TOKEN_LIMIT = 9000
    supported_structured_data = ["xlsx", "xls", "csv", "json"]
    supported_documents = ["pdf"]

# Add error handling decorator
def handle_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    return wrapper

# Add input validation
def validate_file(file):
    if file.size > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File size exceeds {Config.MAX_FILE_SIZE_MB}MB limit")
    
    extension = os.path.splitext(file.name)[1].lower()[1:]
    if extension not in Config.SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported file type. Please upload {', '.join(Config.SUPPORTED_FILE_TYPES)}")

@handle_exceptions
@st.cache_resource
def load_llm(api_key: str, model_name: str = Config.DEFAULT_MODEL):
    # Create an LLM instance using the provided OpenRouter API key.
    try:
        llm = OpenRouter(
            api_key=api_key,
            model=model_name,
            request_timeout=10000.0,
            headers={"HTTP-Referer": "https://localhost:8501"}  # Add referer for API validation
        )
        # Test the API key with a simple request
        llm.complete("test")
        return llm
    except Exception as e:
        if "401" in str(e):
            raise ValueError("OpenRouter API key authentication failed. Please check your API key.")
        else:
            raise e

def display_file(file):
    # Function kept for compatibility but doesn't display preview
    pass

def process_pdf(file) -> str:
    """Process a PDF file and extract its text content"""
    try:
        # Read PDF file bytes
        pdf_bytes = file.read()
        
        # Try PyMuPDF first for text extraction
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_content = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text_content.append(page.get_text())
        
        full_text = "\n".join(text_content)
        
        # If text extraction yields empty or minimal results, try OCR
        if len(full_text.strip()) < 100:  # Arbitrary threshold for minimum text
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes)
            ocr_text = []
            
            for img in images:
                # Perform OCR on each page
                page_text = pytesseract.image_to_string(img)
                ocr_text.append(page_text)
            
            full_text = "\n".join(ocr_text)
        
        doc.close()
        return full_text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def read_file_to_text(file):
    """Enhanced file reading function with PDF support"""
    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == ".pdf":
            text_content = process_pdf(file)
            if text_content:
                return f"Document Content:\n\n{text_content}"
            return None
        elif file_extension == ".csv":
            df = pd.read_csv(file)
        elif file_extension == ".json":
            # Handle different JSON formats (records, split, etc.)
            try:
                df = pd.read_json(file, orient='records')
            except:
                try:
                    df = pd.read_json(file, orient='split')
                except:
                    json_data = json.load(file)
                    if isinstance(json_data, dict):
                        df = pd.DataFrame([json_data])
                    elif isinstance(json_data, list):
                        df = pd.DataFrame(json_data)
                    else:
                        raise ValueError("Unsupported JSON structure")
        else:  # Excel files
            df = pd.read_excel(file)
        
        # Get data summary for structured data
        data_summary = get_data_summary(df)
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        df_sample = df.head(20).to_string()
        
        return f"Data Summary:\n{data_summary}\n\nDataFrame Info:\n{df_info}\n\nSample Data (first 20 rows):\n{df_sample}"
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def format_response(text):
    """Format response to show output only and handle visualizations"""
    # Split the response into parts
    parts = text.split("```")
    output_parts = []
    
    # If there are code blocks, extract output and filter visualization blocks
    if len(parts) > 1:
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text
                # Clean up any trailing "Here's a visualization:" type text
                cleaned_text = re.sub(r'\n*Here\'s? (?:a |the )?visualization.*$', '', part, flags=re.IGNORECASE)
                cleaned_text = re.sub(r'\n*I\'ll create (?:a |the )?visualization.*$', '', cleaned_text, flags=re.IGNORECASE)
                cleaned_text = re.sub(r'\n*Let me show (?:a |the )?visualization.*$', '', cleaned_text, flags=re.IGNORECASE)
                if cleaned_text.strip():
                    output_parts.append(cleaned_text)
            else:  # Code block
                # Skip visualization blocks entirely
                if not part.strip().startswith("visualization"):
                    # Handle other code blocks
                    lines = part.split('\n')
                    if len(lines) > 1:
                        content = '\n'.join(line for line in lines[1:] if line.strip())
                        if content:
                            output_parts.append(f'📊 Output:\n{content}')
    else:
        output_parts = [text]
    
    return '\n\n'.join(part for part in output_parts if part.strip())

def get_dataframe_from_file(file):
    """Load a DataFrame from the uploaded file"""
    file_extension = os.path.splitext(file.name)[1].lower()
    file.seek(0)  # Reset file pointer
    
    if file_extension == ".csv":
        return pd.read_csv(file)
    elif file_extension == ".json":
        try:
            return pd.read_json(file, orient='records')
        except:
            try:
                return pd.read_json(file, orient='split')
            except:
                json_data = json.load(file)
                if isinstance(json_data, dict):
                    return pd.DataFrame([json_data])
                elif isinstance(json_data, list):
                    return pd.DataFrame(json_data)
                else:
                    raise ValueError("Unsupported JSON structure")
    else:  # Excel files
        return pd.read_excel(file)

# Add performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = []

    def log_metric(self, name: str, value: float):
        self.metrics.append({
            "name": name,
            "value": value,
            "timestamp": time.time()
        })

    def get_summary(self):
        return pd.DataFrame(self.metrics)

# Initialize performance monitor
if "performance_monitor" not in st.session_state:
    st.session_state.performance_monitor = PerformanceMonitor()

# -----------------------------
# App Configuration & Styling
# -----------------------------
st.set_page_config(
    page_title="DataQnA AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/dataqna-ai',
        'Report a bug': "https://github.com/yourusername/dataqna-ai/issues",
        'About': "# DataQnA AI\nAn intelligent platform for natural language data analytics."
    }
)

# Add header component
def display_header():
    with st.container():
        st.markdown("""
            <div class="navbar">
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <h1 style="margin: 0;">🤖 DataQnA AI</h1>
                    </div>
                    <div class="nav-links" style="display: flex; gap: 1rem;">
                        <a href="#" class="nav-link tooltip" data-tooltip="View Documentation">📚</a>
                        <a href="#" class="nav-link tooltip" data-tooltip="Settings">⚙️</a>
                        <a href="#" class="nav-link tooltip" data-tooltip="Help">❓</a>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Call header at the start of the app
display_header()

# Custom theme initialization
def init_theme():
    # Detect system theme
    is_dark_theme = st.get_option("theme.base") == "dark"
    theme_class = "dark" if is_dark_theme else "light"
    
    # Inject theme class
    st.markdown(f"""
        <div data-theme="{theme_class}">
        <style>
            /* Theme-specific overrides */
            .stApp {{
                background-color: var(--background);
                color: var(--text);
            }}
            
            .stButton>button {{
                border-radius: 8px;
                padding: 0.5rem 1rem;
                border: 1px solid var(--border);
                background-color: var(--background);
                color: var(--text);
                transition: all 0.2s ease;
            }}
            
            .stButton>button:hover {{
                transform: translateY(-2px);
                box-shadow: var(--shadow-md);
            }}
            
            .stTextInput>div>div>input {{
                border-radius: 8px;
                border: 1px solid var(--border);
                background-color: var(--background);
                color: var(--text);
            }}
            
            .stSelectbox>div>div>div {{
                border-radius: 8px;
                border: 1px solid var(--border);
                background-color: var(--background);
                color: var(--text);
            }}
            
            .stTabs {{
                background-color: var(--background);
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
            }}
        </style>
        </div>
    """, unsafe_allow_html=True)

# Load custom CSS and setup theme
def load_custom_css():
    css_file = Path(__file__).parent / "styles" / "custom.css"
    with open(css_file) as f:
        css_content = f.read()
    
    # Detect if dark theme is enabled in Streamlit
    is_dark_theme = st.get_option("theme.base") == "dark"
    theme_class = "dark" if is_dark_theme else "light"
    
    # Inject the CSS with theme detection
    st.markdown(f"""
        <style>
            {css_content}
        </style>
    """, unsafe_allow_html=True)
    
    # Close the theme div at the bottom of the page
    st.markdown("</div>", unsafe_allow_html=True)

# Add custom components
def custom_metric(label, value, delta=None):
    html = f"""
        <div class="custom-metric">
            <h3>{label}</h3>
            <h2>{value}</h2>
            {f'<p>{delta}</p>' if delta else ''}
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Update page config with custom theme
load_custom_css()

# ====== Customize prompt template for multi-file support ======
qa_prompt_tmpl_str = (
    "You are analyzing the file: {file_name}\n\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above, follow these rules:\n"
    "1. When analyzing structured data (CSV, Excel, JSON):\n"
    "   - Focus on showing results and insights\n"
    "   - Present numerical results in a clear format\n"
    "   - Suggest visualizations when appropriate using the visualization format\n"
    "2. When analyzing documents (PDF):\n"
    "   - Provide concise summaries\n"
    "   - Extract key points and insights\n"
    "   - Quote relevant passages in markdown format\n"
    "   - Cite page numbers when available\n"
    "3. General rules:\n"
    "   - Keep explanations clear and concise\n"
    "   - Use bullet points for lists\n"
    "   - Don't show implementation details\n"
    "4. For data visualizations, use this format:\n"
    "```visualization\n"
    "{\n"
    '  "type": "bar|line|scatter|pie|histogram|heatmap",\n'
    '  "x_column": "column_name",\n'
    '  "y_column": "column_name",  # Optional for some chart types\n'
    '  "color_column": "column_name",  # Optional\n'
    '  "title": "Chart Title"\n'
    "}\n"
    "```\n"
    "Query: {query_str}\n"
    "Answer: ")

# -----------------------------
# Sidebar - Config & Navigation
# -----------------------------
# Enhanced file upload component
def display_file_upload_zone():
    st.markdown("""
        <div class="upload-zone">
            <div style="text-align: center;">
                <p style="font-size: 0.8rem; color: var(--text); opacity: 0.6;">
                    Supported formats: CSV, Excel, JSON, PDF
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Update sidebar file upload section
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: var(--primary-color);">⚙️ Configuration</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Display modern file upload zone
    display_file_upload_zone()
    
    file_type = st.radio(
        "Select file type",
        ["Structured Data (CSV, Excel, JSON)", "Documents (PDF)"],
        key="file_type",
        help="Choose the type of files you want to analyze"
    )
    
    # Multiple file uploader based on type
    if file_type == "Structured Data (CSV, Excel, JSON)":
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["xlsx", "xls", "csv", "json"],
            accept_multiple_files=True,
            help="Upload structured data files for analysis and visualization"
        )
    else:
        uploaded_files = st.file_uploader(
            "Choose PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF documents for text analysis and Q&A"
        )
    
    # Store uploaded files in session state without displaying the list
    if uploaded_files:
        for uploaded_file in uploaded_files:
            model_key = Config.DEFAULT_MODEL
            embedding_key = Config.DEFAULT_EMBEDDING
            file_key = f"{session_id}-{uploaded_file.name}-{model_key}-{embedding_key}"
            
            # Only process new files
            if file_key not in st.session_state.uploaded_files:
                st.session_state.uploaded_files[file_key] = uploaded_file
                st.info(f"Processing: {uploaded_file.name}")
                
                try:
                    # Convert file content to text
                    file_content = read_file_to_text(uploaded_file)
                    if file_content:
                        docs = [Document(text=file_content)]
                        
                        # Setup LLM & embedding model
                        llm = load_llm(openrouter_api_key, model_key)
                        embed_model = HuggingFaceEmbedding(model_name=embedding_key, trust_remote_code=True)
                        
                        # Create index
                        Settings.embed_model = embed_model
                        node_parser = MarkdownNodeParser()
                        index = VectorStoreIndex.from_documents(documents=docs, transformations=[node_parser], show_progress=True)
                        
                        # Create the query engine
                        Settings.llm = llm
                        query_engine = index.as_query_engine(streaming=True)
                        
                        # Store in file cache
                        st.session_state.file_cache[file_key] = query_engine
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Navigation section moved below
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="color: var(--primary-color);">Navigation</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation selection
    selected_section = st.radio(
        "Select Section",
        ["Data Chat", "Document Chat", "Data Explorer", "Performance Analytics", "File Manager"],
        key="navigation"
    )

# -----------------------------
# Main Content Area
# -----------------------------
def get_chat_history_for_file(file_name):
    """Get chat history for a specific file"""
    data_messages = [msg for msg in st.session_state.data_messages if msg.get("file_name") == file_name]
    doc_messages = [msg for msg in st.session_state.doc_messages if msg.get("file_name") == file_name]
    return {
        "data": data_messages,
        "document": doc_messages
    }

def download_chat_history(file_name):
    """Generate downloadable chat history for a file"""
    history = get_chat_history_for_file(file_name)
    
    # Combine all messages and format them
    formatted_history = []
    for chat_type, messages in history.items():
        if messages:
            formatted_history.append(f"\n=== {chat_type.upper()} CHAT ===\n")
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_history.append(f"\n[{role}]:\n{msg['content']}\n")
    
    return "\n".join(formatted_history)

def display_chat_interface(chat_type="data"):
    """Display enhanced chat interface with modern styling"""
    if not st.session_state.current_file:
        st.markdown("""
            <div class="card" style="text-align: center; padding: 3rem;">
                <h2>👋 Welcome to DataQnA AI</h2>
                <p style="color: var(--text); opacity: 0.8;">Upload a file in the sidebar to start analyzing your data</p>
                <div style="max-width: 500px; margin: 2rem auto;">
                    <h3>📄 Get Started</h3>
                    <p>Use the file uploader in the sidebar to begin</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Display minimalist file context banner
    st.markdown(f"""
        <div class="file-context-banner">
            <div class="file-info">
                <span class="file-context-icon">{'📊' if chat_type == 'data' else '📄'}</span>
                <div class="file-context-info">
                    <div class="file-context-name">{st.session_state.current_file.name}</div>
                </div>
            </div>
            <div class="file-actions">
                <button class="button button-secondary" onclick="clearChat()">🗑️ Clear</button>
                <button class="button button-primary" onclick="downloadChat()">📥 Save</button>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Chat messages container
    st.markdown('<div class="chat-thread">', unsafe_allow_html=True)
    
    # Get messages for current file
    messages = st.session_state.data_messages if chat_type == "data" else st.session_state.doc_messages
    current_file_messages = [msg for msg in messages 
                           if msg.get("file_name") == st.session_state.current_file.name]
    
    # Display messages with enhanced styling
    for message in current_file_messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(f"""
            <div class="chat-message {role_class}">
                <div class="message-content">{message["content"]}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced chat input area

def display_performance_metrics():
    """Display performance metrics in a modern grid layout"""
    st.markdown("""
        <div class="metrics-grid">
    """, unsafe_allow_html=True)
    
    # Calculate metrics
    if st.session_state.response_times:
        avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
        total_queries = len(st.session_state.response_times)
        avg_tokens = sum(st.session_state.token_counts) / len(st.session_state.token_counts) if st.session_state.token_counts else 0
        avg_rating = sum(r["rating"] for r in st.session_state.ratings) / len(st.session_state.ratings) if st.session_state.ratings else 0
        
        metrics = [
            {"label": "Total Queries", "value": f"{total_queries}", "icon": "🔍"},
            {"label": "Avg Response Time", "value": f"{avg_time:.2f}s", "icon": "⚡"},
            {"label": "Avg Token Usage", "value": f"{avg_tokens:.0f}", "icon": "🎯"},
            {"label": "User Satisfaction", "value": f"{avg_rating:.1f}/5", "icon": "⭐"}
        ]
        
        # Create metric cards
        for metric in metrics:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem;">{metric['icon']}</div>
                    <div class="metric-value">{metric['value']}</div>
                    <div style="color: var(--text); opacity: 0.8;">{metric['label']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_loading_animation():
    """Display an animated loading indicator"""
    st.markdown("""
        <div class="loading-container" style="text-align: center; padding: 2rem;">
            <div class="loading"></div>
            <p style="color: var(--text); opacity: 0.8; margin-top: 1rem;">
                Processing your request...
            </p>
        </div>
    """, unsafe_allow_html=True)

def show_file_processing_progress(file_name, progress):
    """Show file processing progress with animated bar"""
    st.markdown(f"""
        <div class="progress-container">
            <p style="margin-bottom: 0.5rem;">Processing: {file_name}</p>
            <div class="progress-bar">
                <div class="progress-bar-fill" style="width: {progress}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_success_toast(message):
    """Show a success toast notification"""
    st.markdown(f"""
        <div class="toast success-toast">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: var(--success-color);">✓</span>
                <span>{message}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_error_toast(message):
    """Show an error toast notification"""
    st.markdown(f"""
        <div class="toast error-toast">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: var(--error-color);">⚠</span>
                <span>{message}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_file_manager():
    """Display the file manager interface"""
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: var(--primary-color);">📁 File Manager</h1>
            <p style="color: var(--text); opacity: 0.8;">Manage your uploaded files</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.uploaded_files:
        st.info("No files uploaded yet. Use the file uploader in the sidebar to get started.")
        return
    
    # Display file list in modern cards
    for file_key, file in st.session_state.uploaded_files.items():
        file_type = "📊" if file.name.lower().endswith(tuple(Config.supported_structured_data)) else "📄"
        st.markdown(f"""
            <div class="file-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span style="font-size: 1.5rem;">{file_type}</span>
                        <div>
                            <div style="font-weight: bold;">{file.name}</div>
                            <div style="font-size: 0.8rem; opacity: 0.7;">
                                Type: {os.path.splitext(file.name)[1].upper()[1:]}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"Select '{file.name}'", key=f"select_{file_key}"):
                st.session_state.current_file = file
                st.success(f"Selected {file.name}")
                
        with col2:
            if st.button(f"Remove", key=f"remove_{file_key}"):
                del st.session_state.uploaded_files[file_key]
                if st.session_state.current_file and st.session_state.current_file.name == file.name:
                    st.session_state.current_file = None
                st.success(f"Removed {file.name}")
                st.experimental_rerun()

if selected_section == "Data Chat":
    if not st.session_state.current_file or os.path.splitext(st.session_state.current_file.name)[1].lower()[1:] not in Config.supported_structured_data:
        st.markdown(
            """
            <h1 style="color: #4B8BBE; text-align: center;">📊 Data Analysis Chat</h1>
            <h3 style="text-align: center;">Upload structured data (CSV, Excel, JSON) to begin analysis</h3>
            """,
            unsafe_allow_html=True,
        )
        st.info("Please upload a structured data file (CSV, Excel, or JSON) in the sidebar to start the analysis.")
    else:
        st.markdown(
            """
            <h1 style="color: #4B8BBE; text-align: center;">📊 Data Analysis Chat</h1>
            <h3 style="text-align: center;">Ask questions about your data in natural language</h3>
            """,
            unsafe_allow_html=True,
        )
        
        # Display chat interface with persistent history
        display_chat_interface(chat_type="data")
        
        # Chat input and processing
        if prompt := st.chat_input("What would you like to know about your data? 💬"):
            # Add user message with file reference
            new_message = {
                "role": "user",
                "content": prompt,
                "chat_type": "data",
                "file_name": st.session_state.current_file.name if st.session_state.current_file else None
            }
            st.session_state.data_messages.append(new_message)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                viz_params = None
                
                try:
                    # Get the query engine for this file
                    file_key = f"{session_id}-{st.session_state.current_file.name}-{model_key}-{embedding_key}"
                    if file_key not in st.session_state.file_cache:
                        st.error("File not indexed. Please try uploading again.")
                        st.stop()
                        
                    query_engine = st.session_state.file_cache[file_key]
                    
                    # Setup token counter for this query
                    token_counter = TokenCountingHandler()
                    Settings.callback_manager = CallbackManager([token_counter])
                    
                    # Initialize metrics
                    start_time = time.time()
                    token_count = 0

                    # Execute query
                    streaming_response = query_engine.query(prompt)
                    
                    # Process streaming response
                    for chunk in streaming_response.response_gen:
                        full_response += chunk
                        # Check for visualization in the response
                        current_viz = parse_visualization_request(full_response, get_dataframe_from_file(st.session_state.current_file))
                        if current_viz:
                            viz_params = current_viz
                        # Format the response without showing JSON
                        formatted_response = format_response(full_response)
                        message_placeholder.markdown(
                            formatted_response + "▌", 
                            unsafe_allow_html=True
                        )
                    
                    # Calculate response time
                    response_time = time.time() - start_time
                    st.session_state.response_times.append(response_time)
                    
                    # Final display of formatted response
                    formatted_response = format_response(full_response)
                    message_placeholder.markdown(
                        formatted_response,
                        unsafe_allow_html=True
                    )
                    
                    # Store assistant response with file reference
                    assistant_message = {
                        "role": "assistant",
                        "content": formatted_response,
                        "chat_type": "data",
                        "file_name": st.session_state.current_file.name
                    }
                    st.session_state.data_messages.append(assistant_message)
                    
                    # Generate visualization if params were found
                    if viz_params:
                        df = get_dataframe_from_file(st.session_state.current_file)
                        try:
                            fig = generate_visualization(
                                df=df,
                                viz_type=viz_params["type"],
                                x_col=viz_params["x_column"],
                                y_col=viz_params.get("y_column"),
                                color_col=viz_params.get("color_column"),
                                title=viz_params.get("title", "Data Visualization")
                            )
                            with st.container():
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
                    
                    # Rest of the metrics and feedback code...
                    
                    if hasattr(token_counter, 'total_llm_token_count'):
                        token_count = token_counter.total_llm_token_count
                        if token_count > 0:
                            st.session_state.token_counts.append(token_count)
                    
                    with st.sidebar:
                        st.markdown("### ⚙️ Response Metrics")
                        st.info(f"Response time: {response_time:.2f} seconds")
                        if token_count > 0:
                            st.info(f"Tokens used: {token_count}")
                        
                        if hasattr(streaming_response, 'sources') and streaming_response.sources:
                            with st.expander("View Source Chunks", expanded=False):
                                st.write(streaming_response.sources)
                            
                            source_quality = st.radio("Were the retrieved chunks relevant?", 
                                                    ["Yes", "Partially", "No"])
                            if st.button("Submit Source Feedback"):
                                st.session_state.source_feedback.append({
                                    "query": prompt, 
                                    "rating": source_quality
                                })
                                st.success("Thank you for your feedback!")
                        
                        user_rating = st.slider("Rate answer accuracy (1-5)", 1, 5, 3)
                        if st.button("Submit Rating"):
                            st.session_state.ratings.append({
                                "query": prompt, 
                                "rating": user_rating
                            })
                            st.success("Thank you for your feedback!")
                        
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg
                    
            st.session_state.data_messages.append({"role": "assistant", "content": full_response})

elif selected_section == "Document Chat":
    if not st.session_state.current_file or os.path.splitext(st.session_state.current_file.name)[1].lower()[1:] not in Config.supported_documents:
        st.markdown(
            """
            <h1 style="color: #4B8BBE; text-align: center;">📄 Document Chat</h1>
            <h3 style="text-align: center;">Upload PDF documents to begin analysis</h3>
            """,
            unsafe_allow_html=True,
        )
        st.info("Please upload a PDF document in the sidebar to start the conversation.")
    else:
        st.markdown(
            """
            <h1 style="color: #4B8BBE; text-align: center;">📄 Document Chat</h1>
            <h3 style="text-align: center;">Ask questions about your document</h3>
            """,
            unsafe_allow_html=True,
        )
        
        # Display chat interface with persistent history for documents
        display_chat_interface(chat_type="document")
        
        # Chat input processing for document analysis
        if prompt := st.chat_input("What would you like to know about your document? 💬"):
            # Add user message with file reference
            new_message = {
                "role": "user",
                "content": prompt,
                "chat_type": "document",
                "file_name": st.session_state.current_file.name if st.session_state.current_file else None
            }
            st.session_state.doc_messages.append(new_message)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Get the query engine for this file
                    file_key = f"{session_id}-{st.session_state.current_file.name}-{model_key}-{embedding_key}"
                    if file_key not in st.session_state.file_cache:
                        st.error("File not indexed. Please try uploading again.")
                        st.stop()
                        
                    query_engine = st.session_state.file_cache[file_key]
                    
                    # Setup token counter for this query
                    token_counter = TokenCountingHandler()
                    Settings.callback_manager = CallbackManager([token_counter])
                    
                    # Initialize metrics
                    start_time = time.time()
                    token_count = 0

                    # Execute query
                    streaming_response = query_engine.query(prompt)
                    
                    # Process streaming response
                    for chunk in streaming_response.response_gen:
                        full_response += chunk
                        # Format the response without showing JSON
                        formatted_response = format_response(full_response)
                        message_placeholder.markdown(
                            formatted_response + "▌", 
                            unsafe_allow_html=True
                        )
                    
                    # Calculate response time
                    response_time = time.time() - start_time
                    st.session_state.response_times.append(response_time)
                    
                    # Final display of formatted response
                    formatted_response = format_response(full_response)
                    message_placeholder.markdown(
                        formatted_response,
                        unsafe_allow_html=True
                    )
                    
                    # Store assistant response with file reference for document chat
                    assistant_message = {
                        "role": "assistant",
                        "content": formatted_response,
                        "chat_type": "document",
                        "file_name": st.session_state.current_file.name
                    }
                    st.session_state.doc_messages.append(assistant_message)
                    
                    # Rest of the metrics and feedback code...
                    
                    if hasattr(token_counter, 'total_llm_token_count'):
                        token_count = token_counter.total_llm_token_count
                        if token_count > 0:
                            st.session_state.token_counts.append(token_count)
                    
                    with st.sidebar:
                        st.markdown("### ⚙️ Response Metrics")
                        st.info(f"Response time: {response_time:.2f} seconds")
                        if token_count > 0:
                            st.info(f"Tokens used: {token_count}")
                        
                        if hasattr(streaming_response, 'sources') and streaming_response.sources:
                            with st.expander("View Source Chunks", expanded=False):
                                st.write(streaming_response.sources)
                            
                            source_quality = st.radio("Were the retrieved chunks relevant?", 
                                                    ["Yes", "Partially", "No"])
                            if st.button("Submit Source Feedback"):
                                st.session_state.source_feedback.append({
                                    "query": prompt, 
                                    "rating": source_quality
                                })
                                st.success("Thank you for your feedback!")
                        
                        user_rating = st.slider("Rate answer accuracy (1-5)", 1, 5, 3)
                        if st.button("Submit Rating"):
                            st.session_state.ratings.append({
                                "query": prompt, 
                                "rating": user_rating
                            })
                            st.success("Thank you for your feedback!")
                        
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg
                    
            st.session_state.doc_messages.append({"role": "assistant", "content": full_response})

elif selected_section == "Data Explorer":
    st.markdown(
        """
        <h1 style="color: #4B8BBE; text-align: center;">📊 Data Explorer</h1>
        """,
        unsafe_allow_html=True,
    )
    
    # Data Explorer code
    if st.session_state.current_file:
        uploaded_file = st.session_state.current_file
        
        # Get the DataFrame
        df = get_dataframe_from_file(uploaded_file)
        
        # Data overview
        st.subheader("Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info)
        
        # Quick visualizations
        st.subheader("Quick Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            viz_type = st.selectbox(
                "Visualization Type",
                options=["bar", "line", "scatter", "pie", "histogram", "heatmap"]
            )
        
        with viz_col2:
            x_col = st.selectbox("X-axis Column", options=df.columns)
        
        # Additional parameters based on chart type
        if viz_type in ["bar", "line", "scatter"]:
            y_col = st.selectbox("Y-axis Column", options=df.columns)
            color_col_options = ["None"] + list(df.columns)
            color_col = st.selectbox("Color By (optional)", options=color_col_options)
            color_col = None if color_col == "None" else color_col
            
            try:
                fig = generate_visualization(
                    df=df,
                    viz_type=viz_type,
                    x_col=x_col,
                    y_col=y_col,
                    color_col=color_col,
                    title=f"{viz_type.capitalize()} Chart: {y_col} by {x_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
                
        elif viz_type in ["histogram", "pie"]:
            color_col_options = ["None"] + list(df.columns)
            color_col = st.selectbox("Color By (optional)", options=color_col_options)
            color_col = None if color_col == "None" else color_col
            
            try:
                fig = generate_visualization(
                    df=df,
                    viz_type=viz_type,
                    x_col=x_col,
                    color_col=color_col,
                    title=f"{viz_type.capitalize()} Chart: {x_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
        elif viz_type == "heatmap":
            # For heatmap, we need to decide if we're showing correlation or crosstab
            heatmap_type = st.radio("Heatmap Type", ["Correlation Matrix", "Cross Tabulation"])
            
            if heatmap_type == "Correlation Matrix":
                # Filter to only numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns for correlation matrix")
                else:
                    try:
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(
                            corr_matrix,
                            title="Correlation Matrix",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating correlation heatmap: {str(e)}")
            else:
                # Cross tabulation between two categorical columns
                y_col = st.selectbox("Y-axis Column", options=df.columns)
                try:
                    cross_tab = pd.crosstab(df[x_col], df[y_col])
                    fig = px.imshow(
                        cross_tab,
                        title=f"Cross Tabulation: {x_col} vs {y_col}",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating cross tabulation: {str(e)}")
        
        # Data filtering and exploration
        st.subheader("Data Filtering")
        
        # Select columns to display
        with st.expander("Select Columns to Display"):
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect(
                "Choose columns to display",
                options=all_columns,
                default=all_columns[:5] if len(all_columns) > 5 else all_columns
            )
            
            if selected_columns:
                st.dataframe(df[selected_columns])
        
        # Filter data
        with st.expander("Filter Data"):
            filter_col = st.selectbox("Select column to filter", options=df.columns)
            
            # Different filter options based on column type
            if pd.api.types.is_numeric_dtype(df[filter_col]):
                min_val = float(df[filter_col].min())
                max_val = float(df[filter_col].max())
                filter_range = st.slider(
                    f"Filter range for {filter_col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                filtered_df = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
            else:
                # For categorical columns
                unique_values = df[filter_col].unique().tolist()
                selected_values = st.multiselect(
                    f"Select values for {filter_col}",
                    options=unique_values,
                    default=unique_values
                )
                if selected_values:
                    filtered_df = df[df[filter_col].isin(selected_values)]
                else:
                    filtered_df = df
            
            st.write(f"Filtered data: {len(filtered_df)} rows")
            st.dataframe(filtered_df)
            
            # Option to download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
        
        # Data statistics
        with st.expander("Data Statistics"):
            # Select columns for statistics
            stat_cols = st.multiselect(
                "Select columns for statistics",
                options=df.select_dtypes(include=['number']).columns.tolist(),
                default=df.select_dtypes(include=['number']).columns.tolist()[:3]
            )
            
            if stat_cols:
                st.write("Descriptive Statistics:")
                st.dataframe(df[stat_cols].describe())
                
                # Correlation matrix for selected columns
                if len(stat_cols) > 1:
                    st.write("Correlation Matrix:")
                    st.dataframe(df[stat_cols].corr())
    else:
        st.info("Please upload a file in the sidebar to explore data.")

elif selected_section == "File Manager":
    display_file_manager()

elif selected_section == "Performance Analytics":
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: var(--primary-color);">📈 Performance Analytics</h1>
            <p style="color: var(--text); opacity: 0.8;">Track and analyze system performance metrics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display metrics grid
    display_performance_metrics()
    
    # Response time metrics
    if st.session_state.response_times:
        avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
        
        # Plot response times
        times_df = pd.DataFrame({
            "query": range(1, len(st.session_state.response_times) + 1),
            "time": st.session_state.response_times
        })
        
        chart = alt.Chart(times_df).mark_line().encode(
            x=alt.X('query:O', title='Query Number'),
            y=alt.Y('time:Q', title='Response Time (s)')
        ).properties(title='Response Times')
        
        st.altair_chart(chart, use_container_width=True)
    
    # Token usage metrics
    if st.session_state.token_counts:
        tokens_df = pd.DataFrame({
            "query": range(1, len(st.session_state.token_counts) + 1),
            "tokens": st.session_state.token_counts
        })
        
        token_chart = alt.Chart(tokens_df).mark_bar().encode(
            x=alt.X('query:O', title='Query Number'),
            y=alt.Y('tokens:Q', title='Token Count')
        ).properties(title='Token Usage per Query')
        
        st.altair_chart(token_chart, use_container_width=True)
    
    # User ratings visualization
    if st.session_state.ratings:
        ratings = [r["rating"] for r in st.session_state.ratings]
        
        # Rating distribution
        rating_df = pd.DataFrame({"rating": ratings})
        rating_chart = alt.Chart(rating_df).mark_bar().encode(
            x=alt.X('rating:O', title='Rating'),
            y=alt.Y('count()', title='Count')
        ).properties(title='User Accuracy Ratings')
        
        st.altair_chart(rating_chart, use_container_width=True)
    
    # Source feedback visualization
    if st.session_state.source_feedback:
        source_ratings = [r["rating"] for r in st.session_state.source_feedback]
        
        # Count occurrences of each rating
        from collections import Counter
        rating_counts = Counter(source_ratings)
        
        # Create DataFrame for visualization
        source_df = pd.DataFrame({
            "rating": list(rating_counts.keys()),
            "count": list(rating_counts.values())
        })
        
        st.subheader("Source Relevance Feedback")
        source_chart = alt.Chart(source_df).mark_bar().encode(
            x=alt.X('rating:N', title='Rating'),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color('rating:N', scale=alt.Scale(
                domain=['Yes', 'Partially', 'No'],
                range=['#1d9a6c', '#ffbf00', '#e74c3c']
            ))
        ).properties(title='Source Chunk Relevance')
        
        st.altair_chart(source_chart, use_container_width=True)
    
    st.subheader("Model Performance")
    # Instead of comparison, just show current model's performance
    if st.session_state.response_times:
        st.info("Model: Mistral Small (Free)")
    
    # Export performance data
    if st.session_state.response_times:
        st.subheader("Export Performance Data")
        
        # Prepare data for export
        export_data = {
            "response_times": st.session_state.response_times,
            "model_used": Config.DEFAULT_MODEL
        }
        
        if st.session_state.token_counts:
            export_data["token_counts"] = st.session_state.token_counts
        
        if st.session_state.ratings:
            export_data["ratings"] = st.session_state.ratings
        
        if st.session_state.source_feedback:
            export_data["source_feedback"] = st.session_state.source_feedback
        
        # Convert to DataFrame for CSV export
        export_df = pd.DataFrame({
            "query_num": range(1, len(st.session_state.response_times) + 1),
            "response_time": st.session_state.response_times,
            "model": [Config.DEFAULT_MODEL] * len(st.session_state.response_times)
        })
        
        if st.session_state.token_counts and len(st.session_state.token_counts) == len(st.session_state.response_times):
            export_df["token_count"] = st.session_state.token_counts
        
        # Add ratings if available and match the length
        if st.session_state.ratings and len(st.session_state.ratings) == len(st.session_state.response_times):
            export_df["user_rating"] = [r["rating"] for r in st.session_state.ratings]
        
        # Convert to CSV for download
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download Performance Data as CSV",
            data=csv,
            file_name="rag_performance_data.csv",
            mime="text/csv"
        )
