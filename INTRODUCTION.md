# 🤖 DataQnA AI: Natural Language Data Analytics

## Introduction

DataQnA AI is an intelligent platform that empowers users to interact with structured and unstructured data through natural language queries. Built on the Retrieval-Augmented Generation (RAG) architecture, it bridges the gap between complex data tools and business users by offering intuitive data insights without requiring coding knowledge.

## 🎯 Purpose

The primary goal of DataQnA AI is to democratize data analytics by allowing users to:

- Query data using natural language instead of complex query languages
- Automatically generate appropriate visualizations based on the query intent
- Extract insights from various data sources including structured files, PDFs, websites, and YouTube videos
- Interact with data in a conversational manner, similar to chatting with a data analyst

## 🏗️ Architecture Overview

DataQnA AI follows a modular architecture with the following key components:

### Core Components

1. **Data Processing Layer**
   - `FileHandler`: Manages file operations for various formats (CSV, Excel, JSON, PDF)
   - `WebsiteHandler`: Extracts and processes content from web pages
   - `YouTubeHandler`: Processes transcripts and metadata from YouTube videos

2. **AI & Model Layer**
   - `ModelManager`: Manages LLM and embedding models using OpenRouter API
   - `VisualizationManager`: Handles data visualization generation and interpretation

3. **User Interface Layer**
   - `UIManager`: Controls the Streamlit-based user interface components
   - `SessionManager`: Manages user sessions and data persistence

4. **Performance & Configuration Layer**
   - `PerformanceMonitor`: Tracks response times and token usage
   - `ConfigManager`: Handles application configuration settings

## 💻 Technical Stack

- **Frontend**: Streamlit for interactive web interface
- **Backend**: Python with modular OOP architecture
- **AI Models**: 
  - LLM: Mistral via OpenRouter API
  - Embeddings: BAAI/bge-large-en-v1.5 via HuggingFace
- **Data Processing**: pandas, numpy, PyMuPDF, pdf2image, pytesseract
- **Visualization**: plotly, altair, matplotlib, seaborn
- **Web Scraping**: requests, beautifulsoup4, validators
- **Media Processing**: youtube-transcript-api

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ installed
- OpenRouter API key (for LLM access)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Unix/MacOS
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

### Running the Application

```bash
# Activate virtual environment if not already activated
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS

# Start the application
streamlit run app.py
```

## 🔍 Features in Detail

### 1. Data Chat
Query structured data files (CSV, Excel, JSON) using natural language. The system automatically generates appropriate visualizations based on your query.

### 2. Document Chat
Extract insights from PDF documents. The system uses text extraction and OCR to make document content queryable.

### 3. Website Chat
Analyze content from websites by providing a URL. The system extracts and indexes the content for querying.

### 4. YouTube Chat
Extract insights from YouTube videos by providing a URL. The system processes the transcript and makes it queryable.

### 5. Data Explorer
Explore uploaded data files with interactive filtering, sorting, and visualization options.

### 6. Performance Analytics
Track system performance metrics including response times and token usage.

## 📊 Visualization Capabilities

DataQnA AI supports various visualization types:

- Bar charts for categorical comparisons
- Line charts for time series data
- Scatter plots for relationship analysis
- Pie charts for proportion visualization
- Histograms for distribution analysis
- Heatmaps for correlation matrices

The system automatically selects the most appropriate visualization type based on the query and data characteristics.

## 🧩 Project Structure

```
/
├── app.py                  # Main application entry point
├── models/                 # Core modules
│   ├── config_manager.py   # Configuration settings
│   ├── file_handler.py     # File processing
│   ├── model_manager.py    # LLM and embedding models
│   ├── ui_manager.py       # UI components
│   ├── visualization_manager.py  # Data visualization
│   ├── session_manager.py  # User session management
│   ├── website_handler.py  # Website content processing
│   └── youtube_handler.py  # YouTube content processing
├── styles/                 # CSS styling
│   └── custom.css          # Custom UI styles
├── images/                 # System diagrams and images
├── visualization_utils.py  # Visualization utilities
└── requirements.txt        # Project dependencies
```

## 🔮 Future Development

Planned enhancements include:

- Voice-based querying for hands-free operation
- Multi-user dashboards for team collaboration
- SQL database support for direct database querying
- Support for switching between multiple LLMs
- Enhanced privacy and security features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

- **Developer**: Naman Jain
- **Internal Guide**: Prof. Honey Shivnani

---

*This document provides an introduction to the DataQnA AI project. For more detailed information, please refer to the README.md and the source code documentation.*