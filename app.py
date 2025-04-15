import os
import streamlit as st
import re
from dotenv import load_dotenv

# Import models
from models.config_manager import ConfigManager
from models.file_handler import FileHandler
from models.model_manager import ModelManager
from models.visualization_manager import VisualizationManager
from models.performance_monitor import PerformanceMonitor
from models.ui_manager import UIManager
from models.session_manager import SessionManager

# Load environment variables
load_dotenv()

class DataQnAApp:
    """
    Main application class that orchestrates all components.
    Implements the application logic using the OOP structure.
    """
    def __init__(self):
        # Initialize managers
        self.config_manager = ConfigManager()
        self.file_handler = FileHandler(max_file_size_mb=self.config_manager.max_file_size_mb)
        self.visualization_manager = VisualizationManager()
        self.ui_manager = UIManager(title=self.config_manager.title)
        self.session_manager = SessionManager()
        
        # Get API key from environment variables
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        elif not self.openrouter_api_key.startswith('sk-or-v1-'):
            raise ValueError("Invalid OpenRouter API key format. Key should start with 'sk-or-v1-'")
        
        # Initialize model manager with API key
        self.model_manager = ModelManager(
            api_key=self.openrouter_api_key,
            model_name=self.config_manager.default_model,
            embedding_model_name=self.config_manager.default_embedding
        )
    
    def run(self):
        """
        Run the application.
        """
        # Display header
        self.ui_manager.display_header()
        
        # Display sidebar and get selected section
        selected_section = self.ui_manager.display_sidebar(
            file_type_callback=self.handle_file_type_change,
            file_upload_callback=self.handle_file_upload,
            website_callback=self.handle_website_url,
            youtube_callback=self.handle_youtube_url
        )
        
        # Display selected section
        if selected_section == "Data Chat":
            self.display_data_chat()
        elif selected_section == "Document Chat":
            self.display_document_chat()
        elif selected_section == "Website Chat":
            self.display_website_chat()
        elif selected_section == "YouTube Chat":
            self.display_youtube_chat()
        elif selected_section == "Data Explorer":
            self.display_data_explorer()
        elif selected_section == "Performance Analytics":
            self.display_performance_analytics()
        elif selected_section == "File Manager":
            self.display_file_manager()
    
    def handle_file_type_change(self, file_type):
        """
        Handle file type change in the sidebar.
        """
        # This is a callback for the file type radio button
        # Currently, we don't need to do anything here
        pass
    
    def handle_file_upload(self, uploaded_files):
        """
        Handle file upload in the sidebar.
        """
        session_id = self.session_manager.get_session_id()
        model_key = self.config_manager.default_model
        embedding_key = self.config_manager.default_embedding
        
        for uploaded_file in uploaded_files:
            file_key = f"{session_id}-{uploaded_file.name}-{model_key}-{embedding_key}"
            
            # Only process new files
            if file_key not in st.session_state.uploaded_files:
                self.session_manager.add_uploaded_file(file_key, uploaded_file)
                st.info(f"Processing: {uploaded_file.name}")
                
                try:
                    # Validate file
                    self.file_handler.validate_file(uploaded_file)
                    
                    # Convert file content to text
                    file_content = self.file_handler.read_file_to_text(uploaded_file)
                    if file_content:
                        from llama_index.core import Document
                        docs = [Document(text=file_content)]
                        
                        # Setup LLM & embedding model
                        llm = self.model_manager.load_llm()
                        embed_model = self.model_manager.load_embedding_model()
                        
                        # Create index and query engine
                        index = self.model_manager.create_index_from_documents(docs)
                        query_engine = self.model_manager.create_query_engine(index)
                        
                        # Store in file cache
                        self.session_manager.add_file_to_cache(file_key, query_engine)
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    def handle_website_url(self, url):
        """
        Handle website URL input in the sidebar.
        """
        from models.website_handler import WebsiteHandler
        
        session_id = self.session_manager.get_session_id()
        model_key = self.config_manager.default_model
        embedding_key = self.config_manager.default_embedding
        
        # Create a unique key for this website
        url_key = f"{session_id}-website-{url}-{model_key}-{embedding_key}"
        
        # Only process new websites
        if url_key not in st.session_state.website_urls:
            st.info(f"Processing website: {url}")
            
            try:
                # Initialize website handler
                website_handler = WebsiteHandler()
                
                # Validate and fetch website content
                website_content = website_handler.fetch_website_content(url)
                
                if website_content:
                    # Get website metadata
                    metadata = website_handler.get_website_metadata(url)
                    
                    # Store website info in session state
                    website_info = {
                        "url": url,
                        "content": website_content,
                        "title": metadata.get("title", "Website Content"),
                        "description": metadata.get("description", ""),
                        "key": url_key
                    }
                    
                    self.session_manager.add_website_url(url_key, url, website_content)
                    
                    # Set as current website
                    self.session_manager.set_current_website(website_info)
                    
                    # Create document for indexing
                    from llama_index.core import Document
                    docs = [Document(text=website_content)]
                    
                    # Setup LLM & embedding model
                    llm = self.model_manager.load_llm()
                    embed_model = self.model_manager.load_embedding_model()
                    
                    # Create index and query engine
                    index = self.model_manager.create_index_from_documents(docs)
                    query_engine = self.model_manager.create_query_engine(index)
                    
                    # Store in file cache (using the same cache mechanism)
                    self.session_manager.add_file_to_cache(url_key, query_engine)
                    
                    st.success(f"✅ Website processed successfully!")
                else:
                    st.error("Failed to fetch website content.")
            except Exception as e:
                st.error(f"Error processing website: {str(e)}")
                
    def handle_youtube_url(self, url):
        """
        Handle YouTube URL input in the sidebar.
        """
        from models.youtube_handler import YouTubeHandler
        
        session_id = self.session_manager.get_session_id()
        model_key = self.config_manager.default_model
        embedding_key = self.config_manager.default_embedding
        
        # Create a unique key for this YouTube video
        url_key = f"{session_id}-youtube-{url}-{model_key}-{embedding_key}"
        
        # Only process new YouTube videos
        if url_key not in st.session_state.youtube_videos:
            st.info(f"Processing YouTube video: {url}")
            
            try:
                # Initialize YouTube handler
                youtube_handler = YouTubeHandler()
                
                # Validate and fetch YouTube content
                youtube_content = youtube_handler.fetch_youtube_content(url)
                
                if youtube_content:
                    # Get YouTube metadata
                    metadata = youtube_handler.get_youtube_metadata(url)
                    
                    # Store YouTube info in session state
                    youtube_info = {
                        "url": url,
                        "content": youtube_content,
                        "title": metadata.get("title", "YouTube Video"),
                        "description": metadata.get("description", ""),
                        "thumbnail": metadata.get("thumbnail", ""),
                        "key": url_key
                    }
                    
                    self.session_manager.add_youtube_url(url_key, url, youtube_content)
                    
                    # Set as current YouTube video
                    self.session_manager.set_current_youtube(youtube_info)
                    
                    # Create document for indexing
                    from llama_index.core import Document
                    docs = [Document(text=youtube_content)]
                    
                    # Setup LLM & embedding model
                    llm = self.model_manager.load_llm()
                    embed_model = self.model_manager.load_embedding_model()
                    
                    # Create index and query engine
                    index = self.model_manager.create_index_from_documents(docs)
                    query_engine = self.model_manager.create_query_engine(index)
                    
                    # Store in file cache (using the same cache mechanism)
                    self.session_manager.add_file_to_cache(url_key, query_engine)
                    
                    st.success(f"✅ YouTube video processed successfully!")
                else:
                    st.error("Failed to fetch YouTube content.")
            except Exception as e:
                st.error(f"Error processing YouTube video: {str(e)}")
    
    def display_data_chat(self):
        """
        Display the data chat interface.
        """
        current_file = self.session_manager.get_current_file()
        
        if not current_file or os.path.splitext(current_file.name)[1].lower()[1:] not in self.config_manager.supported_structured_data:
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
            prompt = self.ui_manager.display_chat_interface(
                chat_type="data", 
                current_content=current_file, 
                messages=self.session_manager.get_messages("data")
            )
            
            # Process chat input
            if prompt:
                self.process_chat_input(prompt, "data")
    
    def display_document_chat(self):
        """
        Display the document chat interface.
        """
        current_file = self.session_manager.get_current_file()
        
        if not current_file or os.path.splitext(current_file.name)[1].lower()[1:] not in self.config_manager.supported_documents:
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
            prompt = self.ui_manager.display_chat_interface(
                chat_type="document", 
                current_content=current_file, 
                messages=self.session_manager.get_messages("document")
            )
            
            # Process chat input
            if prompt:
                self.process_chat_input(prompt, "document")
                
    def display_website_chat(self):
        """
        Display the website chat interface.
        """
        current_website = self.session_manager.get_current_website()
        
        if not current_website:
            st.markdown(
                """
                <h1 style="color: #4B8BBE; text-align: center;">🌐 Website Chat</h1>
                <h3 style="text-align: center;">Enter a website URL to begin analysis</h3>
                """,
                unsafe_allow_html=True,
            )
            st.info("Please enter a website URL in the sidebar to start the conversation.")
        else:
            st.markdown(
                """
                <h1 style="color: #4B8BBE; text-align: center;">🌐 Website Chat</h1>
                <h3 style="text-align: center;">Ask questions about the website content</h3>
                """,
                unsafe_allow_html=True,
            )
            
            # Display chat interface with persistent history for websites
            prompt = self.ui_manager.display_chat_interface(
                chat_type="website", 
                current_content=current_website, 
                messages=self.session_manager.get_messages("website")
            )
            
            # Process chat input
            if prompt:
                self.process_chat_input(prompt, "website")
                
    def display_youtube_chat(self):
        """
        Display the YouTube chat interface.
        """
        current_youtube = self.session_manager.get_current_youtube()
        
        if not current_youtube:
            st.markdown(
                """
                <h1 style="color: #4B8BBE; text-align: center;">📺 YouTube Chat</h1>
                <h3 style="text-align: center;">Enter a YouTube URL to begin analysis</h3>
                """,
                unsafe_allow_html=True,
            )
            st.info("Please enter a YouTube URL in the sidebar to start the conversation.")
        else:
            st.markdown(
                """
                <h1 style="color: #4B8BBE; text-align: center;">📺 YouTube Chat</h1>
                <h3 style="text-align: center;">Ask questions about the YouTube video content</h3>
                """,
                unsafe_allow_html=True,
            )
            
            # Display video thumbnail if available
            if current_youtube.get("thumbnail"):
                st.image(current_youtube["thumbnail"], use_column_width=True)
            
            # Display chat interface with persistent history for YouTube videos
            prompt = self.ui_manager.display_chat_interface(
                chat_type="youtube", 
                current_content=current_youtube, 
                messages=self.session_manager.get_messages("youtube")
            )
            
            # Process chat input
            if prompt:
                self.process_chat_input(prompt, "youtube")
    
    def process_chat_input(self, prompt, chat_type):
        """
        Process chat input and generate response.
        """
        if chat_type == "website":
            current_content = self.session_manager.get_current_website()
            content_identifier = current_content.get("url") if current_content else None
            content_name = "website"
        elif chat_type == "youtube":
            current_content = self.session_manager.get_current_youtube()
            content_identifier = current_content.get("url") if current_content else None
            content_name = "youtube"
        else:
            current_content = self.session_manager.get_current_file()
            content_identifier = current_content.name if current_content else None
            content_name = "file_name"
        
        # Add user message with content reference
        new_message = {
            "role": "user",
            "content": prompt,
            "chat_type": chat_type,
            content_name: content_identifier
        }
        self.session_manager.add_message(new_message, chat_type)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            viz_params = None
            
            try:
                # Get the query engine for this content
                session_id = self.session_manager.get_session_id()
                model_key = self.config_manager.default_model
                embedding_key = self.config_manager.default_embedding
                
                if chat_type == "website":
                    content_key = f"{session_id}-website-{current_content.get('url')}-{model_key}-{embedding_key}"
                elif chat_type == "youtube":
                    content_key = f"{session_id}-youtube-{current_content.get('url')}-{model_key}-{embedding_key}"
                else:
                    content_key = f"{session_id}-{current_content.name}-{model_key}-{embedding_key}"
                
                query_engine = self.session_manager.get_file_from_cache(content_key)
                if not query_engine:
                    error_msg = "Website not processed." if chat_type == "website" else "File not indexed. Please try uploading again."
                    st.error(error_msg)
                    st.stop()
                
                # Execute query
                result = self.model_manager.execute_query(query_engine, prompt)
                streaming_response = result["streaming_response"]
                response_time = result["response_time"]
                token_count = result["token_count"]
                
                # Process streaming response
                for chunk in streaming_response.response_gen:
                    full_response += chunk
                    
                    # Check for visualization in the response if it's a data chat
                    if chat_type == "data":
                        current_viz = self.visualization_manager.parse_visualization_request(
                            full_response, 
                            self.file_handler.get_dataframe_from_file(current_file)
                        )
                        if current_viz:
                            viz_params = current_viz
                    
                    # Format the response without showing JSON
                    formatted_response = self.visualization_manager.format_response(full_response)
                    message_placeholder.markdown(
                        formatted_response + "▌", 
                        unsafe_allow_html=True
                    )
                
                # Log performance metrics
                self.session_manager.log_response_time(response_time)
                self.session_manager.log_token_count(token_count)
                
                # Final display of formatted response
                formatted_response = self.visualization_manager.format_response(full_response)
                message_placeholder.markdown(
                    formatted_response,
                    unsafe_allow_html=True
                )
                
                # Store assistant response with content reference
                # Ensure we're not storing HTML tags directly in the message content
                # that could be displayed incorrectly in the chat interface
                clean_response = re.sub(r'</?div[^>]*>', '', formatted_response)
                assistant_message = {
                    "role": "assistant",
                    "content": clean_response,
                    "chat_type": chat_type,
                    content_name: content_identifier
                }
                self.session_manager.add_message(assistant_message, chat_type)
                
                # Generate visualization if params were found and it's a data chat
                if viz_params and chat_type == "data":
                    df = self.file_handler.get_dataframe_from_file(current_file)
                    try:
                        fig = self.visualization_manager.generate_visualization(
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
                
                # Display metrics and feedback options in sidebar
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
                            self.session_manager.log_source_feedback(prompt, source_quality)
                            st.success("Thank you for your feedback!")
                    
                    user_rating = st.slider("Rate answer accuracy (1-5)", 1, 5, 3)
                    if st.button("Submit Rating"):
                        self.session_manager.log_user_rating(prompt, user_rating)
                        st.success("Thank you for your feedback!")
                    
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                message_placeholder.error(error_msg)
                full_response = error_msg
    
    def display_data_explorer(self):
        """
        Display the data explorer interface.
        """
        self.ui_manager.display_data_explorer(
            current_file=self.session_manager.get_current_file(),
            get_dataframe_callback=self.file_handler.get_dataframe_from_file,
            visualization_callback=self.visualization_manager.generate_visualization
        )
    
    def display_performance_analytics(self):
        """
        Display the performance analytics interface.
        """
        st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h1 style="color: var(--primary-color);">📈 Performance Analytics</h1>
                <p style="color: var(--text); opacity: 0.8;">Track and analyze system performance metrics</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display performance metrics
        self.ui_manager.display_performance_metrics(self.session_manager.get_performance_monitor())
    
    def display_file_manager(self):
        """
        Display the file manager interface.
        """
        self.ui_manager.display_file_manager(
            uploaded_files=st.session_state.uploaded_files,
            current_file=self.session_manager.get_current_file(),
            file_select_callback=self.session_manager.set_current_file,
            file_remove_callback=self.session_manager.remove_uploaded_file
        )

# Run the application
if __name__ == "__main__":
    app = DataQnAApp()
    app.run()