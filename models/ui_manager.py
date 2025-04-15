import streamlit as st
import pandas as pd
import altair as alt
import os
from typing import List, Dict, Any, Optional, Callable

class UIManager:
    """
    Manages all UI components and interactions for the application.
    Handles display of chat interfaces, file managers, and performance analytics.
    """
    def __init__(self, title: str = "DataQnA AI"):
        self.title = title
        self.setup_page_config()
        self.load_custom_css()
    
    def setup_page_config(self):
        """
        Configure the Streamlit page settings.
        """
        st.set_page_config(
            page_title=self.title,
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/Naman1995jain/DataQnA.git',
                'Report a bug': "https://github.com/yourusername/dataqna-ai/issues",
                'About': f"# {self.title}\nAn intelligent platform for natural language data analytics."
            }
        )
    
    def load_custom_css(self):
        """
        Load custom CSS styles for the application.
        """
        from pathlib import Path
        css_file = Path(__file__).parent.parent / "styles" / "custom.css"
        
        if css_file.exists():
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
    
    def display_header(self):
        """
        Display the application header.
        """
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
    
    def display_file_upload_zone(self):
        """
        Display the file upload zone in the sidebar.
        """
        st.markdown("""
            <div class="upload-zone">
                <div style="text-align: center;">
                    <p style="font-size: 0.8rem; color: var(--text); opacity: 0.6;">
                        Supported formats: CSV, Excel, JSON, PDF, Website URLs
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    def display_sidebar(self, file_type_callback: Callable, file_upload_callback: Callable, website_callback: Callable = None, youtube_callback: Callable = None):
        """
        Display the sidebar with file upload, website URL input, YouTube URL input, and navigation options.
        """
        with st.sidebar:
            st.markdown("""
                <div style="text-align: center; padding: 1rem 0;">
                    <h2 style="color: var(--primary-color);">⚙️ Configuration</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Display modern file upload zone
            self.display_file_upload_zone()
            
            file_type = st.radio(
                "Select content type",
                ["Structured Data (CSV, Excel, JSON)", "Documents (PDF)", "Website URL", "YouTube URL"],
                key="file_type",
                help="Choose the type of content you want to analyze"
            )
            
            # Call the file type callback
            file_type_callback(file_type)
            
            # Handle different content types
            if file_type == "Website URL" and website_callback:
                # Website URL input
                st.markdown("""
                    <div style="margin-bottom: 1rem;">
                        <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">Enter a website URL to analyze</p>
                    </div>
                """, unsafe_allow_html=True)
                
                website_url = st.text_input(
                    "Website URL",
                    placeholder="https://example.com",
                    help="Enter a valid website URL to analyze its content"
                )
                
                if website_url and st.button("Process Website"):
                    website_callback(website_url)
            elif file_type == "YouTube URL" and youtube_callback:
                # YouTube URL input
                st.markdown("""
                    <div style="margin-bottom: 1rem;">
                        <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">Enter a YouTube URL to analyze</p>
                    </div>
                """, unsafe_allow_html=True)
                
                youtube_url = st.text_input(
                    "YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=example",
                    help="Enter a valid YouTube URL to analyze its content"
                )
                
                if youtube_url and st.button("Process YouTube Video"):
                    youtube_callback(youtube_url)
            elif file_type == "Structured Data (CSV, Excel, JSON)":
                uploaded_files = st.file_uploader(
                    "Choose files",
                    type=["xlsx", "xls", "csv", "json"],
                    accept_multiple_files=True,
                    help="Upload structured data files for analysis and visualization"
                )
                
                # Call the file upload callback
                if uploaded_files:
                    file_upload_callback(uploaded_files)
            else:  # Documents (PDF)
                uploaded_files = st.file_uploader(
                    "Choose PDF documents",
                    type=["pdf"],
                    accept_multiple_files=True,
                    help="Upload PDF documents for text analysis and Q&A"
                )
                
                # Call the file upload callback
                if uploaded_files:
                    file_upload_callback(uploaded_files)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Navigation section
            st.markdown("""
                <div style="text-align: center; padding: 1.5rem 0;">
                    <h2 style="color: var(--primary-color);">Navigation</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Navigation selection
            selected_section = st.radio(
                "Select Section",
                ["Data Chat", "Document Chat", "Website Chat", "YouTube Chat", "Data Explorer", "Performance Analytics", "File Manager"],
                key="navigation"
            )
            
            return selected_section
    
    def display_chat_interface(self, chat_type: str, current_content, messages: List[Dict]):
        """
        Display the chat interface for data, document, website, or YouTube chat.
        current_content can be a file object, a website dictionary, or a YouTube video dictionary.
        """
        if not current_content:
            st.markdown("""
                <div class="card" style="text-align: center; padding: 3rem;">
                    <h2>👋 Welcome to DataQnA AI</h2>
                    <p style="color: var(--text); opacity: 0.8;">Upload a file or enter a website/YouTube URL in the sidebar to start analyzing</p>
                    <div style="max-width: 500px; margin: 2rem auto;">
                        <h3>📄 Get Started</h3>
                        <p>Use the sidebar options to begin</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            return
        
        # Determine content type and name
        if chat_type == "website":
            content_icon = "🌐"
            content_name = current_content.get("url", "Website")
            content_identifier = content_name
            content_key = "url"
        elif chat_type == "youtube":
            content_icon = "📺"
            content_name = current_content.get("url", "YouTube Video")
            content_identifier = content_name
            content_key = "url"
        else:
            content_icon = '📊' if chat_type == 'data' else '📄'
            content_name = current_content.name
            content_identifier = content_name
            content_key = "file_name"
        
        # Display minimalist content context banner
        st.markdown(f"""
            <div class="file-context-banner">
                <div class="file-info">
                    <span class="file-context-icon">{content_icon}</span>
                    <div class="file-context-info">
                        <div class="file-context-name">{content_name}</div>
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
        
        # Get messages for current content
        current_content_messages = [msg for msg in messages 
                               if msg.get(content_key) == content_identifier]
        
        # Display messages with enhanced styling
        for message in current_content_messages:
            role_class = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(f"""
                <div class="chat-message {role_class}">
                    <div class="message-content">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Return the chat input for processing
        chat_prompt = "website" if chat_type == "website" else "data" if chat_type == "data" else "document"
        return st.chat_input(f"What would you like to know about this {chat_prompt}? 💬")
    
    def display_file_manager(self, uploaded_files: Dict, current_file, file_select_callback: Callable, file_remove_callback: Callable):
        """
        Display the file manager interface.
        """
        st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h1 style="color: var(--primary-color);">📁 File Manager</h1>
                <p style="color: var(--text); opacity: 0.8;">Manage your uploaded files</p>
            </div>
        """, unsafe_allow_html=True)
        
        if not uploaded_files:
            st.info("No files uploaded yet. Use the file uploader in the sidebar to get started.")
            return
        
        # Display file list in modern cards
        for file_key, file in uploaded_files.items():
            file_type = "📊" if file.name.lower().endswith((".xlsx", ".xls", ".csv", ".json")) else "📄"
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
                    file_select_callback(file)
                    
            with col2:
                if st.button(f"Remove", key=f"remove_{file_key}"):
                    file_remove_callback(file_key)
    
    def display_performance_metrics(self, performance_monitor):
        """
        Display performance metrics in a modern grid layout.
        """
        st.markdown("""
            <div class="metrics-grid">
        """, unsafe_allow_html=True)
        
        # Calculate metrics
        avg_time = performance_monitor.get_average_response_time()
        total_queries = performance_monitor.get_total_queries()
        avg_tokens = performance_monitor.get_average_token_count()
        avg_rating = performance_monitor.get_average_rating()
        
        if avg_time is not None:
            metrics = [
                {"label": "Total Queries", "value": f"{total_queries}", "icon": "🔍"},
                {"label": "Avg Response Time", "value": f"{avg_time:.2f}s", "icon": "⚡"},
                {"label": "Avg Token Usage", "value": f"{avg_tokens:.0f}" if avg_tokens else "N/A", "icon": "🎯"},
                {"label": "User Satisfaction", "value": f"{avg_rating:.1f}/5" if avg_rating else "N/A", "icon": "⭐"}
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
        
        # Display performance charts
        metrics_df = performance_monitor.get_metrics_dataframe()
        
        # Response time metrics
        if "response_times" in metrics_df:
            times_df = metrics_df["response_times"]
            
            chart = alt.Chart(times_df).mark_line().encode(
                x=alt.X('query:O', title='Query Number'),
                y=alt.Y('time:Q', title='Response Time (s)')
            ).properties(title='Response Times')
            
            st.altair_chart(chart, use_container_width=True)
        
        # Token usage metrics
        if "token_counts" in metrics_df:
            tokens_df = metrics_df["token_counts"]
            
            token_chart = alt.Chart(tokens_df).mark_bar().encode(
                x=alt.X('query:O', title='Query Number'),
                y=alt.Y('tokens:Q', title='Token Count')
            ).properties(title='Token Usage per Query')
            
            st.altair_chart(token_chart, use_container_width=True)
        
        # User ratings visualization
        if "ratings" in metrics_df:
            rating_df = metrics_df["ratings"]
            rating_chart = alt.Chart(rating_df).mark_bar().encode(
                x=alt.X('rating:O', title='Rating'),
                y=alt.Y('count()', title='Count')
            ).properties(title='User Accuracy Ratings')
            
            st.altair_chart(rating_chart, use_container_width=True)
        
        # Source feedback visualization
        if "source_feedback" in metrics_df:
            source_df = metrics_df["source_feedback"]
            
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
        
        # Export performance data
        if avg_time is not None:
            st.subheader("Export Performance Data")
            export_df = performance_monitor.export_data()
            
            # Convert to CSV for download
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Performance Data as CSV",
                data=csv,
                file_name="rag_performance_data.csv",
                mime="text/csv"
            )
    
    def display_data_explorer(self, current_file, get_dataframe_callback: Callable, visualization_callback: Callable):
        """
        Display the data explorer interface.
        """
        st.markdown(
            """
            <h1 style="color: #4B8BBE; text-align: center;">📊 Data Explorer</h1>
            """,
            unsafe_allow_html=True,
        )
        
        # Data Explorer code
        if current_file:
            # Get the DataFrame
            df = get_dataframe_callback(current_file)
            
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
                    fig = visualization_callback(
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
                    fig = visualization_callback(
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
                            fig = visualization_callback(
                                df=corr_matrix,
                                viz_type="heatmap",
                                x_col=x_col,
                                title="Correlation Matrix"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating correlation heatmap: {str(e)}")
                else:
                    # Cross tabulation between two categorical columns
                    y_col = st.selectbox("Y-axis Column", options=df.columns)
                    try:
                        cross_tab = pd.crosstab(df[x_col], df[y_col])
                        fig = visualization_callback(
                            df=cross_tab,
                            viz_type="heatmap",
                            x_col=x_col,
                            y_col=y_col,
                            title=f"Cross Tabulation: {x_col} vs {y_col}"
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
    
    def show_loading_animation(self):
        """
        Display an animated loading indicator.
        """
        st.markdown("""
            <div class="loading-container" style="text-align: center; padding: 2rem;">
                <div class="loading"></div>
                <p style="color: var(--text); opacity: 0.8; margin-top: 1rem;">
                    Processing your request...
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    def show_file_processing_progress(self, file_name: str, progress: int):
        """
        Show file processing progress with animated bar.
        """
        st.markdown(f"""
            <div class="progress-container">
                <p style="margin-bottom: 0.5rem;">Processing: {file_name}</p>
                <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: {progress}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    def show_success_toast(self, message: str):
        """
        Show a success toast notification.
        """
        st.markdown(f"""
            <div class="toast success-toast">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: var(--success-color);">✓</span>
                    <span>{message}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    def show_error_toast(self, message: str):
        """
        Show an error toast notification.
        """
        st.markdown(f"""
            <div class="toast error-toast">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: var(--error-color);">⚠</span>
                    <span>{message}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    def custom_metric(self, label: str, value: str, delta: Optional[str] = None):
        """
        Display a custom metric component.
        """
        html = f"""
            <div class="custom-metric">
                <h3>{label}</h3>
                <h2>{value}</h2>
                {f'<p>{delta}</p>' if delta else ''}
            </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        
        