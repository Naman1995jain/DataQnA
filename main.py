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

import streamlit as st

# Import visualization utilities
from visualization_utils import generate_visualization, parse_visualization_request, get_data_summary

# Load environment variables
load_dotenv()

# Get API key from environment variables
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# -----------------------------
# Session & State Management
# -----------------------------
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    st.session_state.response_times = []
    st.session_state.token_counts = []
    st.session_state.ratings = []
    st.session_state.source_feedback = []
    st.session_state.current_file = None

session_id = st.session_state.id

# Add configuration class
class Config:
    DEFAULT_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"
    DEFAULT_EMBEDDING = "BAAI/bge-large-en-v1.5"
    SUPPORTED_FILE_TYPES = ["xlsx", "xls", "csv"]
    MAX_FILE_SIZE_MB = 300
    TOKEN_LIMIT = 9000

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
    llm = OpenRouter(
        api_key=api_key,
        model=model_name,
        request_timeout=10000.0
    )
    return llm

def display_file(file):
    # Function kept for compatibility but doesn't display preview
    pass

def read_file_to_text(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == ".csv":
            df = pd.read_csv(file)
        else:  # Excel files
            df = pd.read_excel(file)
        
        # Get data summary
        data_summary = get_data_summary(df)
        
        # Convert DataFrame to a formatted string
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        
        # Only include a sample of the data to avoid token limits
        df_sample = df.head(20).to_string()
        
        return f"Data Summary:\n{data_summary}\n\nDataFrame Info:\n{df_info}\n\nSample Data (first 20 rows):\n{df_sample}"
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def format_response(text):
    """Format response to show output only"""
    # Split the response into parts
    parts = text.split("```")
    formatted = text
    
    # If there are code blocks, extract only the output
    if len(parts) > 1:
        output_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text
                output_parts.append(part)
            else:  # Code block
                # Check if this is a visualization block
                if part.strip().startswith("visualization"):
                    # Don't include visualization blocks in the output
                    continue
                
                # Remove the language identifier if present
                lines = part.split('\n')
                if len(lines) > 1:
                    # Skip language identifier line and empty lines
                    content = '\n'.join(line for line in lines[1:] if line.strip())
                    if content:
                        output_parts.append(f'📊 Output:\n{content}')
        formatted = '\n\n'.join(output_parts)
    
    return formatted

def get_dataframe_from_file(file):
    """Load a DataFrame from the uploaded file"""
    file_extension = os.path.splitext(file.name)[1].lower()
    file.seek(0)  # Reset file pointer
    
    if file_extension == ".csv":
        return pd.read_csv(file)
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

# Load custom CSS
def load_custom_css():
    css_file = Path(__file__).parent / "styles" / "custom.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# -----------------------------
# Sidebar - Config & Navigation
# -----------------------------
with st.sidebar:
    # Configuration section first
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="color: var(--primary-color);">⚙️ Configuration</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("📂 Upload Documents")
    uploaded_file = st.file_uploader("Choose a file (.xlsx, .xls, or .csv)", type=["xlsx", "xls", "csv"])
    
    st.markdown("### 🧠 Model")
    st.info("Using Mistral Small (Free)")
    model_key = "mistralai/mistral-small-3.1-24b-instruct:free"
    
    # Fixed embedding model
    st.markdown("### 🔍 Embedding Model")
    st.info("Using BAAI/bge-large-en-v1.5")
    embedding_key = "BAAI/bge-large-en-v1.5"
    
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
        ["Chat", "Data Explorer", "Performance Analytics"],
        key="navigation"
    )

    # Store the uploaded file in session state
    if uploaded_file:
        st.session_state.current_file = uploaded_file
        file_key = f"{session_id}-{uploaded_file.name}-{model_key}-{embedding_key}"
        st.info("Indexing your document...")

        if file_key not in st.session_state.get('file_cache', {}):
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

                    # Create the query engine with streaming responses
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    # ====== Customize prompt template ======
                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above, follow these rules:\n"
                        "1. Focus on showing results and insights\n"
                        "2. Present numerical results in a clear format\n"
                        "3. Use bullet points for lists\n"
                        "4. Keep explanations concise\n"
                        "5. Don't show code implementation details\n"
                        "6. When appropriate, suggest a visualization by including a JSON block like this:\n"
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
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    st.session_state.file_cache[file_key] = query_engine

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()
        else:
            query_engine = st.session_state.file_cache[file_key]

        st.success("✅ Ready to Chat!")

# -----------------------------
# Main Content Area
# -----------------------------
if selected_section == "Chat":
    st.markdown(
        """
        <h1 style="color: #4B8BBE; text-align: center;">🤖 AI Data Analyst</h1>
        <h3 style="text-align: center;">Ask questions about your data in natural language</h3>
        """,
        unsafe_allow_html=True,
    )
    
    # Chat interface code
    # Display previous chat messages and chat input
    for message in st.session_state.messages:
        message_class = "user-message" if message["role"] == "user" else "assistant-message"
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="chat-message {message_class}">{message["content"]}</div>', 
                       unsafe_allow_html=True)

    # Chat input processing
    if prompt := st.chat_input("What would you like to know about your data? 💬"):
        if not st.session_state.current_file:
            st.warning("Please upload a file first!")
            st.stop()
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Initialize metrics
            start_time = time.time()
            token_count = 0
            
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
                
                # Execute query
                streaming_response = query_engine.query(prompt)
                
                # Process streaming response
                for chunk in streaming_response.response_gen:
                    full_response += chunk
                    # Format the response with enhanced code block handling
                    formatted_response = format_response(full_response)
                    message_placeholder.markdown(
                        formatted_response + "▌", 
                        unsafe_allow_html=True
                    )
                
                # Final display with syntax highlighting
                message_placeholder.markdown(
                    format_response(full_response),
                    unsafe_allow_html=True
                )
                
                # Check for visualization requests in the response
                if st.session_state.current_file:
                    # Get the DataFrame
                    df = get_dataframe_from_file(st.session_state.current_file)
                    
                    # Parse visualization request
                    viz_params = parse_visualization_request(full_response, df)
                    if viz_params:
                        st.subheader("📊 Data Visualization")
                        try:
                            fig = generate_visualization(
                                df=df,
                                viz_type=viz_params["type"],
                                x_col=viz_params["x_column"],
                                y_col=viz_params.get("y_column"),
                                color_col=viz_params.get("color_column"),
                                title=viz_params.get("title", "Data Visualization")
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
                
                # Calculate and store metrics
                response_time = time.time() - start_time
                st.session_state.response_times.append(response_time)
                
                # Get token count if available
                if hasattr(token_counter, 'total_llm_token_count'):
                    token_count = token_counter.total_llm_token_count
                    if token_count > 0:
                        st.session_state.token_counts.append(token_count)
                
                # Display metrics in sidebar
                with st.sidebar:
                    st.markdown("### ⚙️ Response Metrics")
                    st.info(f"Response time: {response_time:.2f} seconds")
                    if token_count > 0:
                        st.info(f"Tokens used: {token_count}")
                    
                    # Source feedback - only if sources are available
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
                    
                    # Response quality rating
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
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})

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

else:  # Performance Analytics
    st.markdown(
        """
        <h1 style="color: #4B8BBE; text-align: center;">📈 Performance Analytics</h1>
        """,
        unsafe_allow_html=True,
    )
    
    # Performance Analytics code
    # Response time metrics, token usage, etc.
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h2 style="color: var(--primary-color);">📊 Performance Analytics</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Response time metrics
    if st.session_state.response_times:
        avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
        st.metric("Average Response Time", f"{avg_time:.2f}s")
        
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
        avg_tokens = sum(st.session_state.token_counts) / len(st.session_state.token_counts)
        st.metric("Average Token Usage", f"{avg_tokens:.0f} tokens")
        
        # Plot token usage
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
        avg_rating = sum(ratings) / len(ratings)
        st.metric("Average Accuracy Rating", f"{avg_rating:.1f}/5")
        
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
        # ...existing performance metrics code...
    
    # Export performance data
    if st.session_state.response_times:
        st.subheader("Export Performance Data")
        
        # Prepare data for export
        export_data = {
            "response_times": st.session_state.response_times,
            "model_used": model_key
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
            "model": [model_key] * len(st.session_state.response_times)
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
