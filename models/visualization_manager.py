import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json
from typing import Optional, Dict, Any, List, Union

class VisualizationManager:
    """
    Manages data visualization operations including generating charts,
    parsing visualization requests, and providing data summaries.
    """
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> str:
        """Generate an enhanced summary of the dataframe with modern formatting"""
        summary = []
        
        # Basic dataset information
        total_rows = df.shape[0]
        total_cols = df.shape[1]
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert to MB
        
        summary.append(f"📊 Dataset Overview")
        summary.append(f"├─ {total_rows:,} rows")
        summary.append(f"├─ {total_cols} columns")
        summary.append(f"└─ {memory_usage:.2f} MB memory usage")
        summary.append("")
        
        # Column type distribution
        num_numeric = len(df.select_dtypes(include=['number']).columns)
        num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        num_datetime = len(df.select_dtypes(include=['datetime']).columns)
        num_boolean = len(df.select_dtypes(include=['bool']).columns)
        
        summary.append("📋 Column Types")
        summary.append(f"├─ {num_numeric} numeric columns")
        summary.append(f"├─ {num_categorical} categorical columns")
        summary.append(f"├─ {num_datetime} datetime columns")
        summary.append(f"└─ {num_boolean} boolean columns")
        summary.append("")
        
        # Missing values analysis
        missing_values = df.isna().sum()
        cols_with_missing = missing_values[missing_values > 0]
        if len(cols_with_missing) > 0:
            summary.append("⚠️ Missing Values")
            for col, count in cols_with_missing.items():
                percent = (count / total_rows) * 100
                summary.append(f"├─ {col}: {count:,} missing ({percent:.1f}%)")
            summary.append("")
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary.append("📈 Numeric Column Statistics")
            for col in numeric_cols:
                stats = df[col].describe()
                summary.append(f"├─ {col}")
                summary.append(f"│  ├─ Range: {stats['min']:.2f} to {stats['max']:.2f}")
                summary.append(f"│  ├─ Mean: {stats['mean']:.2f}")
                summary.append(f"│  ├─ Median: {stats['50%']:.2f}")
                summary.append(f"│  └─ Std Dev: {stats['std']:.2f}")
            summary.append("")
        
        # Categorical column analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary.append("📑 Categorical Column Analysis")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(3)
                summary.append(f"├─ {col}")
                summary.append(f"│  ├─ {unique_count:,} unique values")
                summary.append(f"│  └─ Top values:")
                for val, count in top_values.items():
                    percent = (count / total_rows) * 100
                    summary.append(f"│     └─ {val}: {count:,} ({percent:.1f}%)")
            summary.append("")
        
        # Datetime column analysis
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        if len(datetime_cols) > 0:
            summary.append("📅 Datetime Column Analysis")
            for col in datetime_cols:
                summary.append(f"├─ {col}")
                summary.append(f"│  ├─ Range: {df[col].min()} to {df[col].max()}")
                summary.append(f"│  └─ Distinct dates: {df[col].nunique():,}")
            summary.append("")
        
        return "\n".join(summary)
    
    @staticmethod
    def parse_visualization_request(response_text: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Extract visualization parameters from the response text"""
        # Regular expression to find the visualization JSON block
        viz_pattern = r"```visualization\s*([\s\S]*?)\s*```"
        match = re.search(viz_pattern, response_text)
        
        if not match:
            return None
        
        try:
            # Extract and parse the JSON
            viz_json = match.group(1)
            # Remove any comments from the JSON
            viz_json = re.sub(r'#.*$', '', viz_json, flags=re.MULTILINE)
            viz_params = json.loads(viz_json)
            
            # Validate required parameters
            if "type" not in viz_params or "x_column" not in viz_params:
                return None
            
            # Validate column names
            if viz_params["x_column"] not in df.columns:
                return None
            
            if "y_column" in viz_params and viz_params["y_column"] not in df.columns:
                return None
            
            if "color_column" in viz_params and viz_params["color_column"] not in df.columns:
                return None
            
            return viz_params
        except Exception as e:
            print(f"Error parsing visualization request: {e}")
            return None
    
    @staticmethod
    def generate_visualization(df: pd.DataFrame, viz_type: str, x_col: str, 
                              y_col: Optional[str] = None, color_col: Optional[str] = None, 
                              title: Optional[str] = None):
        """Generate an enhanced visualization with modern styling and interactivity"""
        if title is None:
            title = f"{viz_type.capitalize()} Chart"
        
        # Common layout settings
        layout_settings = dict(
            template="plotly_white",
            title=dict(
                text=title,
                y=0.95,
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(size=20)
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial, sans-serif"
            ),
            modebar=dict(
                bgcolor="rgba(0,0,0,0)",
                color="rgba(0,0,0,0.3)",
                activecolor="rgba(0,0,0,0.6)"
            )
        )
        
        try:
            if viz_type == "bar":
                if y_col is None:
                    # Count plot (frequency of each category)
                    counts = df[x_col].value_counts().reset_index()
                    counts.columns = [x_col, 'count']
                    fig = px.bar(
                        counts,
                        x=x_col,
                        y='count',
                        title=title,
                        color=color_col,
                        labels={x_col: x_col, 'count': 'Frequency'},
                        text='count'  # Show values on bars
                    )
                    fig.update_traces(textposition='outside')
                else:
                    fig = px.bar(
                        df,
                        x=x_col,
                        y=y_col,
                        title=title,
                        color=color_col,
                        labels={x_col: x_col, y_col: y_col},
                        barmode='group' if color_col else None,
                        text=y_col  # Show values on bars
                    )
                    fig.update_traces(textposition='outside')
            
            elif viz_type == "line":
                if y_col is None:
                    raise ValueError("Line charts require both x and y columns")
                
                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    title=title,
                    color=color_col,
                    labels={x_col: x_col, y_col: y_col},
                    line_shape="spline",  # Smooth lines
                    markers=True  # Show markers
                )
                
                # Add hover information
                fig.update_traces(hovertemplate='%{x}: %{y}')
            
            elif viz_type == "scatter":
                if y_col is None:
                    raise ValueError("Scatter plots require both x and y columns")
                
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=title,
                    color=color_col,
                    labels={x_col: x_col, y_col: y_col},
                    opacity=0.7,  # Semi-transparent points
                    size_max=15,  # Maximum marker size
                    hover_name=df.index if df.index.name else None
                )
                
                # Add trendline
                fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            
            elif viz_type == "pie":
                # Count values for the pie chart
                value_counts = df[x_col].value_counts().reset_index()
                value_counts.columns = [x_col, 'count']
                
                fig = px.pie(
                    value_counts,
                    names=x_col,
                    values='count',
                    title=title,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.4,  # Create a donut chart
                    labels={x_col: x_col, 'count': 'Count'}
                )
                
                # Add percentage and value in hover
                fig.update_traces(textposition='inside', textinfo='percent+label')
            
            elif viz_type == "histogram":
                fig = px.histogram(
                    df,
                    x=x_col,
                    color=color_col,
                    title=title,
                    opacity=0.8,
                    marginal="box",  # Add box plot on the marginal
                    histnorm="probability density" if df[x_col].nunique() > 10 else None,
                    labels={x_col: x_col}
                )
                
                # Add mean line
                if pd.api.types.is_numeric_dtype(df[x_col]):
                    mean_value = df[x_col].mean()
                    fig.add_vline(x=mean_value, line_dash="dash", line_color="red", 
                                annotation_text=f"Mean: {mean_value:.2f}", 
                                annotation_position="top right")
            
            elif viz_type == "heatmap":
                # For heatmap, we need numeric data in a matrix form
                if y_col is not None:
                    # Create a cross-tabulation if y_col is provided
                    heatmap_data = pd.crosstab(df[x_col], df[y_col])
                else:
                    # Otherwise, use correlation matrix of numeric columns
                    numeric_df = df.select_dtypes(include=['number'])
                    heatmap_data = numeric_df.corr()
                
                fig = px.imshow(
                    heatmap_data,
                    title=title,
                    color_continuous_scale="RdBu_r",
                    labels=dict(x=x_col, y=y_col if y_col else x_col, color="Value")
                )
                
                # Add text annotations
                fig.update_traces(text=heatmap_data.round(2), texttemplate="%{text}")
            
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            
            # Apply common layout settings
            fig.update_layout(**layout_settings)
            
            return fig
        
        except Exception as e:
            raise ValueError(f"Error generating {viz_type} visualization: {str(e)}")
    
    @staticmethod
    def format_response(text: str) -> str:
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