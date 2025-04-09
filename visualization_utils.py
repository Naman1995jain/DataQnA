import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json

def get_data_summary(df):
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

def parse_visualization_request(response_text, df):
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

def generate_visualization(df, viz_type, x_col, y_col=None, color_col=None, title=None):
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
                markers=True  # Show points
            )
            # Add range slider
            fig.update_xaxes(rangeslider_visible=True)
        
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
                size=None if not pd.api.types.is_numeric_dtype(df[x_col]) else x_col,
                trendline="ols" if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]) else None
            )
            # Add zoom capabilities
            fig.update_layout(dragmode="zoom")
        
        elif viz_type == "pie":
            if y_col is None:
                # Count occurrences of each category
                counts = df[x_col].value_counts()
                values = counts.values
                names = counts.index
            else:
                values = df[y_col]
                names = df[x_col]
            
            fig = px.pie(
                values=values,
                names=names,
                title=title,
                hole=0.4,  # Create a donut chart
                labels={x_col: x_col}
            )
            # Add percentage and value in hover
            fig.update_traces(
                textinfo="percent+value",
                hovertemplate="%{label}: %{value:,.0f} (%{percent})<extra></extra>"
            )
        
        elif viz_type == "histogram":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_col,
                title=title,
                labels={x_col: x_col},
                marginal="box",  # Add box plot on top
                histnorm='percent' if color_col else None  # Normalize if comparing groups
            )
            # Add KDE curve
            fig.update_traces(opacity=0.7)
        
        elif viz_type == "heatmap":
            if y_col is None:
                # Create correlation matrix
                numeric_df = df.select_dtypes(include=['number'])
                corr_matrix = numeric_df.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title=title or "Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                    zmin=-1,
                    zmax=1
                )
                # Add correlation values
                fig.update_traces(
                    text=corr_matrix.round(2),
                    texttemplate="%{text}"
                )
            else:
                # Create cross-tabulation
                cross_tab = pd.crosstab(df[x_col], df[y_col], normalize='all') * 100
                fig = px.imshow(
                    cross_tab,
                    title=title or f"Heatmap: {x_col} vs {y_col}",
                    color_continuous_scale="Viridis",
                    aspect="auto"
                )
                # Add percentage values
                fig.update_traces(
                    text=cross_tab.round(1),
                    texttemplate="%{text}%"
                )
        
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        # Apply common layout settings
        fig.update_layout(**layout_settings)
        
        # Add watermark
        fig.add_annotation(
            text="DataQnA AI",
            x=0.99,
            y=0.01,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=10, color="gray"),
            opacity=0.7
        )
        
        # Enable responsive behavior
        fig.update_layout(
            autosize=True,
            height=500,
        )
        
        return fig
    
    except Exception as e:
        raise ValueError(f"Error generating {viz_type} visualization: {str(e)}")
