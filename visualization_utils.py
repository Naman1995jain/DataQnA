import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import json

def get_data_summary(df):
    """Generate a summary of the dataframe"""
    summary = []
    
    # Basic info
    summary.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Column types
    num_numeric = len(df.select_dtypes(include=['number']).columns)
    num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
    num_datetime = len(df.select_dtypes(include=['datetime']).columns)
    
    summary.append(f"Contains {num_numeric} numeric columns, {num_categorical} categorical columns, and {num_datetime} datetime columns.")
    
    # Missing values
    missing_values = df.isna().sum().sum()
    if missing_values > 0:
        missing_percent = (missing_values / (df.shape[0] * df.shape[1])) * 100
        summary.append(f"Dataset has {missing_values} missing values ({missing_percent:.2f}% of all cells).")
    else:
        summary.append("Dataset has no missing values.")
    
    # Numeric column stats
    if num_numeric > 0:
        numeric_cols = df.select_dtypes(include=['number']).columns
        summary.append("\nNumeric column statistics:")
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            summary.append(f"- {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}, median={df[col].median():.2f}")
        if len(numeric_cols) > 5:
            summary.append(f"- ... and {len(numeric_cols) - 5} more numeric columns")
    
    # Categorical column stats
    if num_categorical > 0:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        summary.append("\nCategorical column statistics:")
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            unique_count = df[col].nunique()
            top_value = df[col].value_counts().index[0] if not df[col].value_counts().empty else "N/A"
            summary.append(f"- {col}: {unique_count} unique values, most common: '{top_value}'")
        if len(categorical_cols) > 5:
            summary.append(f"- ... and {len(categorical_cols) - 5} more categorical columns")
    
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
    """Generate a visualization based on the specified parameters"""
    if title is None:
        title = f"{viz_type.capitalize()} Chart"
    
    # Handle different visualization types
    if viz_type == "bar":
        if y_col is None:
            # Count plot (frequency of each category)
            fig = px.bar(
                df, 
                x=x_col, 
                title=title,
                color=color_col,
                labels={x_col: x_col}
            )
        else:
            fig = px.bar(
                df, 
                x=x_col, 
                y=y_col, 
                title=title,
                color=color_col,
                labels={x_col: x_col, y_col: y_col}
            )
    
    elif viz_type == "line":
        if y_col is None:
            raise ValueError("Line charts require both x and y columns")
        
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col, 
            title=title,
            color=color_col,
            labels={x_col: x_col, y_col: y_col}
        )
    
    elif viz_type == "scatter":
        if y_col is None:
            raise ValueError("Scatter plots require both x and y columns")
        
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            title=title,
            color=color_col,
            labels={x_col: x_col, y_col: y_col}
        )
    
    elif viz_type == "pie":
        # For pie charts, x_col is the categories and y_col is the values
        if y_col is None:
            # Count occurrences of each category
            value_counts = df[x_col].value_counts().reset_index()
            value_counts.columns = [x_col, 'count']
            fig = px.pie(
                value_counts, 
                names=x_col, 
                values='count', 
                title=title
            )
        else:
            # Use y_col for values
            fig = px.pie(
                df, 
                names=x_col, 
                values=y_col, 
                title=title
            )
    
    elif viz_type == "histogram":
        fig = px.histogram(
            df, 
            x=x_col, 
            color=color_col,
            title=title,
            labels={x_col: x_col}
        )
    
    elif viz_type == "heatmap":
        if y_col is None:
            # If no y_col is provided, create a correlation matrix of numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            corr_matrix = numeric_df.corr()
            fig = px.imshow(
                corr_matrix,
                title=title or "Correlation Matrix",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
        else:
            # Create a cross-tabulation between x_col and y_col
            cross_tab = pd.crosstab(df[x_col], df[y_col])
            fig = px.imshow(
                cross_tab,
                title=title or f"Heatmap: {x_col} vs {y_col}",
                color_continuous_scale="Viridis"
            )
    
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")
    
    # Enhance the figure with better styling
    fig.update_layout(
        template="plotly_white",
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
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
        )
    )
    
    return fig
