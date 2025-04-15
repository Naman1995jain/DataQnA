import time
import pandas as pd
from typing import List, Dict, Any, Optional

class PerformanceMonitor:
    """
    Tracks and manages performance metrics for the application.
    Includes response times, token usage, user ratings, and source feedback.
    """
    def __init__(self):
        self.start_time = time.time()
        self.metrics = []
        self.response_times = []
        self.token_counts = []
        self.ratings = []
        self.source_feedback = []
    
    def log_metric(self, name: str, value: float):
        """
        Log a generic metric with name and value.
        """
        self.metrics.append({
            "name": name,
            "value": value,
            "timestamp": time.time()
        })
    
    def log_response_time(self, response_time: float):
        """
        Log a response time metric.
        """
        self.response_times.append(response_time)
        self.log_metric("response_time", response_time)
    
    def log_token_count(self, token_count: int):
        """
        Log a token count metric.
        """
        if token_count > 0:
            self.token_counts.append(token_count)
            self.log_metric("token_count", token_count)
    
    def log_user_rating(self, query: str, rating: int):
        """
        Log a user rating for a query.
        """
        self.ratings.append({
            "query": query,
            "rating": rating
        })
        self.log_metric("user_rating", rating)
    
    def log_source_feedback(self, query: str, rating: str):
        """
        Log feedback about source relevance.
        """
        self.source_feedback.append({
            "query": query,
            "rating": rating
        })
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary of all metrics as a DataFrame.
        """
        return pd.DataFrame(self.metrics)
    
    def get_average_response_time(self) -> Optional[float]:
        """
        Get the average response time.
        """
        if not self.response_times:
            return None
        return sum(self.response_times) / len(self.response_times)
    
    def get_average_token_count(self) -> Optional[float]:
        """
        Get the average token count.
        """
        if not self.token_counts:
            return None
        return sum(self.token_counts) / len(self.token_counts)
    
    def get_average_rating(self) -> Optional[float]:
        """
        Get the average user rating.
        """
        if not self.ratings:
            return None
        return sum(r["rating"] for r in self.ratings) / len(self.ratings)
    
    def get_total_queries(self) -> int:
        """
        Get the total number of queries processed.
        """
        return len(self.response_times)
    
    def get_metrics_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Get DataFrames for different metrics for visualization.
        """
        result = {}
        
        if self.response_times:
            result["response_times"] = pd.DataFrame({
                "query": range(1, len(self.response_times) + 1),
                "time": self.response_times
            })
        
        if self.token_counts:
            result["token_counts"] = pd.DataFrame({
                "query": range(1, len(self.token_counts) + 1),
                "tokens": self.token_counts
            })
        
        if self.ratings:
            result["ratings"] = pd.DataFrame({
                "rating": [r["rating"] for r in self.ratings]
            })
        
        if self.source_feedback:
            from collections import Counter
            rating_counts = Counter([r["rating"] for r in self.source_feedback])
            result["source_feedback"] = pd.DataFrame({
                "rating": list(rating_counts.keys()),
                "count": list(rating_counts.values())
            })
        
        return result
    
    def export_data(self) -> pd.DataFrame:
        """
        Export performance data as a DataFrame for download.
        """
        if not self.response_times:
            return pd.DataFrame()
        
        export_df = pd.DataFrame({
            "query_num": range(1, len(self.response_times) + 1),
            "response_time": self.response_times,
        })
        
        if self.token_counts and len(self.token_counts) == len(self.response_times):
            export_df["token_count"] = self.token_counts
        
        if self.ratings and len(self.ratings) == len(self.response_times):
            export_df["user_rating"] = [r["rating"] for r in self.ratings]
        
        return export_df