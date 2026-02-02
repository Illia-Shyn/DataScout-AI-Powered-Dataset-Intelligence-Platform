import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class AutoVisualizer:
    """
    Chart Selection Rules:
        - Single numeric → Histogram
        - Single categorical (cardinality < 20) → Bar chart
        - Datetime + numeric → Time series line
        - Two numeric → Scatter plot
        - All numeric columns → Correlation heatmap
    """
    
    def __init__(self, df: pd.DataFrame, column_types: dict):
        self.df = df
        self.column_types = column_types
    
    def generate_all_charts(self) -> list:
        """
        Returns:
            list: [
                {
                    'title': str,
                    'chart_type': str,
                    'figure': plotly.graph_objects.Figure,
                    'insight': str  # Optional auto-insight
                }
            ]
        """
        charts = []
        
        # 1. Univariate Analysis (Distributions & Counts)
        for col, info in self.column_types.items():
            if info['recommended_viz'] == 'histogram':
                fig = self.create_distribution_chart(col)
                if fig:
                    charts.append({
                        'title': f'Distribution of {col}',
                        'chart_type': 'histogram',
                        'figure': fig,
                        'insight': f'Distribution analysis for {col}'
                    })
            elif info['recommended_viz'] == 'bar':
                fig = self.create_categorical_chart(col)
                if fig:
                    charts.append({
                        'title': f'Count of {col}',
                        'chart_type': 'bar',
                        'figure': fig,
                        'insight': f'Category breakdown for {col}'
                    })
        
        # 2. Bivariate Analysis (Time Series)
        # Find datetime columns
        datetime_cols = [col for col, info in self.column_types.items() if info['inferred_type'] == 'datetime']
        numeric_cols = [col for col, info in self.column_types.items() if info['inferred_type'] == 'numeric']
        
        if datetime_cols and numeric_cols:
             date_col = datetime_cols[0] # Take primary date column
             for num_col in numeric_cols[:3]: # Limit to first 3 to avoid clutter
                 fig = self.create_time_series_chart(date_col, num_col)
                 if fig:
                     charts.append({
                         'title': f'{num_col} over Time',
                         'chart_type': 'line',
                         'figure': fig,
                         'insight': f'Trend of {num_col} over time'
                     })

        # 3. Multivariate Analysis (Correlation)
        if len(numeric_cols) > 1:
            fig = self.create_correlation_heatmap()
            if fig:
                charts.append({
                    'title': 'Feature Correlation Heatmap',
                    'chart_type': 'heatmap',
                    'figure': fig,
                    'insight': 'Correlation between numeric features'
                })
                
        return charts
    
    def create_distribution_chart(self, column: str) -> go.Figure:
        try:
            fig = px.histogram(self.df, x=column, marginal="box", title=f"Distribution of {column}")
            fig.update_layout(template="plotly_white")
            return fig
        except Exception:
            return None
    
    def create_categorical_chart(self, column: str) -> go.Figure:
        try:
            counts = self.df[column].value_counts().reset_index()
            counts.columns = [column, 'count']
            fig = px.bar(counts, x=column, y='count', title=f"Count of {column}")
            fig.update_layout(template="plotly_white")
            return fig
        except Exception:
            return None
    
    def create_time_series_chart(self, date_col: str, num_col: str) -> go.Figure:
        try:
            # Aggregate if there are multiple entries per date, or just plot raw
            # For robustness, let's sort by date
            df_sorted = self.df.sort_values(by=date_col)
            fig = px.line(df_sorted, x=date_col, y=num_col, title=f"{num_col} over {date_col}")
            fig.update_layout(template="plotly_white")
            return fig
        except Exception:
            return None

    def create_correlation_heatmap(self) -> go.Figure:
        try:
            numeric_df = self.df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale='RdBu_r')
            fig.update_layout(template="plotly_white")
            return fig
        except Exception:
            return None
