import pandas as pd
import numpy as np

class DataProfiler:
    def __init__(self, df: pd.DataFrame):
        """Initialize with dataframe."""
        self.df = df
    
    def generate_profile(self) -> dict:
        """
        Generates a comprehensive profile of the dataframe.
        
        Returns:
            dict: {
                'overview': {
                    'n_rows': int,
                    'n_cols': int,
                    'memory_usage': str,
                    'duplicates': int,
                    'total_missing': int
                },
                'columns': {
                    'column_name': {
                        'dtype': str,
                        'missing': int,
                        'missing_pct': float,
                        'unique': int,
                        'cardinality': str  # 'low', 'medium', 'high'
                    }
                },
                'numeric_stats': pd.DataFrame,
                'categorical_stats': pd.DataFrame,
                'correlations': pd.DataFrame
            }
        """
        profile = {
            'overview': self._get_overview(),
            'columns': self._get_column_details(),
            'numeric_stats': self.df.describe(),
            'categorical_stats': self.df.describe(include=['O']) if self.df.select_dtypes(include=['O']).shape[1] > 0 else pd.DataFrame(),
            'correlations': self.df.select_dtypes(include=[np.number]).corr() if self.df.select_dtypes(include=[np.number]).shape[1] > 1 else pd.DataFrame()
        }
        return profile

    def _get_overview(self) -> dict:
        """Helper to get dataset overview."""
        return {
            'n_rows': self.df.shape[0],
            'n_cols': self.df.shape[1],
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            'duplicates': self.df.duplicated().sum(),
            'total_missing': self.df.isnull().sum().sum()
        }

    def _get_column_details(self) -> dict:
        """Helper to get detailed column stats."""
        details = {}
        for col in self.df.columns:
            n_unique = self.df[col].nunique()
            n_rows = self.df.shape[0]
            if n_rows > 0:
                cardinality_ratio = n_unique / n_rows
                if cardinality_ratio < 0.05:
                    cardinality = 'low'
                elif cardinality_ratio < 0.2:
                    cardinality = 'medium'
                else:
                    cardinality = 'high'
            else:
                cardinality = 'unknown'

            details[col] = {
                'dtype': str(self.df[col].dtype),
                'missing': self.df[col].isnull().sum(),
                'missing_pct': (self.df[col].isnull().sum() / n_rows) * 100 if n_rows > 0 else 0,
                'unique': n_unique,
                'cardinality': cardinality
            }
        return details
    
    def detect_column_types(self) -> dict:
        """
        Infer semantic types beyond pandas dtypes.
        
        Returns:
            dict: {
                'column_name': {
                    'pandas_dtype': str,
                    'inferred_type': str,  # 'numeric', 'categorical', 'datetime', 'text', 'id'
                    'recommended_viz': str  # 'histogram', 'bar', 'line', 'none'
                }
            }
        """
        column_types = {}
        for col in self.df.columns:
            pd_dtype = str(self.df[col].dtype)
            inferred_type = 'unknown'
            recommended_viz = 'none'

            if pd.api.types.is_numeric_dtype(self.df[col]):
                inferred_type = 'numeric'
                # Check if it might be an ID
                if self.df[col].nunique() == self.df.shape[0]:
                     inferred_type = 'id'
                     recommended_viz = 'none'
                else:
                    recommended_viz = 'histogram'
            
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                inferred_type = 'datetime'
                recommended_viz = 'line' # Often good for time series
            
            elif pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                # Check if it looks like a date string
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    # If successfull, but was object, it might be datetime. 
                    # But be careful not to false positive on simple numbers or small strings. 
                    # For now keep simple heuristic.
                    # inferred_type = 'datetime' 
                    pass
                except:
                    pass
                
                if self.df[col].nunique() < 20:
                    inferred_type = 'categorical'
                    recommended_viz = 'bar'
                else:
                    inferred_type = 'text'
                    recommended_viz = 'none' # Wordcloud maybe later
            
            column_types[col] = {
                'pandas_dtype': pd_dtype,
                'inferred_type': inferred_type,
                'recommended_viz': recommended_viz
            }
        
        return column_types
