import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats

class AnomalyDetector:
    """
    Methods:
        1. Isolation Forest (multivariate outliers) - with filtering
        2. Z-score (univariate, normal distributions)
        3. IQR (univariate, skewed distributions)
        4. Business Rules (negative values)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def detect_all_anomalies(self, contamination: float = 0.05) -> dict:
        """
        Orchestrates detection and returns summary and details with REASONS.
        """
        # Store reasons mapping: row_index -> list of reason strings
        anomaly_reasons = {}
        
        # 1. Isolation Forest (Multivariate)
        if_indices = self.isolation_forest_detect(contamination)
        for idx in if_indices:
            if idx not in anomaly_reasons:
                anomaly_reasons[idx] = []
            anomaly_reasons[idx].append("Unusual combination of values (Isolation Forest)")

        # 2. Statistical Methods (Univariate)
        # Filter for meaningful numeric columns only
        numeric_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns 
                       if self._is_meaningful_numeric(c)]
        
        for col in numeric_cols:
            stat_res = self.statistical_detect(col)
            if stat_res:
                for idx, reason in zip(stat_res['indices'], stat_res['reasons']):
                    if idx not in anomaly_reasons:
                        anomaly_reasons[idx] = []
                    anomaly_reasons[idx].append(reason)
        
        # 3. Simple Business Rules (e.g. Negative Values)
        # Check for negative values in columns where it might be odd (Sales, Profit, etc)
        # Though Profit CAN be negative, usually large negatives are interesting.
        # Let's just check for specific keywords "sales", "quantity", "price", "cost" -> shouldn't be negative usually.
        # "Profit" negative is valid but maybe anomalous if very large.
        for col in numeric_cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ['sales', 'quantity', 'price', 'cost']):
                neg_indices = self.df[self.df[col] < 0].index.tolist()
                for idx in neg_indices:
                     if idx not in anomaly_reasons:
                        anomaly_reasons[idx] = []
                     anomaly_reasons[idx].append(f"Negative {col}: {self.df.loc[idx, col]}")

        # 4. Missing Value Detection
        missing_info = self._detect_missing_values()
        for idx, cols in missing_info.items():
            if idx not in anomaly_reasons:
                anomaly_reasons[idx] = []
            anomaly_reasons[idx].append(f"Missing values in: {', '.join(cols)}")

        # Compile Results
        all_anomaly_indices = list(anomaly_reasons.keys())
        
        # Create marked DataFrame
        flagged_df = self.df.copy()
        flagged_df['is_anomaly'] = flagged_df.index.isin(all_anomaly_indices)
        # Add Reasons Column
        flagged_df['anomaly_reason'] = flagged_df.index.map(
            lambda x: "; ".join(anomaly_reasons.get(x, [])) if x in anomaly_reasons else ""
        )
        
        # Subset to only anomalies
        anomalies_only = flagged_df[flagged_df['is_anomaly']]
        
        results = {
            'summary': {
                'total_anomalies': len(all_anomaly_indices),
                'anomaly_rate': len(all_anomaly_indices) / len(self.df) if len(self.df) > 0 else 0.0
            },
            'flagged_rows': anomalies_only,
            'reasons': anomaly_reasons # Raw dict for LLM context
        }
        
        return results
    
    def _is_meaningful_numeric(self, col_name: str) -> bool:
        """Filter out IDs, codes, and high cardinality columns."""
        col_lower = col_name.lower()
        
        # 1. Name keywords to exclude
        exclude_keywords = ['id', 'code', 'postal', 'zip', 'date', 'year', 'month', 'day']
        if any(k in col_lower for k in exclude_keywords):
            return False
            
        # 2. High Cardinality (heuristic: if unique values approx equals row count, it's likely an ID)
        n_unique = self.df[col_name].nunique()
        n_rows = len(self.df)
        if n_rows > 50 and n_unique > n_rows * 0.9: # 90% unique
            return False
            
        return True

    def _detect_missing_values(self) -> dict:
        """
        Adaptive missing value detection.
        
        Only flags missing values as anomalies when:
        1. Column is mostly complete (missing is rare)
        2. Uses dynamic threshold based on dataset characteristics
        
        This auto-adapts to any dataset.
        """
        missing_by_row = {}
        n_rows = len(self.df)
        
        if n_rows == 0:
            return missing_by_row
        
        # Calculate missing rate for each column
        col_missing_rates = {}
        for col in self.df.columns:
            missing_rate = self.df[col].isna().sum() / n_rows
            col_missing_rates[col] = missing_rate
        
        # Dynamic threshold: use the lesser of:
        # - 10% absolute (if <10% missing, it's probably meant to be complete)
        # - Half the median missing rate (adapts to messy datasets)
        missing_rates = list(col_missing_rates.values())
        median_missing = np.median(missing_rates) if missing_rates else 0
        
        threshold = min(0.10, max(0.01, median_missing * 0.5))
        
        # Only flag columns below threshold
        for col, rate in col_missing_rates.items():
            if rate > threshold or rate == 0:
                continue  # Skip: too many missing (not anomalous) or none missing
            
            missing_indices = self.df[self.df[col].isna()].index.tolist()
            for idx in missing_indices:
                if idx not in missing_by_row:
                    missing_by_row[idx] = []
                missing_by_row[idx].append(col)
        
        return missing_by_row

    def isolation_forest_detect(self, contamination: float = 0.05) -> list:
        """Returns list of anomaly indices using Isolation Forest."""
        # Filter for meaningful columns first
        numeric_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns 
                       if self._is_meaningful_numeric(c)]
        
        if not numeric_cols:
            return []

        numeric_df = self.df[numeric_cols].dropna()
        if numeric_df.empty:
             return []

        try:
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(numeric_df)
            # -1 is anomaly
            anom_indices = numeric_df.index[preds == -1].tolist()
            return anom_indices
        except Exception:
            return []
    
    def statistical_detect(self, column: str) -> dict:
        """Returns indices AND reasons for statistical outliers."""
        series = self.df[column].dropna()
        if len(series) < 10:
            return None
            
        skewness = series.skew()
        anom_indices = []
        reasons = []
        
        if abs(skewness) < 1.0:
            # Z-Score
            z_scores = stats.zscore(series)
            # Get indices where |z| > 3
            outlier_mask = np.abs(z_scores) > 3
            anom_indices = series.index[outlier_mask].tolist()
            # Generate specific reason
            for idx in anom_indices:
                val = series.loc[idx]
                z = z_scores.loc[idx] if hasattr(z_scores, 'loc') else z_scores[series.index.get_loc(idx)]
                reasons.append(f"{column}={val:.2f} (Z-Score: {z:.1f})")
        else:
            # Adaptive IQR / Percentile based on skewness
            method_desc = ""
            if abs(skewness) > 2.0:
                # Highly skewed: percentile-based (top/bottom 1%)
                lower_bound = series.quantile(0.01)
                upper_bound = series.quantile(0.99)
                method_desc = "outside 1st-99th percentile"
            else:
                # Moderate skew: 3× IQR (industry standard for outliers)
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                method_desc = "outside 3× IQR"
            
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            anom_indices = series.index[outlier_mask].tolist()
            
            for idx in anom_indices:
                val = series.loc[idx]
                reasons.append(f"{column}={val:.2f} ({method_desc})")
            
        if anom_indices:
            return {
                'indices': anom_indices,
                'reasons': reasons
            }
        return None
