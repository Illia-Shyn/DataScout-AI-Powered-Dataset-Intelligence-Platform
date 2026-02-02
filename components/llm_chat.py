import pandas as pd
import numpy as np
import ollama
from utils.prompts import SYSTEM_PROMPT, CONTEXT_TEMPLATE

class DataChat:
    """Natural language Q&A over the dataset using Ollama."""
    
    def __init__(self, df: pd.DataFrame, profile: dict, anomalies: dict, model_name: str = "mistral"):
        self.df = df
        self.profile = profile
        self.anomalies = anomalies
        self.model_name = model_name
    
    def build_context_prompt(self) -> str:
        """Build context with schema, stats, sample data."""
        # Simplify profile for prompt to save tokens
        overview = self.profile.get('overview', {})
        columns_summary = {}
        for col, details in self.profile.get('columns', {}).items():
            columns_summary[col] = {
                'dtype': details.get('dtype'),
                'unique': details.get('unique'),
                'missing': details.get('missing')
            }
        
        # Simplify anomalies for context
        anomaly_summary = self.anomalies.get('summary', {})
        reasons_dict = self.anomalies.get('reasons', {})
        
        # Get top 5 most interesting anomalies (by number of reasons or just first few)
        top_anomalies_text = ""
        count = 0
        for idx, reasons in reasons_dict.items():
            if count >= 5: break
            row_data = self.df.loc[idx].to_dict()
            # Format: 'Row 14: Profit=-2000 (Z-score), Sales=5 (Low). Full Row: {...}'
            # Keep it concise
            top_anomalies_text += f"\n- Row {idx}: {'; '.join(reasons)}"
            count += 1
            
        if not top_anomalies_text:
            top_anomalies_text = "None detected."

        context = CONTEXT_TEMPLATE.format(
            overview=str(overview),
            columns=str(columns_summary),
            anomalies=f"Summary: {anomaly_summary}\n\nTop Anomalies:{top_anomalies_text}",
            sample_data=self.df.head(3).to_markdown(index=False)
        )
        
        # --- ENRICHMENT: Add detailed analysis ---
        
        # 1. Column Statistics
        context += "\n\n--- DETAILED STATISTICS ---\n"
        context += "COLUMN STATISTICS:\n"
        context += self.df.describe().round(2).to_string() + "\n"
        
        # 2. Missing Values by Column
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            context += "\nMISSING VALUES BY COLUMN:\n"
            for col, count in missing.items():
                context += f"- {col}: {count} missing ({count/len(self.df)*100:.1f}%)\n"
        else:
            context += "\nMISSING VALUES: None\n"
            
        # 3. Top Correlations
        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            pairs = []
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i+1:]:
                    pairs.append((c1, c2, corr.loc[c1, c2]))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            context += "\nTOP CORRELATIONS:\n"
            for c1, c2, val in pairs[:5]:
                context += f"- {c1} vs {c2}: {val:.2f}\n"

        # 4. Category Aggregations
        cat_cols = self.df.select_dtypes(include=['object']).columns[:3]  # First 3 categoricals
        num_cols = self.df.select_dtypes(include=[np.number]).columns[:2]  # First 2 numerics

        if len(cat_cols) > 0 and len(num_cols) > 0:
            context += "\nKEY AGGREGATIONS:\n"
            for cat in cat_cols:
                if self.df[cat].nunique() <= 10:  # Only low-cardinality
                    for num in num_cols:
                        agg = self.df.groupby(cat)[num].mean().round(2)
                        context += f"- Average {num} by {cat}: {agg.to_dict()}\n"
        
        return context
    
    def ask(self, question: str) -> dict:
        """
        Returns:
            dict: {
                'answer': str,
                'confidence': str
            }
        """
        context = self.build_context_prompt()
        full_prompt = f"{SYSTEM_PROMPT}\n{context}\n\nUser Question: {question}"
        
        try:
            # Call Ollama
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': full_prompt},
            ])
            
            answer = response['message']['content']
            
            return {
                'answer': answer,
                'confidence': 'High' # Mock confidence for now
            }
        except Exception as e:
            return {
                'answer': f"Error communicating with Ollama: {str(e)}. Please ensure Ollama is running.",
                'confidence': 'Low'
            }

    def debug_context(self) -> dict:
        """Returns context info for debugging."""
        context = self.build_context_prompt()
        
        # Approximate token count (rough: 1 token ≈ 4 chars)
        approx_tokens = len(context) // 4
        
        return {
            'char_count': len(context),
            'approx_tokens': approx_tokens,
            'line_count': context.count('\n'),
            'context_preview': context[:500] + "\n...\n" + context[-500:],
            'full_context': context  # For detailed inspection
        }
