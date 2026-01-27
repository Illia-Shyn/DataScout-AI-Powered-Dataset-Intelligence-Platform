import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.visualizer import AutoVisualizer
from components.data_profiler import DataProfiler

def test_visualizer():
    # Create sample data
    data = {
        'id': range(1, 101),
        'category': ['A', 'B', 'C', 'A'] * 25,
        'value': [x * 1.5 + np.random.normal(0, 5) for x in range(100)],
        'date': pd.date_range(start='2023-01-01', periods=100),
        'text': [f'Description {i}' for i in range(100)]
    }
    df = pd.DataFrame(data)
    
    # Need column types first
    profiler = DataProfiler(df)
    col_types = profiler.detect_column_types()
    
    print("Testing AutoVisualizer...")
    visualizer = AutoVisualizer(df, col_types)
    
    print("\n--- Generating All Charts ---")
    charts = visualizer.generate_all_charts()
    print(f"Generated {len(charts)} charts")
    
    for chart in charts:
        print(f"- {chart['title']} ({chart['chart_type']})")
        # assert isinstance(chart['figure'], go.Figure) # Simple check

if __name__ == "__main__":
    test_visualizer()
