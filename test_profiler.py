import pandas as pd
from components.data_profiler import DataProfiler

def test_profiler():
    # Create sample data
    data = {
        'id': range(1, 101),
        'category': ['A', 'B', 'C', 'A'] * 25,
        'value': [x * 1.5 for x in range(100)],
        'date': pd.date_range(start='2023-01-01', periods=100),
        'text': [f'Description {i}' for i in range(100)]
    }
    df = pd.DataFrame(data)
    
    print("Testing DataProfiler...")
    profiler = DataProfiler(df)
    
    print("\n--- Generating Profile ---")
    profile = profiler.generate_profile()
    print("Overview:", profile['overview'])
    print("Columns keys:", list(profile['columns'].keys()))
    
    print("\n--- Detecting Column Types ---")
    col_types = profiler.detect_column_types()
    for col, info in col_types.items():
        print(f"{col}: {info}")

if __name__ == "__main__":
    test_profiler()
