import pandas as pd
import numpy as np
from components.anomaly_detector import AnomalyDetector

def test_anomaly_detector():
    # Create sample data with outliers
    np.random.seed(42)
    n_rows = 200
    data = {
        'value_normal': np.random.normal(100, 10, n_rows),
        'value_skewed': np.random.exponential(10, n_rows)
    }
    df = pd.DataFrame(data)
    
    # Inject anomalies
    df.loc[0, 'value_normal'] = 500  # Extreme high Z-score
    df.loc[1, 'value_skewed'] = 200  # Extreme tail value
    
    print("Testing AnomalyDetector...")
    detector = AnomalyDetector(df)
    
    print("\n--- Running Detection ---")
    results = detector.detect_all_anomalies(contamination=0.05)
    
    summary = results['summary']
    print(f"Total Anomalies Found: {summary['total_anomalies']}")
    print(f"Anomaly Rate: {summary['anomaly_rate']:.2%}")
    
    print("\n--- By Method ---")
    iso = results['by_method']['isolation_forest']
    print(f"Isolation Forest found: {iso['count']} anomalies")
    
    stats = results['by_method']['statistical']
    for col, detail in stats.items():
        print(f"Statistical ({detail['method']}) on {col}: {detail['count']} anomalies")
        
    print("\n--- Flagged Rows Sample ---")
    print(results['flagged_rows'].head())

if __name__ == "__main__":
    test_anomaly_detector()
