import pandas as pd
from typing import List, Dict

def detect_anomalies(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Detects rule-based anomalies in support ticket dataset."""
    anomalies = []
    
    # 1. Rule 1: Critical tickets unresolved for more than 24 hours
    ref_time = df['created_at'].max()
    df_copy = df.copy()
    df_copy['age_hours'] = (ref_time - df_copy['created_at']).dt.total_seconds() / 3600.0
    
    critical_unresolved = df_copy[
        (df_copy['priority'] == 'Critical') & 
        (df_copy['status'] != 'Resolved') & 
        (df_copy['age_hours'] > 24.0)
    ]
    
    for _, row in critical_unresolved.iterrows():
        anomalies.append({
            "ticket_id": str(row['ticket_id']),
            "reason": f"Critical ticket unresolved for more than 24 hours (age: {row['age_hours']:.1f} hours)"
        })
        
    # 2. Rule 2: Abnormally long resolution time
    # Threshold: mean + (2 * standard deviation) of resolved tickets
    resolved = df[df['status'] == 'Resolved'].copy()
    if len(resolved) > 0:
        mean_res = resolved['resolution_time_hrs'].mean()
        std_res = resolved['resolution_time_hrs'].std()
        threshold = mean_res + (2.0 * std_res)
        
        long_res = resolved[resolved['resolution_time_hrs'] > threshold]
        for _, row in long_res.iterrows():
            anomalies.append({
                "ticket_id": str(row['ticket_id']),
                "reason": f"Abnormally long resolution time ({row['resolution_time_hrs']:.1f} hours vs threshold of {threshold:.1f} hours)"
            })
            
    # Sort anomalies by ticket_id for clean output
    anomalies.sort(key=lambda x: x['ticket_id'])
    return anomalies
