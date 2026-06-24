import os
import pandas as pd

def load_data(filepath: str = None) -> pd.DataFrame:
    """Loads support ticket CSV data and cleans the data types."""
    if filepath is None:
        # Resolve filepath relative to project structure: project/app/utils/loader.py -> project/data/support_tickets.csv
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filepath = os.path.join(base_dir, "data", "support_tickets.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Standardize types and handle missing data
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['response_time_hrs'] = pd.to_numeric(df['response_time_hrs'], errors='coerce')
    df['resolution_time_hrs'] = pd.to_numeric(df['resolution_time_hrs'], errors='coerce')
    df['customer_rating'] = pd.to_numeric(df['customer_rating'], errors='coerce')
    
    # Clean whitespace in string fields
    for col in ['ticket_id', 'category', 'priority', 'status', 'agent_id', 'issue_summary']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    return df
