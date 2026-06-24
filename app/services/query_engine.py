import pandas as pd
from typing import Any, List
from app.models.schemas import StructuredQuery, FilterCondition

def apply_filters(df: pd.DataFrame, filters: List[FilterCondition]) -> pd.DataFrame:
    """Applies filter conditions to a pandas DataFrame."""
    filtered_df = df.copy()
    
    # Setup age calculation relative to the max timestamp in the dataset
    ref_time = df['created_at'].max()
    filtered_df['age_hours'] = (ref_time - filtered_df['created_at']).dt.total_seconds() / 3600.0
    
    for f in filters:
        col = f.column
        op = f.operator
        val = f.value
        
        # Verify column exists or is age_hours
        if col not in filtered_df.columns and col != 'age_hours':
            continue
            
        # Support business rule: "resolution_time_hrs > X" should include unresolved tickets older than X
        if col == 'resolution_time_hrs' and op == '>' and val is not None:
            try:
                limit_val = float(val)
                # Resolved tickets with duration > X
                cond_resolved = filtered_df['resolution_time_hrs'] > limit_val
                # Unresolved tickets with age > X
                cond_unresolved = (filtered_df['status'] != 'Resolved') & (filtered_df['age_hours'] > limit_val)
                filtered_df = filtered_df[cond_resolved | cond_unresolved]
                continue
            except ValueError:
                pass

        # Datetime column filter handling
        if col == 'created_at':
            try:
                datetime_val = pd.to_datetime(val)
                if op == '==':
                    filtered_df = filtered_df[filtered_df[col] == datetime_val]
                elif op == '!=':
                    filtered_df = filtered_df[filtered_df[col] != datetime_val]
                elif op == '>':
                    filtered_df = filtered_df[filtered_df[col] > datetime_val]
                elif op == '<':
                    filtered_df = filtered_df[filtered_df[col] < datetime_val]
                elif op == '>=':
                    filtered_df = filtered_df[filtered_df[col] >= datetime_val]
                elif op == '<=':
                    filtered_df = filtered_df[filtered_df[col] <= datetime_val]
                continue
            except Exception as e:
                print(f"Error parsing datetime filter for {col}: {e}")
                continue

        # Standard operator comparisons
        try:
            if op == '==':
                if filtered_df[col].dtype in ['float64', 'int64']:
                    filtered_df = filtered_df[filtered_df[col] == float(val)]
                else:
                    filtered_df = filtered_df[filtered_df[col].astype(str) == str(val)]
            elif op == '!=':
                if filtered_df[col].dtype in ['float64', 'int64']:
                    filtered_df = filtered_df[filtered_df[col] != float(val)]
                else:
                    filtered_df = filtered_df[filtered_df[col].astype(str) != str(val)]
            elif op == '>':
                filtered_df = filtered_df[filtered_df[col] > float(val)]
            elif op == '<':
                filtered_df = filtered_df[filtered_df[col] < float(val)]
            elif op == '>=':
                filtered_df = filtered_df[filtered_df[col] >= float(val)]
            elif op == '<=':
                filtered_df = filtered_df[filtered_df[col] <= float(val)]
            elif op == 'contains':
                filtered_df = filtered_df[filtered_df[col].astype(str).str.lower().str.contains(str(val).lower())]
            elif op == 'is_null':
                filtered_df = filtered_df[filtered_df[col].isna()]
            elif op == 'not_null':
                filtered_df = filtered_df[filtered_df[col].notna()]
        except Exception as e:
            print(f"Error applying filter for {col} {op} {val}: {e}")
            
    return filtered_df

def execute_query(df: pd.DataFrame, query: StructuredQuery) -> Any:
    """Executes a structured query against a DataFrame and returns the raw result."""
    # Apply filters first
    filtered_df = apply_filters(df, query.filters or [])
    
    # Scenario 1: Group By aggregation
    if query.group_by:
        gb_col = query.group_by
        if gb_col not in filtered_df.columns:
            return f"Error: Column '{gb_col}' does not exist for grouping"
            
        target = query.target_column or 'ticket_id'
        if target not in filtered_df.columns:
            return f"Error: Target column '{target}' does not exist"
            
        if query.metric == 'count':
            grouped = filtered_df.groupby(gb_col)[target].count()
        elif query.metric == 'mean':
            grouped = filtered_df.groupby(gb_col)[target].mean()
        elif query.metric == 'sum':
            grouped = filtered_df.groupby(gb_col)[target].sum()
        elif query.metric == 'min':
            grouped = filtered_df.groupby(gb_col)[target].min()
        elif query.metric == 'max':
            grouped = filtered_df.groupby(gb_col)[target].max()
        else:
            return f"Error: Unsupported grouping metric '{query.metric}'"
            
        # Apply sorting
        if query.sort:
            ascending = True if query.sort == 'ascending' else False
            grouped = grouped.sort_values(ascending=ascending)
            
        # Apply limit
        if query.limit:
            grouped = grouped.head(query.limit)
            
        # Format floating points to 2 decimal places for cleaner output
        if grouped.dtype == 'float64':
            grouped = grouped.round(2)
            
        return grouped.to_dict()
        
    # Scenario 2: Flat aggregation
    else:
        target = query.target_column
        
        # Verify target is available for aggregation metrics
        if query.metric in ['mean', 'sum', 'min', 'max'] and not target:
            return f"Error: Target column must be defined for metric '{query.metric}'"
            
        if query.metric == 'count':
            return int(filtered_df.shape[0])
            
        elif query.metric == 'mean':
            val = filtered_df[target].mean()
            return round(float(val), 2) if pd.notna(val) else 0.0
            
        elif query.metric == 'sum':
            val = filtered_df[target].sum()
            return round(float(val), 2) if pd.notna(val) else 0.0
            
        elif query.metric == 'min':
            val = filtered_df[target].min()
            return float(val) if pd.notna(val) else 0.0
            
        elif query.metric == 'max':
            val = filtered_df[target].max()
            return float(val) if pd.notna(val) else 0.0
            
        elif query.metric == 'list':
            limit = query.limit or 10
            # Select subset of readable columns to display
            display_cols = ['ticket_id', 'created_at', 'category', 'priority', 'status', 'resolution_time_hrs', 'agent_id', 'customer_rating', 'issue_summary']
            display_cols = [c for c in display_cols if c in filtered_df.columns]
            
            subset = filtered_df[display_cols].head(limit)
            if 'created_at' in subset.columns:
                subset['created_at'] = subset['created_at'].dt.strftime('%Y-%m-%d %H:%M')
                
            # Replace NaNs with None for JSON serialization
            subset = subset.replace({pd.NA: None})
            subset = subset.where(pd.notnull(subset), None)
            
            return subset.to_dict(orient='records')
            
        else:
            return f"Error: Unsupported metric '{query.metric}'"
