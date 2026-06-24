from fastapi import APIRouter, Request, HTTPException
from typing import List
from app.models.schemas import AnomalyResponse
from app.services.anomaly_engine import detect_anomalies

router = APIRouter()

@router.get("/anomalies", response_model=List[AnomalyResponse])
def get_anomalies(request: Request):
    """Retrieve all flagged support ticket anomalies."""
    try:
        # Retrieve the shared dataframe from app state
        df = request.app.state.df
        results = detect_anomalies(df)
        return results
    except AttributeError:
        raise HTTPException(status_code=500, detail="Data loader is not initialized in app state.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")
