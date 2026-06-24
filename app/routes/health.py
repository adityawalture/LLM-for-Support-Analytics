from fastapi import APIRouter
from app.models.schemas import QuestionRequest # just a generic import or none

router = APIRouter()

@router.get("/health")
def health_check():
    """Endpoint to check the health status of the API."""
    return {"status": "running"}
