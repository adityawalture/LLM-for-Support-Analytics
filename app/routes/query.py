from fastapi import APIRouter, Request, HTTPException
from app.models.schemas import QuestionRequest, QueryResponse, SummaryResponse
from app.services.llm_service import extract_query_intent, format_answer
from app.services.query_engine import execute_query
import pandas as pd

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query_tickets(request: Request, body: QuestionRequest):
    """Processes a natural language query against the support ticket database."""
    try:
        df = request.app.state.df
        
        # 1. Translate question to structured query representation
        structured_query, used_llm = extract_query_intent(body.question)
        print(f"Structured Query Extracted: {structured_query} (LLM Used: {used_llm})")
        
        # 2. Execute structured query on pandas DataFrame
        calculation_result = execute_query(df, structured_query)
        print(f"Pandas Calculation Result: {calculation_result}")
        
        # 3. Format the result back to natural language response
        answer = format_answer(body.question, calculation_result, skip_llm=not used_llm)
        
        return QueryResponse(answer=answer)
    except AttributeError:
        raise HTTPException(status_code=500, detail="Data loader is not initialized in app state.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/summary", response_model=SummaryResponse)
def get_summary(request: Request):
    """Retrieve overall stats summary of the ticket dataset."""
    try:
        df = request.app.state.df
        
        total = int(df.shape[0])
        # Count open tickets as status == "Open"
        open_tkts = int(df[df['status'] == 'Open'].shape[0])
        resolved_tkts = int(df[df['status'] == 'Resolved'].shape[0])
        
        avg_rating = df[df['status'] == 'Resolved']['customer_rating'].mean()
        avg_rating_val = round(float(avg_rating), 2) if pd.notna(avg_rating) else None
        
        return SummaryResponse(
            total_tickets=total,
            open_tickets=open_tkts,
            resolved_tickets=resolved_tkts,
            average_customer_rating=avg_rating_val
        )
    except AttributeError:
        raise HTTPException(status_code=500, detail="Data loader is not initialized in app state.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary retrieval failed: {str(e)}")
