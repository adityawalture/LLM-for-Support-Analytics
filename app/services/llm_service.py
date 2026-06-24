import os
import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI
from app.models.schemas import StructuredQuery, FilterCondition

# Configure LLM Client - Groq Only
groq_api_key = os.environ.get("GROQ_API_KEY")
if groq_api_key:
    groq_api_key = groq_api_key.strip("'\"")

if groq_api_key:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        timeout=5.0  # Prevent infinite hangs if API is slow or unreachable
    )
    model_name = "llama-3.3-70b-versatile"  # High-quality model for reasoning
else:
    client = None
    model_name = None


SYSTEM_PROMPT_INTENT = """You are the Natural Language Query translator for a Support Ticket Analytics System.
Your job is to parse a user's natural language question and translate it into a structured JSON query object representing Pandas operations.

The support ticket database has the following schema:
- ticket_id: string (e.g. "TKT-001")
- created_at: datetime (YYYY-MM-DD HH:MM) [Min: 2024-01-01, Max: 2024-03-30]
- category: string ('General', 'Billing', 'Technical')
- priority: string ('Low', 'Medium', 'High', 'Critical')
- status: string ('Resolved', 'Open', 'Escalated')
- response_time_hrs: float (first response time in hours)
- resolution_time_hrs: float (null if ticket is unresolved/open/escalated)
- agent_id: string (e.g. "AGT-01")
- customer_rating: float (1 to 5, null if ticket is unresolved)
- issue_summary: string (free text)

JSON Output Schema:
{
  "metric": "count" | "mean" | "sum" | "min" | "max" | "list",
  "target_column": "customer_rating" | "resolution_time_hrs" | "response_time_hrs" | null,
  "filters": [
    {
      "column": "status" | "priority" | "category" | "agent_id" | "created_at" | "resolution_time_hrs" | "response_time_hrs" | "customer_rating",
      "operator": "==" | "!=" | ">" | "<" | ">=" | "<=" | "contains" | "is_null" | "not_null",
      "value": Any
    }
  ],
  "group_by": "agent_id" | "category" | "priority" | "status" | null,
  "sort": "ascending" | "descending" | null,
  "limit": integer | null
}

Guidelines:
1. "this month" refers to March 2024 (the latest month in the dataset). When filtering for this month, check created_at >= "2024-03-01 00:00:00" and created_at <= "2024-03-30 23:59:59".
2. "open tickets" or "currently open" refers to status == "Open" or status == "Escalated". Use status != "Resolved" or multiple status == "Open" filters, or status == "Open".
3. "not resolved within X hours" maps to a filter on "resolution_time_hrs" > X.
4. "lowest rating" or "most tickets" requires sorting and limiting (limit=1).
5. Output ONLY the raw JSON object. Do not include markdown codeblocks or explanation.

Example Questions & Outputs:
Q: "How many tickets are currently open?"
A: {
  "metric": "count",
  "filters": [{"column": "status", "operator": "==", "value": "Open"}]
}

Q: "Which agent has the lowest average customer rating?"
A: {
  "metric": "mean",
  "target_column": "customer_rating",
  "group_by": "agent_id",
  "sort": "ascending",
  "limit": 1
}

Q: "What is the average customer rating for Technical category tickets?"
A: {
  "metric": "mean",
  "target_column": "customer_rating",
  "filters": [{"column": "category", "operator": "==", "value": "Technical"}]
}

Q: "Show me all Critical tickets not resolved within 12 hours."
A: {
  "metric": "list",
  "filters": [
    {"column": "priority", "operator": "==", "value": "Critical"},
    {"column": "resolution_time_hrs", "operator": ">", "value": 12.0}
  ]
}

Q: "Which agent resolved the most tickets this month?"
A: {
  "metric": "count",
  "filters": [
    {"column": "status", "operator": "==", "value": "Resolved"},
    {"column": "created_at", "operator": ">=", "value": "2024-03-01 00:00:00"},
    {"column": "created_at", "operator": "<=", "value": "2024-03-31 23:59:59"}
  ],
  "group_by": "agent_id",
  "sort": "descending",
  "limit": 1
}
"""

def extract_query_intent_llm(question: str) -> Optional[StructuredQuery]:
    """Asks the LLM to translate a natural language query into a StructuredQuery."""
    if not client:
        print("LLM client not initialized (missing GROQ_API_KEY). Using fallback parser.")
        return None
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_INTENT},
                {"role": "user", "content": f"Translate this question: '{question}'"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        
        # Clean any accidental leading/trailing markdown code blocks if the LLM ignored the instructions
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)
        return StructuredQuery(**data)
    except Exception as e:
        print(f"LLM intent extraction failed: {e}")
        return None

def extract_query_intent_fallback(question: str) -> StructuredQuery:
    """Fallback regex parser for common queries to ensure reliability without LLM connectivity."""
    q = question.lower()
    
    # 1. "How many tickets are currently open?" / "How many open tickets exist?"
    if "how many" in q and "open" in q:
        return StructuredQuery(
            metric="count",
            filters=[FilterCondition(column="status", operator="==", value="Open")]
        )
        
    # 2. "Which agent has the lowest average customer rating?"
    if "agent" in q and "lowest" in q and "rating" in q:
        return StructuredQuery(
            metric="mean",
            target_column="customer_rating",
            group_by="agent_id",
            sort="ascending",
            limit=1
        )
        
    # 3. "Which agent resolved the most tickets this month?"
    if "agent" in q and ("most" in q or "highest" in q) and "resolved" in q:
        return StructuredQuery(
            metric="count",
            filters=[
                FilterCondition(column="status", operator="==", value="Resolved"),
                FilterCondition(column="created_at", operator=">=", value="2024-03-01 00:00:00"),
                FilterCondition(column="created_at", operator="<=", value="2024-03-31 23:59:59")
            ],
            group_by="agent_id",
            sort="descending",
            limit=1
        )
        
    # 4. "What is the average customer rating for Technical category tickets?"
    if "average" in q and "rating" in q and "technical" in q:
        return StructuredQuery(
            metric="mean",
            target_column="customer_rating",
            filters=[FilterCondition(column="category", operator="==", value="Technical")]
        )
        
    # 5. "Show me all Critical tickets not resolved within 12 hours."
    if "critical" in q and "not resolved" in q and "12" in q:
        return StructuredQuery(
            metric="list",
            filters=[
                FilterCondition(column="priority", operator="==", value="Critical"),
                FilterCondition(column="resolution_time_hrs", operator=">", value=12.0)
            ]
        )

    # Generic fallback: return everything as a list
    return StructuredQuery(metric="list", filters=[])

def extract_query_intent(question: str) -> tuple[StructuredQuery, bool]:
    """Tries LLM extraction, fallback to regex rules if it fails. Returns (StructuredQuery, used_llm)."""
    result = extract_query_intent_llm(question)
    if result is not None:
        return result, True
    return extract_query_intent_fallback(question), False


def format_answer_llm(question: str, calculation_result: Any) -> Optional[str]:
    """Asks the LLM to format a Pandas calculation result into a natural sounding answer."""
    if not client:
        return None
    try:
        prompt = f"""You are the Natural Language response generator for a Support Ticket Analytics System.
A user asked: "{question}"
The system calculated the following answer using Pandas:
{calculation_result}

Formulate a concise, clear, and direct response to the user's question using this calculation result. Keep the response factual and based ONLY on the calculation result provided.
"""
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful, concise assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM answer formatting failed: {e}")
        return None

def format_answer_fallback(question: str, calculation_result: Any) -> str:
    """Fallback response formatter when LLM fails."""
    # Simple formatting based on calculation result type
    if isinstance(calculation_result, (int, float)):
        return f"The calculated result for your query is {calculation_result}."
    elif isinstance(calculation_result, list):
        return f"Found {len(calculation_result)} matching records. Details: {calculation_result}"
    elif isinstance(calculation_result, dict):
        items = [f"{k}: {v}" for k, v in calculation_result.items()]
        return f"Calculation results: {', '.join(items)}."
    else:
        return f"Based on the dataset, the result is: {calculation_result}"

def format_answer(question: str, calculation_result: Any, skip_llm: bool = False) -> str:
    """Tries LLM formatting, fallback to rule-based string if it fails or if skip_llm is True."""
    if skip_llm:
        return format_answer_fallback(question, calculation_result)
    result = format_answer_llm(question, calculation_result)
    if result is not None:
        return result
    return format_answer_fallback(question, calculation_result)
