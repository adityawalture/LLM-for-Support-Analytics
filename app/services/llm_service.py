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

def extract_query_intent_llm(question: str) -> StructuredQuery:
    """Asks the LLM to translate a natural language query into a StructuredQuery."""
    if not client:
        raise RuntimeError("LLM client is not initialized because GROQ_API_KEY is missing from environment variables.")
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
        raise RuntimeError(f"LLM intent extraction failed: {str(e)}")

def extract_query_intent(question: str) -> StructuredQuery:
    """Translates a natural language query into a StructuredQuery using LLM."""
    return extract_query_intent_llm(question)


def format_answer_llm(question: str, calculation_result: Any) -> str:
    """Asks the LLM to format a Pandas calculation result into a natural sounding answer."""
    if not client:
        raise RuntimeError("LLM client is not initialized because GROQ_API_KEY is missing from environment variables.")
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
        raise RuntimeError(f"LLM answer formatting failed: {str(e)}")

def format_answer(question: str, calculation_result: Any) -> str:
    """Formats the calculation result into a natural language response using LLM."""
    return format_answer_llm(question, calculation_result)
