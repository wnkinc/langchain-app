from pydantic import BaseModel, Field
from typing import List, Dict, Any


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(25, ge=1, le=100)  # docs to pull from OpenSearch
    top_k: int = Field(5, ge=1, le=50)  # docs to keep after rerank


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]  # [{id, score, text, metadata}, ...]
