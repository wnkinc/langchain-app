from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # docs to pull from OpenSearch
    k: int = Field(25, ge=1, le=200)
    # docs to keep after rerank
    top_k: int = Field(5, ge=1, le=50)


class SourceItem(BaseModel):
    id: Optional[str] = None
    score: Optional[float] = None
    title: Optional[str] = None
    text: Optional[str] = None  # we'll often omit this in API response
    pmid: Optional[int] = None
    s3: Optional[Dict[str, str]] = None  # {"bucket": "...", "key": "..."}


class QueryResponse(BaseModel):
    answer: str
    # prefer a typed list, but if you want max flexibility, keep Dict[str, Any]
    sources: List[SourceItem]
