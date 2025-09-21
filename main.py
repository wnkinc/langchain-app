import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from opensearchpy import OpenSearch, RequestsHttpConnection
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

import httpx
import google.generativeai as genai

# -----------------------
# Config via env vars
# -----------------------
APP_PORT = int(os.getenv("APP_PORT", "8000"))
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

OPENSEARCH_ENDPOINT = os.getenv(
    "OPENSEARCH_ENDPOINT"
)  # vpc-...us-east-1.es.amazonaws.com
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "documents")
RERANKER_URL = os.getenv("RERANKER_URL", "http://10.0.101.235:9000/rerank")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if not OPENSEARCH_ENDPOINT:
    raise RuntimeError("Set OPENSEARCH_ENDPOINT env var")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY env var")

# -----------------------
# Clients
# -----------------------
# OpenSearch (SigV4 with instance role creds)
_session = boto3.Session()
_creds = _session.get_credentials()
_auth = AWSV4SignerAuth(_creds, AWS_REGION)

os_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
    http_auth=_auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=20,
    max_retries=3,
    retry_on_timeout=True,
)

# Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel(GEMINI_MODEL)

# We'll create/close the HTTP client in FastAPI lifespan
http: httpx.AsyncClient | None = None


# -----------------------
# API models
# -----------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(25, ge=1, le=100)  # docs to pull from OpenSearch
    top_k: int = Field(5, ge=1, le=50)  # docs to keep after rerank


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]  # [{id, score, text, metadata}, ...]


# -----------------------
# App
# -----------------------
app = FastAPI()


@app.on_event("startup")
async def _startup():
    global http
    http = httpx.AsyncClient(timeout=20.0)


@app.on_event("shutdown")
async def _shutdown():
    global http
    if http:
        await http.aclose()
        http = None


@app.get("/health")
def health():
    return {"ok": True}


def os_search(query: str, k: int) -> List[dict]:
    # Simple BM25 search; adjust field names to your mapping
    body = {
        "size": k,
        "query": {
            "multi_match": {"query": query, "fields": ["text^2", "title^3", "*"]}
        },
        "_source": ["text", "title", "url", "metadata"],
    }
    try:
        res = os_client.search(index=OPENSEARCH_INDEX, body=body)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenSearch error: {e}")

    hits = res.get("hits", {}).get("hits", [])
    out: List[dict] = []
    for h in hits:
        src = h.get("_source") or {}
        text = (src.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "id": h.get("_id"),
                "score": h.get("_score"),
                "text": text,
                "title": src.get("title"),
                "url": src.get("url"),
                "metadata": src.get("metadata") or {},
            }
        )
    return out


async def call_reranker(query: str, passages: List[dict], top_k: int) -> List[dict]:
    if not passages:
        return []
    payload = {
        "query": query,
        "passages": [{"id": p.get("id"), "text": p["text"]} for p in passages],
        "top_k": top_k,
    }
    assert http is not None
    r = await http.post(RERANKER_URL, json=payload)
    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"Reranker error {r.status_code}: {r.text}"
        )
    data = r.json()
    reranked = data.get("reranked") or data.get("results") or []

    # merge metadata back by id where possible
    by_id = {p.get("id"): p for p in passages if p.get("id") is not None}
    out = []
    for item in reranked[:top_k]:
        base = by_id.get(item.get("id"), {})
        merged = {**base, **item}
        out.append(merged)
    # fallback if reranker didn’t return anything
    if not out:
        out = passages[:top_k]
    return out


def build_gemini_prompt(question: str, docs: List[dict]) -> list[str]:
    lines = []
    for i, d in enumerate(docs, 1):
        title = (d.get("title") or "").strip()
        url = (d.get("url") or "").strip()
        lines.append(f"[{i}] {title} {url}\n{d['text']}\n")
    context = "\n".join(lines)
    prompt = f"""You are a helpful assistant. Given the user question and the context chunks below, answer concisely.
Cite sources with bracket numbers, e.g., [1], [2] where relevant. If unsure, say you don't know.

Question:
{question}

Context:
{context}
"""
    return [prompt]


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    # 1) Retrieve
    raw = os_search(req.question, req.k)
    if not raw:
        return QueryResponse(answer="I couldn't find anything relevant.", sources=[])

    # 2) Rerank
    reranked = await call_reranker(req.question, raw, req.top_k)

    # 3) Gemini
    prompt_parts = build_gemini_prompt(req.question, reranked)
    try:
        resp = gemini.generate_content(prompt_parts)
        answer = getattr(resp, "text", None) or ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    return QueryResponse(
        answer=answer.strip(),
        sources=[
            {
                "id": d.get("id"),
                "score": d.get("score"),
                "title": d.get("title"),
                "url": d.get("url"),
                "metadata": d.get("metadata"),
            }
            for d in reranked
        ],
    )
