import os
from typing import List, Dict, Any
from fastapi import HTTPException
from opensearchpy import OpenSearch
import httpx

OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "documents")
RERANKER_URL = os.getenv("RERANKER_URL", "http://10.0.101.235:9000/rerank")


def os_search(os_client: OpenSearch, query: str, k: int) -> List[Dict[str, Any]]:
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
    out: List[Dict[str, Any]] = []
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


async def call_reranker(
    http: httpx.AsyncClient, query: str, passages: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    if not passages:
        return []
    payload = {
        "query": query,
        "passages": [{"id": p.get("id"), "text": p["text"]} for p in passages],
        "top_k": top_k,
    }
    r = await http.post(RERANKER_URL, json=payload)
    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"Reranker error {r.status_code}: {r.text}"
        )
    data = r.json()
    reranked = data.get("reranked") or data.get("results") or []

    by_id = {p.get("id"): p for p in passages if p.get("id") is not None}
    out = []
    for item in reranked[:top_k]:
        base = by_id.get(item.get("id"), {})
        out.append({**base, **item})
    return out or passages[:top_k]
