# retrieval.py
import os
import json
from typing import List, Dict, Any, Optional

import httpx
from fastapi import HTTPException

from opensearchpy import OpenSearch, RequestsHttpConnection  # type: ignore

try:
    # Available in opensearch-py >= 2.x for AWS-managed domains
    from opensearchpy.aws4auth import AWSV4SignerAuth  # type: ignore
    import boto3  # only needed if you want build_os_client()

    _HAS_AWS_SIGNER = True
except Exception:
    _HAS_AWS_SIGNER = False


# -----------------------
# Environment / Defaults
# -----------------------
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "pubmed-abstracts")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT") or os.getenv("OS_ENDPOINT")
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

RERANKER_URL = os.getenv("RERANKER_URL", "http://10.0.101.235:9000/rerank")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "50"))  # how many docs to fetch pre-rerank


# -----------------------
# Optional: build client
# -----------------------
def build_os_client() -> OpenSearch:
    """
    Builds an OpenSearch client for an AWS-managed domain using SigV4.
    Use only if you don't already build/pass a client elsewhere.
    """
    if not OPENSEARCH_ENDPOINT:
        raise RuntimeError(
            "OPENSEARCH_ENDPOINT env var is required to build the OpenSearch client."
        )
    if not _HAS_AWS_SIGNER:
        raise RuntimeError(
            "AWSV4SignerAuth not available. Ensure `opensearch-py` (>=2.x) is installed."
        )

    session = boto3.Session()
    credentials = session.get_credentials()
    auth = AWSV4SignerAuth(credentials, AWS_REGION)

    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )


# -----------------------
# Helpers
# -----------------------
def _parse_message_json(s: Optional[str]) -> Dict[str, Any]:
    """If message contains a JSON string, parse it safely."""
    if not s:
        return {}
    s = s.strip()
    if not s or s[0] not in ("{", "["):
        return {}
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _pick_text(src: Dict[str, Any]) -> str:
    """
    Prefer structured fields (abstract/title) and fall back to 'message'
    (which might itself be a JSON string).
    """
    text = (src.get("abstract") or "").strip()
    if text:
        return text

    msg = src.get("message")
    # Sometimes the pipeline produced both structured fields AND a JSON string in 'message'.
    # If abstract wasn't present, try to parse it out of message.
    parsed = _parse_message_json(msg)
    text = (parsed.get("abstract") or parsed.get("text") or "").strip()
    if text:
        return text

    # Last resort: return 'message' raw if it's plain text
    if isinstance(msg, str):
        return msg.strip()

    return ""


def _pick_title(src: Dict[str, Any]) -> Optional[str]:
    title = src.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    parsed = _parse_message_json(src.get("message"))
    title = parsed.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    return None


def _pick_pmid(src: Dict[str, Any]) -> Optional[str]:
    pmid = src.get("PMID")
    if pmid is not None:
        return str(pmid)

    parsed = _parse_message_json(src.get("message"))
    pmid = parsed.get("PMID") or parsed.get("pmid")
    if pmid is not None:
        return str(pmid)

    return None


# -----------------------
# Search + Rerank
# -----------------------
def os_search(
    os_client: OpenSearch, query: str, k: int = RETRIEVE_K
) -> List[Dict[str, Any]]:
    """
    Retrieve k candidates from OpenSearch, biased toward title/abstract,
    but able to match 'message' or other fields if present.
    """
    body = {
        "size": k,
        "track_total_hits": False,
        "query": {
            "multi_match": {
                "query": query,
                # Your mapping has: title(text), abstract(text), message(text)
                # We weight title/abstract higher than message.
                "fields": ["title^4", "abstract^3", "message^1", "*"],
                "type": "best_fields",
            }
        },
        "_source": ["PMID", "title", "abstract", "message", "s3.*"],
    }

    try:
        res = os_client.search(index=OPENSEARCH_INDEX, body=body)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenSearch error: {e}")

    hits = res.get("hits", {}).get("hits", [])
    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source") or {}
        text = _pick_text(src)
        if not text:
            # Skip empty payloads
            continue

        out.append(
            {
                "id": h.get("_id"),  # OpenSearch doc _id (PMID in your sample)
                "score": h.get("_score"),
                "pmid": _pick_pmid(src),
                "title": _pick_title(src),
                "text": text,
                # Handy to keep the S3 origin around for debugging/tracing
                "s3": src.get("s3") or {},
            }
        )
    return out


async def call_reranker(
    http: httpx.AsyncClient, query: str, passages: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """
    Call your MedCPT (or similar) reranker. Expects the API to return a list of
    {"id": <id>, "score": <float>} (or similar).
    """
    if not passages:
        return []

    payload = {
        "query": query,
        "candidates": [p["text"] for p in passages if p.get("text")],
        "top_k": top_k,
    }

    try:
        r = await http.post(RERANKER_URL, json=payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Reranker request failed: {e}")

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"Reranker error {r.status_code}: {r.text}"
        )

    data = r.json()
    reranked = data.get("reranked") or data.get("results") or []

    # Merge reranker scores back onto the original passages, preserving title/pmid/etc.
    by_id = {p.get("id"): p for p in passages if p.get("id") is not None}
    merged: List[Dict[str, Any]] = []
    for item in reranked[:top_k]:
        base = by_id.get(item.get("id"), {})
        if not base:
            continue
        merged.append({**base, **item})

    # Fallback: if reranker returns nothing, just pass through the top_k
    return merged or passages[:top_k]


# Convenience orchestration (optional)
async def retrieve_and_rerank(
    os_client: OpenSearch,
    http: httpx.AsyncClient,
    query: str,
    prefetch_k: int = RETRIEVE_K,
    final_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fetch `prefetch_k` docs from OpenSearch, rerank to `final_k`, return the top results.
    """
    candidates = os_search(os_client, query, prefetch_k)
    return await call_reranker(http, query, candidates, final_k)
