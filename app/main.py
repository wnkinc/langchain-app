from fastapi import FastAPI, HTTPException
from .schemas import QueryRequest, QueryResponse
from .clients import get_os_client, get_llm, get_http_client
from .retrieval import os_search, call_reranker
from .chain import build_chain, render_context

app = FastAPI()

# singletons
os_client = get_os_client()
llm = get_llm()
chain = build_chain(llm)
http = None


@app.on_event("startup")
async def _startup():
    global http
    http = get_http_client()


@app.on_event("shutdown")
async def _shutdown():
    global http
    if http:
        await http.aclose()
        http = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    # optional sanity caps
    k = max(1, min(req.k or 50, 200))
    top_k = max(1, min(req.top_k or 10, k))

    raw = os_search(os_client, req.question, k)
    if not raw:
        return QueryResponse(answer="I couldn't find anything relevant.", sources=[])

    if http is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready")

    reranked = await call_reranker(http, req.question, raw, top_k)
    context = render_context(reranked)

    try:
        answer = chain.invoke({"question": req.question, "context": context})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini (LangChain) error: {e}")

    return QueryResponse(
        answer=(answer or "").strip(),
        sources=[
            {
                "id": d.get("id"),
                "score": d.get("score"),
                "title": d.get("title"),
                "text": (lambda t: (t[:500] + "…") if t and len(t) > 500 else t)(
                    d.get("text")
                ),
                "pmid": d.get("pmid"),
                "s3": d.get("s3"),
                "url": d.get("url")
                or (
                    f"https://pubmed.ncbi.nlm.nih.gov/{d.get('pmid')}/"
                    if d.get("pmid")
                    else None
                ),
            }
            for d in reranked
        ],
    )
