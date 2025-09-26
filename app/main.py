from fastapi import FastAPI, HTTPException
from chainlit import mount_chainlit
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

    # Match actual OS fields (pmid/title/text + optional s3 origin)
    return QueryResponse(
        answer=(answer or "").strip(),
        sources=[
            {
                "id": d.get("id"),
                "score": d.get("score"),
                "title": d.get("title"),
                "pmid": d.get("pmid"),
                "s3": d.get("s3"),
            }
            for d in reranked
        ],
    )


# Mount Chainlit UI+API at /chat
mount_chainlit(app=app, path="/chat", target="chainlit.cl_app")
