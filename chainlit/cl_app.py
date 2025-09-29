# langchain-app/chainlit/cl_app.py
import os
import uuid
import json
import anyio
import chainlit as cl
from typing import Optional, List, Dict, Any

# --- Reuse your app logic (no HTTP hop) ---
from app.clients import get_os_client, get_llm, get_http_client
from app.retrieval import os_search, call_reranker
from app.chain import build_streaming_chain, render_context

# --- Data layer: ALWAYS enabled (env must be set) ---
import boto3
from chainlit.data.dynamodb import DynamoDBDataLayer
from chainlit.data.storage_clients.s3 import S3StorageClient

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
CHAINLIT_TABLE = os.environ["CHAINLIT_TABLE"]  # required
CHAINLIT_BUCKET = os.environ["CHAINLIT_BUCKET"]  # required


@cl.data_layer
def init_data_layer():
    dynamo = boto3.client("dynamodb", region_name=AWS_REGION)
    storage = S3StorageClient(bucket=CHAINLIT_BUCKET, region_name=AWS_REGION)
    return DynamoDBDataLayer(
        table_name=CHAINLIT_TABLE,
        client=dynamo,
        storage_provider=storage,
        user_thread_limit=25,
    )


# --- Config ---
DEFAULT_K = int(os.getenv("SEARCH_K", "25"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
SESSION_ID = os.getenv("SESSION_ID", str(uuid.uuid4()))

# --- Singletons reused by the steps ---
os_client = get_os_client()
llm = get_llm()
chain = build_streaming_chain(llm)
http = None  # async HTTP client for reranker


@cl.on_chat_start
async def on_chat_start():
    """Warm up dependencies and greet."""
    global http
    if http is None:
        http = get_http_client()
    await cl.Message(
        content="Welcome to Bio-RAG 👋\nAsk me something biomedical!"
    ).send()


@cl.on_chat_end
async def on_chat_end():
    """Clean up per-chat resources if needed."""
    global http
    if http:
        try:
            await http.aclose()
        except Exception:
            pass
        http = None


@cl.password_auth_callback
def auth(username: str, password: str) -> Optional[cl.User]:
    """Optional simple password auth (set CHAINLIT_ADMIN_PASSWORD)."""
    expected = os.getenv("CHAINLIT_ADMIN_PASSWORD")
    if username == "admin" and expected and password == expected:
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    return None


def _cap(n: int, lo: int, hi: int) -> int:
    return max(lo, min(n, hi))


def _shorten(text: str, max_len: int = 1000) -> str:
    if not text:
        return ""
    text = text.strip()
    return (text[: max_len - 1] + "…") if len(text) > max_len else text


@cl.on_message
async def on_message(message: cl.Message):
    """Search -> Rerank -> Build Context -> LLM (stream) -> Sources."""
    q = (message.content or "").strip()
    if not q:
        await cl.Message(content="Ask me something!").send()
        return

    # Per-message knobs (simple defaults; wire UI later if you want)
    k = _cap(DEFAULT_K, 1, 200)
    top_k = _cap(DEFAULT_TOP_K, 1, k)

    # ---- STEP 1: SEARCH ----
    with cl.Step(name="Search") as search_step:
        search_step.input = {"query": q, "k": k}  # primitives only
        try:
            raw: List[Dict[str, Any]] = await anyio.to_thread.run_sync(
                os_search, os_client, q, k
            )
            search_step.metadata = {"k": k, "candidates_found": len(raw)}
        except Exception as e:
            search_step.output = {"error": str(e)}
            await cl.Message(content=f"Search error: {e}").send()
            return
        search_step.output = {"candidates_found": len(raw)}

    if not raw:
        await cl.Message(content="I couldn't find anything relevant.").send()
        return

    # ---- STEP 2: RERANK ----
    with cl.Step(name="Rerank") as rerank_step:
        rerank_step.input = {"top_k": top_k}
        try:
            if http is None:
                # Fallback safety: create a temp client if chat_start didn’t run
                from app.clients import get_http_client as _get_http_client

                temp_http = _get_http_client()
                reranked = await call_reranker(temp_http, q, raw, top_k)
                await temp_http.aclose()
            else:
                reranked = await call_reranker(http, q, raw, top_k)
            # Persist counts and top titles for quick glance
            rerank_step.metadata = {
                "top_k": top_k,
                "returned": len(reranked),
                "top_titles": [(_s.get("title") or "Untitled") for _s in reranked[:5]],
            }
        except Exception as e:
            rerank_step.output = {"error": str(e)}
            await cl.Message(content=f"Reranker error: {e}").send()
            return
        rerank_step.output = {"returned": len(reranked)}

    # ---- STEP 3: BUILD CONTEXT & STREAM LLM ----
    context = render_context(reranked)

    # Create a streaming message first
    msg = cl.Message(
        content="", author="Assistant", metadata={"session_id": SESSION_ID}
    )
    await msg.send()

    with cl.Step(name="LLM") as llm_step:
        llm_step.input = {"question": q}  # keep it simple
        streamed_any = False
        try:
            # Streaming chain yields text chunks directly
            for chunk in chain.stream({"question": q, "context": context}):
                # Handle different LC chunk types gracefully
                if isinstance(chunk, str):
                    token = chunk
                elif hasattr(chunk, "content"):
                    token = getattr(chunk, "content") or ""
                elif isinstance(chunk, dict):
                    token = chunk.get("text") or chunk.get("content") or ""
                else:
                    token = str(chunk)
                if token:
                    await msg.stream_token(token)
        except Exception:
            # Fallback to non-streaming
            try:
                full = chain.invoke({"question": q, "context": context})
                await msg.stream_token(full or "")
                streamed_any = True
            except Exception as e:
                msg.content = f"LLM error: {e}"
                await msg.update()
                llm_step.output = {"error": str(e)}
                return

        if not streamed_any:
            try:
                full = chain.invoke({"question": q, "context": context})
            except Exception as e:
                msg.content = f"LLM error: {e}"
                await msg.update()
                llm_step.output = {"error": str(e)}
                return
            msg.content = full or ""
            await msg.update()
        else:
            await msg.update()  # finalize the streaming message

        # Keep a small breadcrumb in the step
        llm_step.metadata = {"streamed": streamed_any}
        llm_step.output = {"answer_preview": (msg.content or "")[:160]}

    # ---- STEP 4: SOURCES (elements + structured metadata) ----
    if reranked:
        # Prepare structured payload so it’s queryable in the data layer
        structured_sources: List[Dict[str, Any]] = []
        items = []

        for i, s in enumerate(reranked, 1):
            title = (s.get("title") or f"Source {i}").strip()
            url = s.get("url")
            meta = s.get("metadata") or {}
            pmid = meta.get("PMID") or meta.get("pmid") or s.get("PMID")
            text = s.get("text") or meta.get("snippet") or ""
            snippet = _shorten(text, 500)

            # Save a user-visible element (stored by data layer)
            header_lines = [f"[{i}] {title}"]
            if pmid:
                header_lines.append(f"(PMID: {pmid})")
            if url:
                header_lines.append(f"\nURL: {url}")
            header = " ".join(header_lines)

            items.append(
                cl.Text(
                    name=header,
                    content=snippet,
                    display="inline",
                )
            )

            # Add a fully structured record for persistence/query
            structured_sources.append(
                {
                    "idx": i,
                    "id": s.get("id"),
                    "score": s.get("score"),
                    "title": title,
                    "url": url,
                    "pmid": pmid,
                    "s3": s.get("s3"),
                    "text": _shorten(s.get("text") or "", 2000),  # keep longer copy
                    "metadata": meta,  # full metadata preserved
                }
            )

        # Attach structured payload to the message metadata so it’s saved
        with cl.Step(name="Sources") as sources_step:
            sources_step.input = {"count": len(structured_sources)}
            await cl.Message(
                content="Sources:",
                elements=items,
                metadata={"sources": structured_sources},
            ).send()
            sources_step.output = {"count": len(structured_sources)}
