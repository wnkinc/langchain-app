# langchain-app/chainlit/cl_app.py
import os
import uuid
import json
import anyio
import chainlit as cl
from typing import Optional, List, Dict, Any

# --- Reuse your app logic directly (no HTTP hop) ---
from app.clients import get_os_client, get_llm, get_http_client
from app.retrieval import os_search, call_reranker
from app.chain import (
    build_streaming_chain,
    build_clarifier_chain,
    render_context,
)  # build_streaming_chain should support .stream()

# --- Data layer: enabled when env is set ---
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
DEFAULT_K = int(os.getenv("SEARCH_K", "50"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "10"))
SESSION_ID = os.getenv("SESSION_ID", str(uuid.uuid4()))
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "6"))  # last N turns to send
ALWAYS_RAG = os.getenv("ALWAYS_RAG", "false").lower() == "true"  # bypass clarifier
READY_PREFIX = os.getenv("READY_PREFIX", "READY:")

# --- Singletons reused by steps ---
os_client = get_os_client()
llm = get_llm()
chain = build_streaming_chain(llm)
clarifier = build_clarifier_chain(llm)
http = None  # async HTTP client for reranker


def _cap(n: int, lo: int, hi: int) -> int:
    return max(lo, min(n, hi))


def _shorten(text: str, max_len: int = 1000) -> str:
    if not text:
        return ""
    t = text.strip()
    return (t[: max_len - 1] + "…") if len(t) > max_len else t


def _ensure_text(value: Any) -> str:
    """Coerce a LangChain output or chunk into plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):
        content = getattr(value, "content")
        if isinstance(content, str):
            return content
        return _ensure_text(content)
    if isinstance(value, dict):
        for key in ("text", "content", "message", "value"):
            if key in value:
                return _ensure_text(value[key])
    return str(value)


def _to_source_shape(d: Dict[str, Any]) -> Dict[str, Any]:
    """Shape used for both Search step payload and final Sources list."""
    pmid = d.get("pmid") or d.get("PMID")
    url = d.get("url") or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None)
    text = d.get("text")
    # If len > 500, show first 250 + ellipsis; else full.
    text_shaped = (text[:250] + "…") if (text and len(text) > 500) else text
    return {
        "id": d.get("id"),
        "score": d.get("score"),
        "title": d.get("title"),
        "text": text_shaped,
        "pmid": pmid,
        "s3": d.get("s3"),
        "url": url,
        "metadata": d.get("metadata", {}),
    }


def _render_sources_elements(sources: List[Dict[str, Any]]) -> List[cl.Text]:
    """Render compact source chips as Chainlit elements."""
    items: List[cl.Text] = []
    for i, s in enumerate(sources, 1):
        title = s.get("title") or f"Source {i}"
        url = s.get("url") or "—"
        # Prefer metadata.snippet, then text; cap to 500 here
        snippet = (s.get("metadata", {}).get("snippet") or s.get("text") or "")[:500]
        items.append(
            cl.Text(
                name=f"[{i}] {title}",
                content=f"{snippet}\n\nURL: {url}",
                display="inline",
            )
        )
    return items


def _format_history(history: List[Dict[str, str]], turns: int) -> str:
    """Return a compact text view of the last N user+assistant turns."""
    trimmed = history[-(turns * 2) :] if turns > 0 else []
    lines = []
    for m in trimmed:
        role = m.get("role", "user")
        content = _shorten(str(m.get("content", "")), 1500)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start():
    """Warm up dependencies and greet."""
    global http
    if http is None:
        http = get_http_client()
    # initialize per-thread chat history
    cl.user_session.set(
        "history", []
    )  # list[{"role": "user"|"assistant", "content": str}]
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


@cl.on_message
async def on_message(message: cl.Message):
    """
    Clarify-then-RAG Pipeline:

      Phase 0 (Clarify): LLM chats until it emits "READY: <query>".
                        Users can also force with "/rag <query>".
      Step 1 (Search): show ALL retriever candidates (input to reranker)
      Step 2 (Rerank): show reranked set
      Then: stream final answer in a single bubble, and render Sources below
    """
    q = (message.content or "").strip()
    if not q:
        await cl.Message(content="Ask me something!").send()
        return

    # knobs
    k = _cap(DEFAULT_K, 1, 200)
    top_k = _cap(DEFAULT_TOP_K, 1, k)

    # --- Phase 0: Clarify (decide whether to run retrieval) ---
    history: List[Dict[str, str]] = cl.user_session.get("history", [])
    history_text = _format_history(history, CHAT_HISTORY_TURNS)

    final_query = q
    do_search = ALWAYS_RAG

    # User can force RAG with a slash command
    if q.startswith("/rag "):
        final_query = q[len("/rag ") :].strip() or q
        do_search = True
    elif not ALWAYS_RAG:
        try:
            clarifier_out = clarifier.invoke(
                {"question": q, "history": history_text}
            ).strip()
        except Exception:
            clarifier_out = ""
        if clarifier_out.startswith(READY_PREFIX):
            final_query = clarifier_out[len(READY_PREFIX) :].strip() or q
            do_search = True
        elif clarifier_out:
            # Continue the conversation without RAG this turn
            assistant_reply = clarifier_out
            await cl.Message(
                content=assistant_reply,
                author="Assistant",
                metadata={"session_id": SESSION_ID},
            ).send()
            # Persist turn
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": assistant_reply})
            cl.user_session.set("history", history)
            return

    # If we decided not to search (shouldn’t happen unless clarifier failed silently), fall back
    if not do_search and not ALWAYS_RAG:
        await cl.Message(content="Could you clarify a bit more?").send()
        return

    # ---- STEP 1: SEARCH (show ALL docs passed to reranker) ----
    with cl.Step(name="Search") as search_step:
        search_step.input = {"query": final_query, "k": k}
        try:
            raw: List[Dict[str, Any]] = await anyio.to_thread.run_sync(
                os_search, os_client, final_query, k
            )
            # Emit *all* candidates in the step output (primitives only)
            candidates = [_to_source_shape(doc) for doc in raw]
            search_step.metadata = {"candidates_found": len(candidates)}
            search_step.output = {"candidates": candidates}
        except Exception as e:
            search_step.output = {"error": str(e)}
            await cl.Message(content=f"Search error: {e}").send()
            return

    if not raw:
        await cl.Message(content="I couldn't find anything relevant.").send()
        # Persist this user message even if no results
        history.append({"role": "user", "content": q})
        cl.user_session.set("history", history)
        return

    # ---- STEP 2: RERANK ----
    with cl.Step(name="Rerank") as rerank_step:
        rerank_step.input = {"top_k": top_k}
        try:
            if http is None:
                # Safety fallback
                temp_http = get_http_client()
                reranked = await call_reranker(temp_http, final_query, raw, top_k)
                await temp_http.aclose()
            else:
                reranked = await call_reranker(http, final_query, raw, top_k)
            reranked_view = [_to_source_shape(doc) for doc in reranked]
            rerank_step.metadata = {
                "returned": len(reranked_view),
                "top_titles": [s.get("title") or "Untitled" for s in reranked_view[:5]],
            }
            rerank_step.output = {"results": reranked_view}
        except Exception as e:
            rerank_step.output = {"error": str(e)}
            await cl.Message(content=f"Reranker error: {e}").send()
            return

    # ---- BUILD CONTEXT & STREAM ANSWER (NOT a step) ----
    context = render_context(reranked)

    msg = cl.Message(
        content="", author="Assistant", metadata={"session_id": SESSION_ID}
    )
    await msg.send()

    streamed_any = False
    chunks: List[str] = []
    try:
        for chunk in chain.stream(
            {"question": final_query, "context": context, "history": history_text}
        ):
            token = _ensure_text(chunk)
            if token:
                await msg.stream_token(token)
                streamed_any = True
                chunks.append(token)
    except Exception:
        # Fallback to non-streaming
        try:
            full = chain.invoke(
                {"question": final_query, "context": context, "history": history_text}
            )
            full_text = _ensure_text(full)
            await msg.stream_token(full_text)
            streamed_any = True
            chunks = [full_text]
        except Exception as e:
            msg.content = f"LLM error: {e}"
            await msg.update()
            return

    if not streamed_any:
        msg.content = "(No answer)"
        await msg.update()
    else:
        msg.content = "".join(chunks)
        await msg.update()

    # Persist this turn to history (user + assistant)
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("history", history)

    # ---- SOURCES (separate block, not a step) ----
    if reranked:
        final_sources = [_to_source_shape(d) for d in reranked]
        elements = _render_sources_elements(final_sources)
        if elements:
            await cl.Message(content="Sources:", elements=elements).send()
