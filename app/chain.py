# app/chain.py
import os
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- System instructions ---
# Keep answers short, grounded, and with bracket-number citations that
# correspond exactly to the context snippets we pass in.
SYSTEM_INSTRUCTIONS = """You are a careful biomedical assistant.
Use only the provided context snippets to answer.
Cite using bracket numbers that match the snippet headers, e.g., [1], [2].
If you are not confident the answer is supported by the snippets, say you don't know."""

# Clarifier instructions: converse until the query is ready, then output READY: <query>
CLARIFIER_INSTRUCTIONS = """You are a clarifying assistant.
Hold a short conversation to refine the user's intent.
Ask targeted questions if needed. When the request is specific enough
to search, output a single line exactly in this format:
READY: <final concise query>
Otherwise, reply normally to continue clarifying."""

# --- Prompt templates ---
# Main RAG prompt includes compact chat history + the retrieval context
PROMPT = ChatPromptTemplate.from_template(
    "{system}\n\n"
    "Recent chat history (oldest → newest):\n{history}\n\n"
    "Question:\n{question}\n\n"
    "Context:\n{context}"
)

# Clarifier prompt (no context)
CLARIFIER_PROMPT = ChatPromptTemplate.from_template(
    "{system}\n\n"
    "Recent chat history (oldest → newest):\n{history}\n\n"
    "User:\n{question}"
)


def build_chain(llm: ChatGoogleGenerativeAI, **kwargs):
    """
    Returns a non-streaming chain (use .invoke).
    Keeps StrOutputParser so the result is plain text.
    """
    if kwargs:
        llm = llm.bind(**kwargs)
    return PROMPT.partial(system=SYSTEM_INSTRUCTIONS) | llm | StrOutputParser()


def build_streaming_chain(llm: ChatGoogleGenerativeAI, **kwargs):
    """
    Returns a streaming chain (use .stream).
    No StrOutputParser so chunks come through directly.
    """
    if kwargs:
        llm = llm.bind(**kwargs)
    return PROMPT.partial(system=SYSTEM_INSTRUCTIONS) | llm


def build_clarifier_chain(llm: ChatGoogleGenerativeAI, **kwargs):
    """
    Conversation-first chain. Produces either normal chat text OR
    a single 'READY: <query>' line to trigger RAG.
    (Use .invoke)
    """
    if kwargs:
        llm = llm.bind(**kwargs)
    return (
        CLARIFIER_PROMPT.partial(system=CLARIFIER_INSTRUCTIONS)
        | llm
        | StrOutputParser()
    )


def render_context(docs: List[Dict[str, Any]]) -> str:
    """
    Docs are the reranked items from retrieval:
      expected keys: 'text', 'title', optional 'url', 'metadata' (may include 'PMID')
    We cap per-snippet length to avoid blowing up prompt size.
    """
    max_chars = int(os.getenv("CONTEXT_CHARS_PER_DOC", "1200"))
    lines: List[str] = []

    for i, d in enumerate(docs, 1):
        title = (d.get("title") or "Untitled").strip()
        meta = d.get("metadata") or {}
        pmid: Optional[str] = meta.get("PMID") or meta.get("pmid") or d.get("PMID")
        url = (
            d.get("url") or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "")
        ).strip()
        text = (d.get("text") or "").strip()

        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"

        header = f"[{i}] {title}"
        if pmid:
            header += f" (PMID: {pmid})"
        if url:
            header += f" {url}"

        lines.append(f"{header}\n{text}\n")

    return "\n".join(lines)
