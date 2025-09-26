import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Keep answers short, grounded, and with bracket-number citations that
# correspond exactly to the context snippets we pass in.
SYSTEM_INSTRUCTIONS = """You are a careful biomedical assistant.
Use only the provided context snippets to answer.
Cite using bracket numbers that match the snippet headers, e.g., [1], [2].
If you are not confident the answer is supported by the snippets, say you don't know."""

PROMPT = ChatPromptTemplate.from_template(
    "{system}\n\nQuestion:\n{question}\n\nContext:\n{context}"
)


def build_chain(llm: ChatGoogleGenerativeAI):
    # If you set temperature/max tokens in clients.get_llm(), that will apply here.
    return PROMPT.partial(system=SYSTEM_INSTRUCTIONS) | llm | StrOutputParser()


def render_context(docs: List[Dict[str, Any]]) -> str:
    """
    Docs are the reranked items from retrieval:
      expected keys: 'text', 'title', optional 'url', 'metadata' (may include 'PMID')
    We cap per-snippet length to avoid blowing up prompt size.
    """
    max_chars = int(
        os.getenv("CONTEXT_CHARS_PER_DOC", "1200")
    )  # tweak via env if needed
    lines: List[str] = []

    for i, d in enumerate(docs, 1):
        title = (d.get("title") or "Untitled").strip()
        meta = d.get("metadata") or {}
        pmid = meta.get("PMID") or meta.get("pmid") or d.get("PMID")
        # Prefer explicit url; otherwise synthesize a PubMed link if we have a PMID.
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
