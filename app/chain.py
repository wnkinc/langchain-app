from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Given the user question and the context chunks below, answer concisely.
Cite sources with bracket numbers, e.g., [1], [2] where relevant. If unsure, say you don't know.

Question:
{question}

Context:
{context}
"""
)


def build_chain(llm: ChatGoogleGenerativeAI):
    return PROMPT | llm | StrOutputParser()


def render_context(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        title = (d.get("title") or "").strip()
        url = (d.get("url") or "").strip()
        lines.append(f"[{i}] {title} {url}\n{d['text']}\n")
    return "\n".join(lines)
