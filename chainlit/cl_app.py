import os, httpx, chainlit as cl

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


@cl.on_message
async def on_message(msg: cl.Message):
    q = msg.content.strip()
    if not q:
        return await cl.Message(content="Ask me something!").send()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{BACKEND_URL}/query", json={"question": q, "k": 25, "top_k": 5}
        )
        r.raise_for_status()
        data = r.json()
    await cl.Message(content=data.get("answer", "")).send()
    srcs = data.get("sources") or []
    if srcs:
        els = [
            cl.Text(
                name=f"[{i+1}] {s.get('title') or 'Source'}",
                content=(s.get("metadata", {}).get("snippet") or s.get("text", ""))[
                    :500
                ]
                + f"\n\nURL: {s.get('url') or '—'}",
                display="inline",
            )
            for i, s in enumerate(srcs)
        ]
        await cl.Message(content="Sources:", elements=els).send()
