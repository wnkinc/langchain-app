# langchain-app/chainlit/cl_app.py
import os
import uuid
import httpx
import boto3
import chainlit as cl
from typing import Optional

# Chainlit data layer
from chainlit.data.dynamodb import DynamoDBDataLayer
from chainlit.data.storage_clients.s3 import S3StorageClient

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

CHAINLIT_TABLE = os.environ["CHAINLIT_TABLE"]
CHAINLIT_BUCKET = os.environ["CHAINLIT_BUCKET"]


@cl.data_layer
def init_data_layer():
    dynamo = boto3.client("dynamodb", region_name=AWS_REGION)
    storage = S3StorageClient(
        bucket=CHAINLIT_BUCKET,
        region_name=AWS_REGION,
    )
    return DynamoDBDataLayer(
        table_name=CHAINLIT_TABLE,
        client=dynamo,
        storage_provider=storage,
        user_thread_limit=25,
    )


SESSION_ID = os.getenv("SESSION_ID", str(uuid.uuid4()))


@cl.on_message
async def on_message(message: cl.Message):
    q = (message.content or "").strip()
    if not q:
        await cl.Message(content="Ask me something!").send()
        return

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{BACKEND_URL}/query",
                json={"question": q, "k": 25, "top_k": 5},
                headers={"X-Session-Id": SESSION_ID},
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        await cl.Message(content=f"Backend error: {e}").send()
        return

    answer = (data.get("answer") or "").strip()
    sources = data.get("sources") or []

    await cl.Message(content=answer or "No answer.").send()

    if sources:
        items = []
        for i, s in enumerate(sources, 1):
            title = s.get("title") or f"Source {i}"
            url = s.get("url") or "—"
            snippet = (s.get("metadata", {}).get("snippet") or s.get("text") or "")[
                :500
            ]
            items.append(
                cl.Text(
                    name=f"[{i}] {title}",
                    content=f"{snippet}\n\nURL: {url}",
                    display="inline",
                )
            )
        await cl.Message(content="Sources:", elements=items).send()


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome to Bio-RAG 👋").send()


# Password auth: return cl.User on success, or None to reject.
@cl.password_auth_callback
def auth(username: str, password: str) -> Optional[cl.User]:
    expected = os.getenv("CHAINLIT_ADMIN_PASSWORD")
    if username == "admin" and expected and password == expected:
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    return None
