# app/clients.py
import os
import httpx
import boto3
from urllib.parse import urlparse
from typing import List
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_google_genai import ChatGoogleGenerativeAI

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT_RAW = os.getenv("OPENSEARCH_ENDPOINT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

# Tunables
OS_TIMEOUT = int(os.getenv("OPENSEARCH_TIMEOUT", "20"))
OS_MAX_RETRIES = int(os.getenv("OPENSEARCH_MAX_RETRIES", "3"))
OS_RETRY_ON_TIMEOUT = os.getenv("OPENSEARCH_RETRY_ON_TIMEOUT", "true").lower() == "true"


def _parse_statuses(val: str) -> List[int]:
    if not val:
        return [502, 503, 504]
    out = []
    for s in val.split(","):
        s = s.strip()
        if s.isdigit():
            out.append(int(s))
    return out or [502, 503, 504]


OS_RETRY_ON_STATUS = _parse_statuses(
    os.getenv("OPENSEARCH_RETRY_ON_STATUS", "502,503,504")
)


def _normalize_endpoint(value: str) -> str:
    """
    Accept:
      - vpc-xyz.us-east-1.es.amazonaws.com
      - https://vpc-xyz.us-east-1.es.amazonaws.com
      - https://vpc-xyz.us-east-1.es.amazonaws.com/
      - https://vpc-xyz.us-east-1.es.amazonaws.com/some/path
    Return just the host.
    """
    if not value:
        return ""
    if "://" in value:
        parsed = urlparse(value)
        return parsed.hostname or value
    return value.split("/")[0]


def _infer_opensearch_service(host: str) -> str:
    """
    AWS Managed OpenSearch/Elasticsearch uses service 'es'.
    OpenSearch Serverless uses service 'aoss'.
    """
    host = (host or "").lower()
    if ".aoss." in host or host.endswith("aoss.amazonaws.com"):
        return "aoss"
    return "es"


def get_os_client() -> OpenSearch:
    endpoint_raw = OPENSEARCH_ENDPOINT_RAW or ""
    host = _normalize_endpoint(endpoint_raw)
    if not host:
        raise RuntimeError("Set OPENSEARCH_ENDPOINT env var (host or https URL)")

    session = boto3.Session(region_name=AWS_REGION)
    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError(
            "No AWS credentials found (boto3). Configure env/role/instance profile."
        )

    service = _infer_opensearch_service(host)
    auth = AWSV4SignerAuth(creds, AWS_REGION, service=service)

    client = OpenSearch(
        hosts=[{"host": host, "port": 443, "scheme": "https"}],
        http_auth=auth,
        verify_certs=True,
        http_compress=True,  # saves bandwidth
        connection_class=RequestsHttpConnection,  # OK; urllib3 also works
        timeout=OS_TIMEOUT,
        max_retries=OS_MAX_RETRIES,
        retry_on_timeout=OS_RETRY_ON_TIMEOUT,
        retry_on_status=OS_RETRY_ON_STATUS,
    )

    # Fast fail if domain not reachable. If your domain blocks HEAD /, you can remove this.
    try:
        if not client.ping():
            raise RuntimeError(
                "OpenSearch ping failed (check endpoint, network/VPC access, or IAM perms)."
            )
    except Exception as e:
        raise RuntimeError(f"OpenSearch connection error: {e}")

    return client


def get_llm() -> ChatGoogleGenerativeAI:
    api_key = GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY env var")

    # Optional: read common generation params from env
    temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    max_output_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))

    # Note: ChatGoogleGenerativeAI supports .stream() in LangChain.
    # The 'streaming' kw is accepted in recent LangChain versions; safe to omit if yours is older.
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        streaming=True,  # enables efficient .stream() if supported by your version
    )


def get_http_client() -> httpx.AsyncClient:
    # Used for the reranker (HTTP/2 can improve perf if the service supports it)
    timeout = float(os.getenv("HTTPX_TIMEOUT", "120"))
    return httpx.AsyncClient(timeout=timeout, http2=True)
