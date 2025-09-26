import os
import httpx
import boto3
from urllib.parse import urlparse
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
OS_RETRY_ON_STATUS = [
    int(s)
    for s in os.getenv("OPENSEARCH_RETRY_ON_STATUS", "502,503,504").split(",")
    if s
]

if not OPENSEARCH_ENDPOINT_RAW:
    raise RuntimeError("Set OPENSEARCH_ENDPOINT env var")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY env var")


def _normalize_endpoint(value: str) -> str:
    """
    Accept:
      - vpc-xyz.us-east-1.es.amazonaws.com
      - https://vpc-xyz.us-east-1.es.amazonaws.com
      - https://vpc-xyz.us-east-1.es.amazonaws.com/
      - https://vpc-xyz.us-east-1.es.amazonaws.com/some/path
    Return just the host.
    """
    if "://" in value:
        parsed = urlparse(value)
        return parsed.hostname or value
    return value.split("/")[0]


def get_os_client() -> OpenSearch:
    host = _normalize_endpoint(OPENSEARCH_ENDPOINT_RAW)

    session = boto3.Session(region_name=AWS_REGION)
    creds = session.get_credentials()
    auth = AWSV4SignerAuth(creds, AWS_REGION)

    client = OpenSearch(
        hosts=[{"host": host, "port": 443, "scheme": "https"}],
        http_auth=auth,
        verify_certs=True,
        http_compress=True,  # saves bandwidth on large responses
        connection_class=RequestsHttpConnection,
        timeout=OS_TIMEOUT,
        max_retries=OS_MAX_RETRIES,
        retry_on_timeout=OS_RETRY_ON_TIMEOUT,
        retry_on_status=OS_RETRY_ON_STATUS,
    )

    # Fast fail if we can’t talk to the domain
    try:
        if not client.ping():
            raise RuntimeError(
                "OpenSearch ping failed (check endpoint, VPC access, or IAM)."
            )
    except Exception as e:
        raise RuntimeError(f"OpenSearch connection error: {e}")

    return client


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)


def get_http_client() -> httpx.AsyncClient:
    # Used for the reranker
    return httpx.AsyncClient(timeout=float(os.getenv("HTTPX_TIMEOUT", "20")))
