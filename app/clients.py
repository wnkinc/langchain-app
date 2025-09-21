import os, boto3, httpx
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_google_genai import ChatGoogleGenerativeAI

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if not OPENSEARCH_ENDPOINT:
    raise RuntimeError("Set OPENSEARCH_ENDPOINT env var")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY env var")


def get_os_client() -> OpenSearch:
    session = boto3.Session(region_name=AWS_REGION)
    creds = session.get_credentials()
    auth = AWSV4SignerAuth(creds, AWS_REGION)
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=20,
        max_retries=3,
        retry_on_timeout=True,
    )


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)


def get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=20.0)
