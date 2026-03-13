import os
from dotenv import load_dotenv

load_dotenv()


def _get_streamlit_secret(key: str, default=""):
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default


def get_secret(key: str, default=""):
    value = os.getenv(key)
    if value not in (None, ""):
        return value
    return _get_streamlit_secret(key, default)


def get_bool(key: str, default=False):
    value = str(get_secret(key, str(default))).strip().lower()
    return value in {"1", "true", "yes", "on"}


# ── Gemini ────────────────────────────────────────────────
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY", "")
MODEL_NAME = get_secret("MODEL_NAME", "gemini-3-flash-preview")

# Backward-compatible aliases
GEMINI_API_KEY = GOOGLE_API_KEY
GEMINI_MODEL = MODEL_NAME


# ── Pinecone ──────────────────────────────────────────────
PINECONE_API_KEY = get_secret("PINECONE_API_KEY", "")
PINECONE_INDEX_HOST = get_secret("PINECONE_INDEX_HOST", "")
PINECONE_POLICY_NAMESPACE = get_secret("PINECONE_POLICY_NAMESPACE", "support-policies")
PINECONE_CASE_NAMESPACE = get_secret("PINECONE_CASE_NAMESPACE", "support-cases")


# ── Embedding ─────────────────────────────────────────────
EMBED_MODEL = get_secret("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# ── Data paths ────────────────────────────────────────────
RAW_CSV_PATH = get_secret("RAW_CSV_PATH", "data/raw/Customer_support_data.csv")
KNOWLEDGE_DIR = get_secret("KNOWLEDGE_DIR", "data/knowledge")
DB_PATH = get_secret("DB_PATH", "data/support.db")


# ── Groq ──────────────────────────────────────────────────
GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
GROQ_MODEL_NAME = get_secret("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = get_secret(
    "GROQ_VISION_MODEL",
    "meta-llama/llama-4-scout-17b-16e-instruct"
)

# Backward-compatible alias
GROQ_MODEL = GROQ_MODEL_NAME


# ── HuggingFace ───────────────────────────────────────────
HF_TOKEN = get_secret("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


# ── App identity ──────────────────────────────────────────
APP_NAME = get_secret("APP_NAME", "Support Intelligence Agent")
APP_ROLE = get_secret("APP_ROLE", "AI customer support assistant")
APP_IDENTITY = f"You are {APP_NAME}, an {APP_ROLE} built for this application."


# ── Ingest limit ──────────────────────────────────────────
_limit = str(get_secret("LIMIT_ROWS", "0")).strip()
LIMIT_ROWS = int(_limit) if _limit.isdigit() and int(_limit) > 0 else None


# ── Proxy ─────────────────────────────────────────────────
_proxy_enabled = get_bool("PROXY_ENABLED", False)
_proxy_scheme = get_secret("PROXY_SCHEME", "http")
_proxy_host = get_secret("PROXY_HOST", "")
_proxy_port = get_secret("PROXY_PORT", "")
_proxy_username = get_secret("PROXY_USERNAME", "")
_proxy_password = get_secret("PROXY_PASSWORD", "")

_proxy_auth = f"{_proxy_username}:{_proxy_password}" if _proxy_username else ""

if _proxy_enabled and _proxy_host and _proxy_port:
    if _proxy_auth:
        _proxy_url_base = f"{_proxy_scheme}://{_proxy_auth}@{_proxy_host}:{_proxy_port}"
    else:
        _proxy_url_base = f"{_proxy_scheme}://{_proxy_host}:{_proxy_port}"
else:
    _proxy_url_base = ""

PROXY_CONFIG = {
    "enabled": _proxy_enabled,
    "proxy_url_base": _proxy_url_base,
    "proxy_auth": _proxy_auth,
}

SSL_CA_CERTS = get_secret("SSL_CA_CERTS", "")

if _proxy_enabled and _proxy_url_base:
    os.environ["HTTP_PROXY"] = _proxy_url_base
    os.environ["HTTPS_PROXY"] = _proxy_url_base
    os.environ["http_proxy"] = _proxy_url_base
    os.environ["https_proxy"] = _proxy_url_base

_no_proxy = get_secret("NO_PROXY", "localhost,127.0.0.1")
if _no_proxy:
    os.environ["NO_PROXY"] = _no_proxy
    os.environ["no_proxy"] = _no_proxy


# ── LangSmith observability ───────────────────────────────
LANGSMITH_TRACING = get_bool("LANGSMITH_TRACING", False)
LANGSMITH_API_KEY = get_secret("LANGSMITH_API_KEY", "")
LANGSMITH_ENDPOINT = get_secret("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = get_secret("LANGSMITH_PROJECT", APP_NAME)

# Optional legacy alias used by some integrations
LANGCHAIN_TRACING_V2 = str(LANGSMITH_TRACING).lower()
LANGCHAIN_API_KEY = LANGSMITH_API_KEY
LANGCHAIN_PROJECT = LANGSMITH_PROJECT

if LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

if LANGSMITH_ENDPOINT:
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT

if LANGSMITH_PROJECT:
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT


# ── LangGraph persistence ─────────────────────────────────
LANGGRAPH_ENABLE_PERSISTENCE = get_bool("LANGGRAPH_ENABLE_PERSISTENCE", False)
LANGGRAPH_CHECKPOINTER = get_secret("LANGGRAPH_CHECKPOINTER", "memory")  # memory | sqlite
LANGGRAPH_STORE = get_secret("LANGGRAPH_STORE", "memory")                # memory | sqlite
LANGGRAPH_SQLITE_PATH = get_secret("LANGGRAPH_SQLITE_PATH", "data/langgraph_checkpoints.sqlite")


# ── App/runtime defaults ──────────────────────────────────
DEFAULT_USER_ID = get_secret("DEFAULT_USER_ID", "")
DEFAULT_SESSION_ID = get_secret("DEFAULT_SESSION_ID", "")
DEFAULT_THREAD_ID = get_secret("DEFAULT_THREAD_ID", "")
ENABLE_RETRIEVAL_DEBUG = get_bool("ENABLE_RETRIEVAL_DEBUG", True)
ENABLE_ROUTE_DEBUG = get_bool("ENABLE_ROUTE_DEBUG", True)
