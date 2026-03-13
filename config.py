import os
from dotenv import load_dotenv

load_dotenv()

# ── Gemini ────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-flash-preview")

# Backward-compatible aliases
GEMINI_API_KEY = GOOGLE_API_KEY
GEMINI_MODEL = MODEL_NAME

# ── Pinecone ──────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST", "")
PINECONE_POLICY_NAMESPACE = os.getenv("PINECONE_POLICY_NAMESPACE", "support-policies")
PINECONE_CASE_NAMESPACE = os.getenv("PINECONE_CASE_NAMESPACE", "support-cases")

# ── Embedding ─────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Data paths ────────────────────────────────────────────
RAW_CSV_PATH = os.getenv("RAW_CSV_PATH", "data/raw/Customer_support_data.csv")
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "data/knowledge")
DB_PATH = os.getenv("DB_PATH", "data/support.db")

# ── Groq ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "meta-llama/llama-4-scout-17b-16e-instruct"
)

# Backward-compatible alias
GROQ_MODEL = GROQ_MODEL_NAME

# ── HuggingFace ───────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# ── App identity ──────────────────────────────────────────
APP_NAME = os.getenv("APP_NAME", "Support Intelligence Agent")
APP_ROLE = os.getenv("APP_ROLE", "AI customer support assistant")
APP_IDENTITY = f"You are {APP_NAME}, an {APP_ROLE} built for this application."

# ── Ingest limit ──────────────────────────────────────────
_limit = os.getenv("LIMIT_ROWS", "0").strip()
LIMIT_ROWS = int(_limit) if _limit.isdigit() and int(_limit) > 0 else None

# ── Proxy ─────────────────────────────────────────────────
_proxy_enabled = os.getenv("PROXY_ENABLED", "false").lower() == "true"
_proxy_scheme = os.getenv("PROXY_SCHEME", "http")
_proxy_host = os.getenv("PROXY_HOST", "")
_proxy_port = os.getenv("PROXY_PORT", "")
_proxy_username = os.getenv("PROXY_USERNAME", "")
_proxy_password = os.getenv("PROXY_PASSWORD", "")

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

SSL_CA_CERTS = os.getenv("SSL_CA_CERTS", "")

# Make proxy available to requests/urllib3/httpx
if _proxy_enabled and _proxy_url_base:
    os.environ["HTTP_PROXY"] = _proxy_url_base
    os.environ["HTTPS_PROXY"] = _proxy_url_base
    os.environ["http_proxy"] = _proxy_url_base
    os.environ["https_proxy"] = _proxy_url_base

# Local addresses should bypass proxy
_no_proxy = os.getenv("NO_PROXY", "localhost,127.0.0.1")
if _no_proxy:
    os.environ["NO_PROXY"] = _no_proxy
    os.environ["no_proxy"] = _no_proxy
