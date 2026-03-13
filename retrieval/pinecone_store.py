import requests
from urllib3.util import make_headers
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from config import PINECONE_API_KEY, PINECONE_INDEX_HOST, EMBED_MODEL

try:
    from config import PROXY_CONFIG
except ImportError:
    PROXY_CONFIG = {"enabled": False, "proxy_url_base": "", "proxy_auth": ""}

try:
    from config import SSL_CA_CERTS
except ImportError:
    SSL_CA_CERTS = ""


if not PINECONE_API_KEY:
    raise ValueError("Set PINECONE_API_KEY in .env")

if not PINECONE_INDEX_HOST:
    raise ValueError("Set PINECONE_INDEX_HOST in .env")


pc_kwargs = {"api_key": PINECONE_API_KEY}

if PROXY_CONFIG.get("enabled"):
    pc_kwargs["proxy_url"] = PROXY_CONFIG["proxy_url_base"]

    if PROXY_CONFIG.get("proxy_auth"):
        pc_kwargs["proxy_headers"] = make_headers(
            proxy_basic_auth=PROXY_CONFIG["proxy_auth"]
        )

    if SSL_CA_CERTS:
        pc_kwargs["ssl_ca_certs"] = SSL_CA_CERTS


_pc = Pinecone(**pc_kwargs)
_index = _pc.Index(host=PINECONE_INDEX_HOST)
_embedder = SentenceTransformer(EMBED_MODEL)


def get_index():
    return _index


def get_embedder():
    return _embedder


def embed_texts(texts):
    if isinstance(texts, str):
        texts = [texts]

    texts = [str(t).strip() for t in texts if str(t).strip()]
    if not texts:
        return []

    vectors = _embedder.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_query(text):
    vectors = embed_texts([text])
    return vectors[0] if vectors else []


def _coerce_metadata(obj):
    if isinstance(obj, dict):
        return obj
    return {}


def _extract_text_from_metadata(metadata):
    metadata = _coerce_metadata(metadata)
    return (
        metadata.get("chunk_text")
        or metadata.get("text")
        or metadata.get("content")
        or metadata.get("body")
        or ""
    )


def _normalize_filter(metadata_filter):
    if not metadata_filter or not isinstance(metadata_filter, dict):
        return None
    return metadata_filter


def _merge_filters(*filters):
    valid_filters = [f for f in filters if isinstance(f, dict) and f]
    if not valid_filters:
        return None
    if len(valid_filters) == 1:
        return valid_filters[0]
    return {"$and": valid_filters}


def parse_matches(result):
    matches = (
        result["matches"]
        if isinstance(result, dict)
        else getattr(result, "matches", [])
    )
    parsed = []

    for m in matches:
        score = m["score"] if isinstance(m, dict) else getattr(m, "score", 0.0)
        metadata = m["metadata"] if isinstance(m, dict) else getattr(m, "metadata", {})
        match_id = m["id"] if isinstance(m, dict) else getattr(m, "id", "")
        values = m.get("values") if isinstance(m, dict) else getattr(m, "values", None)

        metadata = _coerce_metadata(metadata)
        text_value = _extract_text_from_metadata(metadata)

        parsed.append({
            "id": match_id,
            "score": float(score),
            "metadata": metadata,
            "text": text_value,
            "chunk_text": text_value,
            "values": values,
        })

    return parsed


def parse_fetch_records(payload):
    if not payload:
        return []

    records = (
        payload.get("vectors")
        or payload.get("records")
        or payload.get("matches")
        or []
    )

    if isinstance(records, dict):
        records = list(records.values())

    parsed = []
    for r in records:
        if isinstance(r, dict):
            metadata = _coerce_metadata(r.get("metadata", {}) or {})
            text_value = _extract_text_from_metadata(metadata)
            parsed.append({
                "id": r.get("id") or r.get("_id", ""),
                "score": float(r.get("score", 1.0) or 1.0),
                "metadata": metadata,
                "text": text_value,
                "chunk_text": text_value,
            })

    return parsed


def _requests_kwargs():
    kwargs = {"timeout": 30}

    if SSL_CA_CERTS:
        kwargs["verify"] = SSL_CA_CERTS
    else:
        kwargs["verify"] = True

    if PROXY_CONFIG.get("enabled") and PROXY_CONFIG.get("proxy_url_base"):
        proxy_url = PROXY_CONFIG["proxy_url_base"]
        kwargs["proxies"] = {
            "http": proxy_url,
            "https": proxy_url,
        }

    return kwargs


def query_namespace(
    namespace,
    query_text,
    top_k=5,
    metadata_filter=None,
    include_metadata=True,
    include_values=False,
):
    vector = embed_query(query_text)
    if not vector:
        return []

    kwargs = {
        "namespace": namespace,
        "vector": vector,
        "top_k": top_k,
        "include_metadata": include_metadata,
    }

    if include_values:
        kwargs["include_values"] = True

    metadata_filter = _normalize_filter(metadata_filter)
    if metadata_filter:
        kwargs["filter"] = metadata_filter

    result = _index.query(**kwargs)
    return parse_matches(result)


def exact_metadata_lookup(
    namespace,
    field_name,
    field_value,
    query_text=None,
    top_k=5,
    include_metadata=True,
):
    if not field_name or field_value is None:
        return []

    metadata_filter = {field_name: {"$eq": field_value}}
    lookup_text = query_text or f"{field_name} {field_value}"

    return query_namespace(
        namespace=namespace,
        query_text=lookup_text,
        top_k=top_k,
        metadata_filter=metadata_filter,
        include_metadata=include_metadata,
    )


def fetch_by_metadata(namespace, metadata_filter, limit=10):
    url = f"https://{PINECONE_INDEX_HOST}/vectors/fetch_by_metadata"
    headers = {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json",
        "X-Pinecone-API-Version": "2025-10",
    }
    payload = {
        "namespace": namespace,
        "filter": metadata_filter,
        "limit": limit,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        **_requests_kwargs(),
    )
    response.raise_for_status()
    return parse_fetch_records(response.json())


def lookup_case_by_order_id(namespace, order_id, query_text=None, top_k=5):
    if not order_id:
        return []

    metadata_filter = {"order_id": {"$eq": order_id}}

    try:
        fetched = fetch_by_metadata(
            namespace=namespace,
            metadata_filter=metadata_filter,
            limit=top_k,
        )
        if fetched:
            return fetched
    except Exception:
        pass

    lookup_text = query_text or f"order id {order_id}"
    return exact_metadata_lookup(
        namespace=namespace,
        field_name="order_id",
        field_value=order_id,
        query_text=lookup_text,
        top_k=top_k,
        include_metadata=True,
    )


def hybrid_case_search(
    namespace,
    query_text,
    order_id=None,
    extra_filter=None,
    top_k=5,
):
    order_filter = {"order_id": {"$eq": order_id}} if order_id else None
    metadata_filter = _merge_filters(order_filter, extra_filter)

    return query_namespace(
        namespace=namespace,
        query_text=query_text,
        top_k=top_k,
        metadata_filter=metadata_filter,
        include_metadata=True,
    )
