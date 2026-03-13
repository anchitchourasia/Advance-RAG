import os
from contextlib import contextmanager

try:
    import langsmith as ls
    from langsmith import traceable
except Exception:
    ls = None

    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return decorator


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "false").lower() == "true"


def tracing_project() -> str:
    return os.getenv("LANGSMITH_PROJECT", "support-intelligence-agent")


@contextmanager
def maybe_tracing_context(project_name: str = None, enabled: bool = None):
    if ls is None:
        yield
        return

    with ls.tracing_context(
        enabled=tracing_enabled() if enabled is None else enabled,
        project_name=project_name or tracing_project(),
    ):
        yield


@traceable(name="ask_gemini", tags=["llm", "gemini"], metadata={"component": "ask_gemini"})
def traced_ask_gemini(prompt: str):
    from utils.gemini_client import ask_gemini
    return ask_gemini(prompt)


@traceable(name="analyze_image", tags=["vision", "gemini"], metadata={"component": "analyze_image"})
def traced_analyze_image(image_path: str, prompt: str):
    from utils.gemini_client import analyze_image
    return analyze_image(image_path, prompt)


@traceable(name="query_namespace", tags=["retrieval", "policy"], metadata={"component": "query_namespace"})
def traced_query_namespace(namespace: str, query_text: str, top_k: int, metadata_filter=None, include_metadata: bool = True):
    from retrieval.pinecone_store import query_namespace
    return query_namespace(
        namespace=namespace,
        query_text=query_text,
        top_k=top_k,
        metadata_filter=metadata_filter,
        include_metadata=include_metadata,
    )


@traceable(name="lookup_case_by_order_id", tags=["retrieval", "case"], metadata={"component": "lookup_case_by_order_id"})
def traced_lookup_case_by_order_id(namespace: str, order_id: str, query_text: str, top_k: int = 3):
    from retrieval.pinecone_store import lookup_case_by_order_id
    return lookup_case_by_order_id(
        namespace=namespace,
        order_id=order_id,
        query_text=query_text,
        top_k=top_k,
    )


@traceable(name="hybrid_case_search", tags=["retrieval", "case"], metadata={"component": "hybrid_case_search"})
def traced_hybrid_case_search(namespace: str, query_text: str, order_id=None, extra_filter=None, top_k: int = 4):
    from retrieval.pinecone_store import hybrid_case_search
    return hybrid_case_search(
        namespace=namespace,
        query_text=query_text,
        order_id=order_id,
        extra_filter=extra_filter,
        top_k=top_k,
    )


@traceable(name="get_order_status", tags=["tool", "order"], metadata={"component": "get_order_status"})
def traced_get_order_status(order_id: str):
    from tools.support_tools import get_order_status
    return get_order_status(order_id)


@traceable(name="get_customer_profile", tags=["tool", "customer"], metadata={"component": "get_customer_profile"})
def traced_get_customer_profile(customer_id: str):
    from tools.support_tools import get_customer_profile
    return get_customer_profile(customer_id)


@traceable(name="create_support_ticket", tags=["tool", "ticket"], metadata={"component": "create_support_ticket"})
def traced_create_support_ticket(customer_id: str, issue: str, priority: str = "high"):
    from tools.support_tools import create_support_ticket
    return create_support_ticket(
        customer_id=customer_id,
        issue=issue,
        priority=priority,
    )
