import json
import re
from typing import Dict, Any, List, Optional

from config import (
    PINECONE_POLICY_NAMESPACE,
    PINECONE_CASE_NAMESPACE,
    APP_NAME,
    APP_ROLE,
    APP_IDENTITY,
)
from utils.helpers import extract_order_id, extract_customer_id, build_case_filter
from utils.langsmith_tracing import (
    traced_ask_gemini,
    traced_analyze_image,
    traced_query_namespace,
    traced_lookup_case_by_order_id,
    traced_hybrid_case_search,
    traced_get_order_status,
    traced_get_customer_profile,
    traced_create_support_ticket,
)

SUPPORT_KEYWORDS = [
    "order", "delivery", "delayed", "delay", "refund", "return", "replace",
    "replacement", "damaged", "damage", "broken", "crack", "defect", "ticket",
    "cancel", "cancellation", "payment", "charged", "upi", "warranty",
    "invoice", "shipment", "product issue", "complaint", "policy", "support",
    "csat", "category", "subcategory", "agent", "city", "shift", "tenure",
    "price", "survey"
]

DIRECT_IMAGE_PHRASES = [
    "what is this image",
    "whats this image",
    "what image is this",
    "what is in this image",
    "what is in the image",
    "describe this image",
    "describe the image",
    "tell me about this image",
    "what image i uploaded",
    "what did i upload",
    "analyze image",
    "analyse image",
    "look at image",
    "see image",
    "identify this image",
    "whats the image it is",
    "what it is in image",
]

AMBIGUOUS_IMAGE_PHRASES = [
    "what is this",
    "what is it",
    "what it is",
    "can you see now",
    "can u see now",
    "see now",
    "done",
    "now",
    "this image",
    "this one",
    "what about this",
    "can you see it",
    "can u see it",
]

IMAGE_CONTEXT_HINTS = [
    "image", "upload", "uploaded", "attach", "attached", "photo", "picture",
    "screenshot", "screen"
]

FIELD_SYNONYMS = {
    "category": ["category", "type"],
    "subcategory": ["subcategory", "sub-category", "sub category"],
    "channel": ["channel"],
    "city": ["city"],
    "product_category": ["product category", "product type"],
    "csat": ["csat", "csat score", "score", "rating"],
    "agent_name": ["who handled", "handled", "agent", "agent name", "handler"],
    "tenure_bucket": ["tenure", "tenure bucket"],
    "shift": ["shift", "agent shift"],
    "item_price": ["price", "item price", "cost"],
    "record_type": ["record type"],
    "source": ["source"],
    "order_id": ["order id"],
}

IDENTITY_QA = """
If the user asks who you are, your name, or what type of agent you are:
- Say: "I’m Support Intelligence Agent, an AI customer support assistant built for this application."

If the user asks what model powers you:
- Say: "I’m Support Intelligence Agent, powered primarily by Google Gemini for language and image understanding in this app, with Groq available as a backup provider for text responses."
"""


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def _normalize_soft(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _contains_any(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def _history_to_text(chat_history):
    if not chat_history:
        return ""

    lines = []
    for msg in chat_history[-8:]:
        role = str(msg.get("role", "user")).strip().title()
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _recent_image_context(chat_history) -> bool:
    if not chat_history:
        return False

    for msg in reversed(chat_history[-8:]):
        content = _normalize(msg.get("content", ""))
        if _contains_any(content, IMAGE_CONTEXT_HINTS):
            return True
    return False


def _recent_messages(state) -> List[Dict[str, Any]]:
    combined = []
    seen = set()

    for source in [state.get("messages", []) or [], state.get("chat_history", []) or []]:
        for msg in source[-12:]:
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()
            key = (role, content)
            if content and key not in seen:
                seen.add(key)
                combined.append({"role": role, "content": content})

    return combined[-12:]


def _resolve_recent_entity_from_history(
    state,
    extractor,
    current_query: str,
):
    current_query = str(current_query or "").strip()
    messages = _recent_messages(state)

    skipped_current = False
    for msg in reversed(messages):
        content = str(msg.get("content", "")).strip()
        if not content:
            continue

        if not skipped_current and current_query and content == current_query:
            skipped_current = True
            continue

        value = extractor(content)
        if value:
            return value

    return None


def _resolve_active_order_id(state, query: str) -> Optional[str]:
    current = state.get("order_id") or extract_order_id(query)
    if current:
        return current
    return _resolve_recent_entity_from_history(state, extract_order_id, query)


def _resolve_active_customer_id(state, query: str) -> Optional[str]:
    current = state.get("customer_id") or extract_customer_id(query)
    if current:
        return current
    return _resolve_recent_entity_from_history(state, extract_customer_id, query)


def _is_support_query(query: str, order_id: Optional[str], customer_id: Optional[str]) -> bool:
    q = _normalize(query)
    if order_id or customer_id:
        return True
    return _contains_any(q, SUPPORT_KEYWORDS)


def _is_general_image_question(query: str, chat_history=None, has_image=False) -> bool:
    q = _normalize(query)

    if _contains_any(q, DIRECT_IMAGE_PHRASES):
        return True

    if has_image and _contains_any(q, AMBIGUOUS_IMAGE_PHRASES):
        return True

    if has_image and len(q.split()) <= 6 and _recent_image_context(chat_history or []):
        return True

    return False


def _local_fast_reply(query: str):
    q = _normalize(query)

    if q in {"hi", "hello", "hey", "hii", "helo"}:
        return f"Hello! I’m {APP_NAME}. How can I help you today?"

    if q in {"who are you", "your name", "what is your name", "whats your name", "tell me your name"}:
        return f"I’m {APP_NAME}."

    if q in {"which model powers you", "what model powers you", "which model are you using", "what model are you using"}:
        return (
            f"I’m {APP_NAME}, powered primarily by Google Gemini for language and image understanding "
            f"in this app, with Groq available as a backup provider for text responses."
        )

    return None


def _looks_like_explicit_field_request(query: str, alias: str) -> bool:
    q = _normalize_soft(query)
    a = _normalize_soft(alias)

    if not q or not a or a not in q:
        return False

    if q == a or q == f"the {a}":
        return True

    short_forms = {
        f"{a} please",
        f"show {a}",
        f"tell {a}",
        f"share {a}",
    }
    if q in short_forms:
        return True

    ask_patterns = [
        rf"\bwhat\s+is\s+(the\s+)?{re.escape(a)}\b",
        rf"\bwhat's\s+(the\s+)?{re.escape(a)}\b",
        rf"\bwhich\s+{re.escape(a)}\b",
        rf"\btell\s+me\s+(the\s+)?{re.escape(a)}\b",
        rf"\bshow\s+me\s+(the\s+)?{re.escape(a)}\b",
        rf"\bgive\s+me\s+(the\s+)?{re.escape(a)}\b",
        rf"\bprovide\s+(the\s+)?{re.escape(a)}\b",
        rf"\bshare\s+(the\s+)?{re.escape(a)}\b",
        rf"\bcan\s+you\s+tell\s+me\s+(the\s+)?{re.escape(a)}\b",
        rf"\bi\s+want\s+(the\s+)?{re.escape(a)}\b",
        rf"\bneed\s+(the\s+)?{re.escape(a)}\b",
    ]

    if any(re.search(pattern, q) for pattern in ask_patterns):
        return True

    non_request_patterns = [
        rf"\bmy\s+{re.escape(a)}\s+is\b",
        rf"\bthe\s+{re.escape(a)}\s+is\b",
        rf"\bfor\s+{re.escape(a)}\b",
        rf"\bwith\s+{re.escape(a)}\b",
        rf"\b{re.escape(a)}\s*[:=]\s*",
    ]

    if any(re.search(pattern, q) for pattern in non_request_patterns):
        return False

    return False


def _extract_requested_case_field(query: str):
    for field, aliases in FIELD_SYNONYMS.items():
        for alias in aliases:
            if _looks_like_explicit_field_request(query, alias):
                return field
    return None


def _normalize_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _matched_case_from_hits(case_hits, requested_order_id):
    if not requested_order_id:
        return None

    target = _normalize_id(requested_order_id)

    for hit in case_hits or []:
        md = hit.get("metadata", {}) or {}
        if _normalize_id(md.get("order_id")) == target:
            return md

    return None


def _best_case_metadata(case_hits):
    if not case_hits:
        return None

    first = case_hits[0] or {}
    return first.get("metadata", {}) or None


def _direct_case_answer(query: str, metadata: dict):
    if not metadata:
        return None

    order_id = metadata.get("order_id", "this order")
    requested_field = _extract_requested_case_field(query)

    if not requested_field:
        return None

    value = metadata.get(requested_field)
    if value in [None, "", []]:
        return "That field is not available in the retrieved case record."

    templates = {
        "category": f"The category for order {order_id} is {value}.",
        "subcategory": f"The subcategory for order {order_id} is {value}.",
        "channel": f"The channel for order {order_id} is {value}.",
        "city": f"The city associated with order {order_id} is {value}.",
        "product_category": f"The product category for order {order_id} is {value}.",
        "csat": f"The CSAT score for order {order_id} is {value}.",
        "agent_name": f"The order was handled by {value}.",
        "tenure_bucket": f"The tenure bucket for order {order_id} is {value}.",
        "shift": f"The agent shift for order {order_id} is {value}.",
        "item_price": f"The item price for order {order_id} is {value}.",
        "record_type": f"The record type for order {order_id} is {value}.",
        "source": f"The source for order {order_id} is {value}.",
        "order_id": f"The order ID is {value}.",
    }

    return templates.get(requested_field, f"The {requested_field} for order {order_id} is {value}.")


def _flatten_filter_clauses(filter_obj: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(filter_obj, dict) or not filter_obj:
        return []

    if "$and" in filter_obj and isinstance(filter_obj["$and"], list):
        clauses = []
        for clause in filter_obj["$and"]:
            if isinstance(clause, dict) and clause:
                clauses.extend(_flatten_filter_clauses(clause))
        return clauses

    return [filter_obj]


def _merge_filters(*filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    seen = set()

    for f in filters:
        for clause in _flatten_filter_clauses(f):
            key = json.dumps(clause, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                clauses.append(clause)

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _build_identity_case_filter(order_id: Optional[str], customer_id: Optional[str]) -> Optional[Dict[str, Any]]:
    clauses = []
    if order_id:
        clauses.append({"order_id": {"$eq": order_id}})
    if customer_id:
        clauses.append({"customer_id": {"$eq": customer_id}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _build_policy_filter(query: str) -> Optional[Dict[str, Any]]:
    q = _normalize(query)
    category = None

    if any(x in q for x in ["refund", "return"]):
        category = "refund"
    elif any(x in q for x in ["shipping", "shipment", "delivery", "delayed", "delay"]):
        category = "shipping"
    elif any(x in q for x in ["warranty", "replace", "replacement", "damaged", "damage", "broken", "defect", "crack"]):
        category = "warranty"
    elif any(x in q for x in ["troubleshoot", "troubleshooting", "issue", "problem", "not working"]):
        category = "troubleshooting"

    base_filter = {"record_type": {"$eq": "policy"}}
    category_filter = {"category": {"$eq": category}} if category else None
    return _merge_filters(base_filter, category_filter)


def _query_mentions_delay(query: str) -> bool:
    q = _normalize(query)
    return any(x in q for x in ["delayed", "delay", "delivery", "shipment", "shipping"])


def _case_looks_like_delay_issue(metadata: dict) -> bool:
    if not metadata:
        return False

    haystack = " ".join(
        str(metadata.get(k, ""))
        for k in ["category", "subcategory", "chunk_text", "text", "source"]
    ).lower()

    return any(x in haystack for x in ["delayed", "delay", "delivery", "shipment", "shipping"])


def _extract_case_highlights(text: str) -> Dict[str, str]:
    text = " ".join(str(text or "").split())
    if not text:
        return {}

    normalized = text.replace(" | ", "|")
    parts = [p.strip() for p in normalized.split("|") if p.strip()]
    highlights: Dict[str, str] = {}

    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if value:
            highlights[key] = value

    return highlights


def _clean_case_text(text: str) -> str:
    highlights = _extract_case_highlights(text)

    preferred_parts = []
    if highlights.get("customer issue"):
        preferred_parts.append(f"Customer issue: {highlights['customer issue']}")
    if highlights.get("response or status note"):
        preferred_parts.append(f"Response or status note: {highlights['response or status note']}")

    if preferred_parts:
        summary = " ".join(preferred_parts)
        return summary[:260].strip()

    text = " ".join(str(text or "").split())
    return text[:220].strip()


def _format_case_summary(query: str, metadata: dict) -> Optional[str]:
    if not metadata:
        return None

    order_id = metadata.get("order_id", "the order")
    category = metadata.get("category")
    subcategory = metadata.get("subcategory")
    product_category = metadata.get("product_category")
    channel = metadata.get("channel")
    shift = metadata.get("shift")
    city = metadata.get("city")
    agent_name = metadata.get("agent_name")
    item_price = metadata.get("item_price")
    csat = metadata.get("csat")
    chunk_text = _clean_case_text(metadata.get("chunk_text") or metadata.get("text") or "")

    lines = [f"I found a matching case for order {order_id}."]

    if _query_mentions_delay(query) and not _case_looks_like_delay_issue(metadata):
        lines.append("")
        lines.append(
            "This record does not look like a delayed-delivery issue. "
            "It appears to be a return or quality-related case."
        )

    detail_lines = []
    if category:
        detail_lines.append(f"- Category: {category}")
    if subcategory:
        detail_lines.append(f"- Subcategory: {subcategory}")
    if product_category:
        detail_lines.append(f"- Product category: {product_category}")
    if channel:
        detail_lines.append(f"- Channel: {channel}")
    if shift:
        detail_lines.append(f"- Shift: {shift}")
    if city:
        detail_lines.append(f"- City: {city}")
    if agent_name:
        detail_lines.append(f"- Agent: {agent_name}")
    if item_price not in [None, ""]:
        detail_lines.append(f"- Item price: {item_price}")
    if csat not in [None, ""]:
        detail_lines.append(f"- CSAT: {csat}")

    if detail_lines:
        lines.append("")
        lines.append("Case details:")
        lines.extend(detail_lines)

    if chunk_text:
        lines.append("")
        lines.append(f"Summary: {chunk_text}")

    return "\n".join(lines).strip()


def _summarize_case_metadata(query: str, metadata: dict) -> Optional[str]:
    return _format_case_summary(query, metadata)


def prepare_context_node(state):
    chat_history = state.get("chat_history", []) or []
    query = state.get("query", "")

    messages = state.get("messages")
    if not messages:
        messages = list(chat_history)
        if query:
            if not messages or messages[-1].get("content") != query:
                messages.append({"role": "user", "content": query})

    route_debug = dict(state.get("route_debug", {}) or {})
    route_debug.update(
        {
            "user_id": state.get("user_id"),
            "session_id": state.get("session_id"),
            "thread_id": state.get("thread_id"),
            "message_count": len(messages),
        }
    )

    return {
        "messages": messages,
        "route_debug": route_debug,
    }


def build_retrieval_filters_node(state):
    query = state.get("query", "")
    order_id = state.get("order_id")
    customer_id = state.get("customer_id")
    intent = state.get("intent", "")
    existing_filters = dict(state.get("filters", {}) or {})

    policy_filter = existing_filters.get("policy") or _build_policy_filter(query)

    case_filter = existing_filters.get("case")
    if case_filter is None:
        case_filter = build_case_filter(query)

    identity_filter = _build_identity_case_filter(order_id, customer_id)
    case_filter = _merge_filters(case_filter, identity_filter)

    filters = dict(existing_filters)
    if intent in {"support", "image_support"}:
        filters["policy"] = policy_filter
        filters["case"] = case_filter

    retrieval_debug = dict(state.get("retrieval_debug", {}) or {})
    retrieval_debug.update(
        {
            "policy_filter": policy_filter,
            "case_filter": case_filter,
        }
    )

    return {
        "filters": filters,
        "retrieval_debug": retrieval_debug,
    }


def route_node(state):
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    has_image = bool(state.get("image_path"))

    order_id = _resolve_active_order_id(state, query)
    customer_id = _resolve_active_customer_id(state, query)

    is_support = _is_support_query(query, order_id, customer_id)
    asks_general_image = _is_general_image_question(query, chat_history, has_image)

    if has_image and asks_general_image and not is_support:
        intent = "image_general"
    elif has_image and is_support:
        intent = "image_support"
    elif is_support:
        intent = "support"
    else:
        intent = "general"

    route_debug = dict(state.get("route_debug", {}) or {})
    route_debug.update(
        {
            "has_image": has_image,
            "asks_general_image": asks_general_image,
            "is_support": is_support,
            "resolved_order_id": order_id,
            "resolved_customer_id": customer_id,
            "intent": intent,
        }
    )

    return {
        "intent": intent,
        "order_id": order_id,
        "customer_id": customer_id,
        "route_debug": route_debug,
    }


def choose_route(state):
    intent = state.get("intent", "general")
    if intent == "general":
        return "general_answer"
    if intent == "image_general":
        return "vision_general"
    return "policy_retrieval"


def vision_general_node(state):
    image_path = state.get("image_path")
    query = state.get("query", "")

    if not image_path:
        return {"vision_output": ""}

    prompt = (
        f"You are {APP_NAME}, a helpful visual assistant. "
        f"The user asked: {query}. "
        "Describe the uploaded image clearly and simply."
    )

    try:
        return {"vision_output": traced_analyze_image(image_path, prompt)}
    except Exception as e:
        return {"vision_output": f"Image analysis unavailable: {e}"}


def policy_retrieval_node(state):
    try:
        filters = dict(state.get("filters", {}) or {})
        policy_filter = filters.get("policy")

        hits = traced_query_namespace(
            namespace=PINECONE_POLICY_NAMESPACE,
            query_text=state["query"],
            top_k=3,
            metadata_filter=policy_filter,
            include_metadata=True,
        )

        retrieval_debug = dict(state.get("retrieval_debug", {}) or {})
        retrieval_debug.update(
            {
                "policy_namespace": PINECONE_POLICY_NAMESPACE,
                "policy_filter_used": policy_filter,
                "policy_hit_count": len(hits or []),
            }
        )

        return {
            "policy_hits": hits,
            "retrieval_debug": retrieval_debug,
        }
    except Exception as e:
        return {"policy_hits": [], "policy_error": str(e)}


def case_retrieval_node(state):
    try:
        query = state.get("query", "")
        order_id = state.get("order_id")
        filters = dict(state.get("filters", {}) or {})
        state_case_filter = filters.get("case")

        if order_id:
            exact_hits = traced_lookup_case_by_order_id(
                namespace=PINECONE_CASE_NAMESPACE,
                order_id=order_id,
                query_text=query,
                top_k=3,
            )
            if exact_hits:
                retrieval_debug = dict(state.get("retrieval_debug", {}) or {})
                retrieval_debug.update(
                    {
                        "case_namespace": PINECONE_CASE_NAMESPACE,
                        "case_lookup_mode": "exact_order_id",
                        "case_filter_used": {"order_id": {"$eq": order_id}},
                        "case_hit_count": len(exact_hits or []),
                    }
                )
                return {
                    "case_hits": exact_hits,
                    "retrieval_debug": retrieval_debug,
                }

        query_filter = build_case_filter(query)
        metadata_filter = _merge_filters(query_filter, state_case_filter)

        search_hits = traced_hybrid_case_search(
            namespace=PINECONE_CASE_NAMESPACE,
            query_text=query,
            order_id=None,
            extra_filter=metadata_filter,
            top_k=4,
        )

        retrieval_debug = dict(state.get("retrieval_debug", {}) or {})
        retrieval_debug.update(
            {
                "case_namespace": PINECONE_CASE_NAMESPACE,
                "case_lookup_mode": "hybrid_search",
                "case_filter_used": metadata_filter,
                "case_hit_count": len(search_hits or []),
            }
        )

        return {
            "case_hits": search_hits,
            "retrieval_debug": retrieval_debug,
        }

    except Exception as e:
        return {
            "case_hits": [],
            "case_error": str(e),
        }


def collect_retrieved_docs_node(state):
    docs: List[Dict[str, Any]] = []

    for source_name, hits in [
        ("policy", state.get("policy_hits", []) or []),
        ("case", state.get("case_hits", []) or []),
    ]:
        for hit in hits:
            metadata = hit.get("metadata", {}) or {}
            docs.append(
                {
                    "source_type": source_name,
                    "id": hit.get("id"),
                    "score": hit.get("score"),
                    "metadata": metadata,
                    "text": (
                        hit.get("chunk_text")
                        or hit.get("text")
                        or metadata.get("chunk_text")
                        or metadata.get("text")
                        or ""
                    ),
                }
            )

    retrieval_debug = dict(state.get("retrieval_debug", {}) or {})
    retrieval_debug.update({"retrieved_doc_count": len(docs)})

    return {
        "retrieved_docs": docs,
        "retrieval_debug": retrieval_debug,
    }


def tool_node(state):
    result = {}

    try:
        if state.get("order_id"):
            result["order"] = traced_get_order_status(state["order_id"])

        if state.get("customer_id"):
            result["customer"] = traced_get_customer_profile(state["customer_id"])

        q = _normalize(state.get("query", ""))
        if any(x in q for x in ["create ticket", "raise ticket", "open ticket", "complaint"]) and state.get("customer_id"):
            result["ticket"] = traced_create_support_ticket(
                customer_id=state["customer_id"],
                issue=state["query"],
                priority="high",
            )
    except Exception as e:
        result["error"] = str(e)

    return {"tool_output": result}


def vision_support_node(state):
    image_path = state.get("image_path")
    query = state.get("query", "")

    if not image_path:
        return {"vision_output": ""}

    prompt = (
        f"You are {APP_NAME}, a customer support image analyst. "
        f"The user asked: {query}. "
        "Identify what the image shows and mention visible support-relevant issues if any."
    )

    try:
        return {"vision_output": traced_analyze_image(image_path, prompt)}
    except Exception as e:
        return {"vision_output": f"Image analysis unavailable: {e}"}


def general_answer_node(state):
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    image_text = state.get("vision_output", "")
    has_image = bool(state.get("image_path"))
    memory_context = state.get("memory_context", []) or []

    fast_reply = _local_fast_reply(query)
    if fast_reply:
        return {"final_answer": fast_reply}

    asks_image = _is_general_image_question(query, chat_history, has_image)

    if asks_image and not has_image:
        return {"final_answer": "No image is currently uploaded, so I can’t describe it yet. Please attach an image from the sidebar."}

    if asks_image and has_image:
        lowered = image_text.lower() if image_text else ""
        bad_markers = ["temporarily unavailable", "image analysis unavailable", "couldn’t analyze", "couldn't analyze"]
        if image_text and not any(marker in lowered for marker in bad_markers):
            return {"final_answer": image_text}
        return {"final_answer": "I can see that an image is attached, but Gemini image analysis is temporarily unavailable right now. Please try again in a moment."}

    history_text = _history_to_text(chat_history)
    memory_text = json.dumps(memory_context, ensure_ascii=False, indent=2) if memory_context else "[]"

    prompt = f"""
{APP_IDENTITY}

{IDENTITY_QA}

You are {APP_NAME}, a {APP_ROLE}.

Conversation history:
{history_text or "No prior chat history."}

Optional memory context:
{memory_text}

Current user message:
{query}

Rules:
1. Answer naturally.
2. Keep the reply concise.
3. Keep identity consistent.
4. Use memory context only if it is relevant.
"""
    try:
        return {"final_answer": traced_ask_gemini(prompt)}
    except Exception as e:
        return {"final_answer": f"I hit an error while generating a response: {e}"}


def support_answer_node(state):
    query = state.get("query", "")
    fast_reply = _local_fast_reply(query)
    if fast_reply:
        return {"final_answer": fast_reply}

    case_hits = state.get("case_hits", []) or []
    policy_hits = state.get("policy_hits", []) or []
    tool_output = state.get("tool_output", {}) or {}
    requested_order_id = state.get("order_id")
    memory_context = state.get("memory_context", []) or []
    retrieval_debug = state.get("retrieval_debug", {}) or {}
    vision_output = state.get("vision_output", "")

    matched_case = _matched_case_from_hits(case_hits, requested_order_id)
    if not matched_case and _extract_requested_case_field(query):
        matched_case = _best_case_metadata(case_hits)

    exact_answer = _direct_case_answer(query, matched_case) if matched_case else None
    if exact_answer:
        return {"final_answer": exact_answer}

    if matched_case:
        case_summary = _summarize_case_metadata(query, matched_case)
        if case_summary:
            return {"final_answer": case_summary}

    order_tool = tool_output.get("order")
    if requested_order_id and order_tool:
        prompt = f"""
{APP_IDENTITY}

You are {APP_NAME}, a {APP_ROLE}.

User question:
{query}

Exact order tool output:
{json.dumps(order_tool, ensure_ascii=False, indent=2)}

Rules:
1. Answer directly from the order tool output.
2. Do not say you lack access.
3. Do not ask for unnecessary extra details.
4. Keep the answer concise and factual.
"""
        try:
            return {"final_answer": traced_ask_gemini(prompt)}
        except Exception as e:
            return {"final_answer": f"I hit an error while generating a support response: {e}"}

    policy_context = [
        {
            "id": hit.get("id"),
            "score": hit.get("score"),
            "metadata": hit.get("metadata", {}) or {},
            "text": (
                hit.get("chunk_text")
                or hit.get("text")
                or (hit.get("metadata", {}) or {}).get("chunk_text", "")
            ),
        }
        for hit in policy_hits[:3]
    ]

    case_context = [
        {
            "id": hit.get("id"),
            "score": hit.get("score"),
            "metadata": hit.get("metadata", {}) or {},
            "text": (
                hit.get("chunk_text")
                or hit.get("text")
                or (hit.get("metadata", {}) or {}).get("chunk_text", "")
            ),
        }
        for hit in case_hits[:3]
    ]

    if policy_context or case_context or vision_output:
        prompt = f"""
{APP_IDENTITY}

You are {APP_NAME}, a {APP_ROLE}.

User question:
{query}

Memory context:
{json.dumps(memory_context, ensure_ascii=False, indent=2)}

Policy retrieval results:
{json.dumps(policy_context, ensure_ascii=False, indent=2)}

Case retrieval results:
{json.dumps(case_context, ensure_ascii=False, indent=2)}

Tool output:
{json.dumps(tool_output, ensure_ascii=False, indent=2)}

Vision output:
{vision_output or "No image analysis."}

Retrieval debug:
{json.dumps(retrieval_debug, ensure_ascii=False, indent=2)}

Rules:
1. Prefer exact case facts when present.
2. Use policy hits for policy/process guidance.
3. Use tool output when it directly answers the user.
4. Use vision output only if it is relevant.
5. If the retrieved context is insufficient, say what is missing briefly.
6. Do not invent order-specific facts.
7. If the user asks a short follow-up, interpret it using the most recent support context in this conversation.
8. Keep the answer concise and helpful.
"""
        try:
            return {"final_answer": traced_ask_gemini(prompt)}
        except Exception as e:
            return {"final_answer": f"I hit an error while generating a support response: {e}"}

    return {
        "final_answer": (
            "I couldn’t find an exact matching case record for that order ID. "
            "Please verify the order ID or try another support question."
        )
    }
