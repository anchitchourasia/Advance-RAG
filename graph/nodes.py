import json

from config import (
    PINECONE_POLICY_NAMESPACE,
    PINECONE_CASE_NAMESPACE,
    APP_NAME,
    APP_ROLE,
    APP_IDENTITY,
)
from utils.helpers import extract_order_id, extract_customer_id, build_case_filter
from utils.gemini_client import ask_gemini, analyze_image
from tools.support_tools import get_order_status, get_customer_profile, create_support_ticket


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


def _is_support_query(query: str, order_id: str | None, customer_id: str | None) -> bool:
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


def _extract_requested_case_field(query: str):
    q = _normalize(query)

    for field, aliases in FIELD_SYNONYMS.items():
        for alias in aliases:
            if alias in q:
                return field
    return None


def _matched_case_from_hits(case_hits, requested_order_id):
    if not requested_order_id:
        return None

    for hit in case_hits or []:
        md = hit.get("metadata", {}) or {}
        if str(md.get("order_id", "")).strip().lower() == str(requested_order_id).strip().lower():
            return md

    return None


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


def route_node(state):
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    has_image = bool(state.get("image_path"))

    order_id = extract_order_id(query)
    customer_id = extract_customer_id(query)

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

    return {
        "intent": intent,
        "order_id": order_id,
        "customer_id": customer_id,
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
        return {"vision_output": analyze_image(image_path, prompt)}
    except Exception as e:
        return {"vision_output": f"Image analysis unavailable: {e}"}


def policy_retrieval_node(state):
    try:
        from retrieval.pinecone_store import query_namespace

        hits = query_namespace(
            namespace=PINECONE_POLICY_NAMESPACE,
            query_text=state["query"],
            top_k=3,
            metadata_filter=None,
            include_metadata=True,
        )
        return {"policy_hits": hits}
    except Exception as e:
        return {"policy_hits": [], "policy_error": str(e)}


def case_retrieval_node(state):
    try:
        from retrieval.pinecone_store import (
            lookup_case_by_order_id,
            hybrid_case_search,
        )

        query = state.get("query", "")
        order_id = state.get("order_id")

        if order_id:
            exact_hits = lookup_case_by_order_id(
                namespace=PINECONE_CASE_NAMESPACE,
                order_id=order_id,
                query_text=query,
                top_k=3,
            )
            if exact_hits:
                return {"case_hits": exact_hits}

        metadata_filter = build_case_filter(query)

        search_hits = hybrid_case_search(
            namespace=PINECONE_CASE_NAMESPACE,
            query_text=query,
            order_id=None,
            extra_filter=metadata_filter,
            top_k=4,
        )

        return {"case_hits": search_hits}

    except Exception as e:
        return {
            "case_hits": [],
            "case_error": str(e),
        }


def tool_node(state):
    result = {}

    try:
        if state.get("order_id"):
            result["order"] = get_order_status(state["order_id"])

        if state.get("customer_id"):
            result["customer"] = get_customer_profile(state["customer_id"])

        q = _normalize(state.get("query", ""))
        if any(x in q for x in ["create ticket", "raise ticket", "open ticket", "complaint"]) and state.get("customer_id"):
            result["ticket"] = create_support_ticket(
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
        return {"vision_output": analyze_image(image_path, prompt)}
    except Exception as e:
        return {"vision_output": f"Image analysis unavailable: {e}"}


def general_answer_node(state):
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    image_text = state.get("vision_output", "")
    has_image = bool(state.get("image_path"))

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

    prompt = f"""
{APP_IDENTITY}

{IDENTITY_QA}

You are {APP_NAME}, a {APP_ROLE}.

Current user message:
{query}

Rules:
1. Answer naturally.
2. Keep the reply concise.
3. Keep identity consistent.
"""
    try:
        return {"final_answer": ask_gemini(prompt)}
    except Exception as e:
        return {"final_answer": f"I hit an error while generating a response: {e}"}


def support_answer_node(state):
    query = state.get("query", "")
    fast_reply = _local_fast_reply(query)
    if fast_reply:
        return {"final_answer": fast_reply}

    case_hits = state.get("case_hits", []) or []
    tool_output = state.get("tool_output", {}) or {}
    requested_order_id = state.get("order_id")

    matched_case = _matched_case_from_hits(case_hits, requested_order_id)
    exact_answer = _direct_case_answer(query, matched_case) if matched_case else None
    if exact_answer:
        return {"final_answer": exact_answer}

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
            return {"final_answer": ask_gemini(prompt)}
        except Exception as e:
            return {"final_answer": f"I hit an error while generating a support response: {e}"}

    return {
        "final_answer": (
            "I couldn’t find an exact matching case record for that order ID. "
            "Please verify the order ID or try another support question."
        )
    }
