import re
from typing import Optional, Dict, Any, List


UUID_PATTERN = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
ORD_PATTERN = r"\bORD[0-9A-Za-z_-]+\b"
CUSTOMER_PATTERN = r"\bCUST[0-9A-Za-z_-]+\b"


def _normalize_spaces(text: str) -> str:
    return " ".join((text or "").strip().split())


def _lower(text: str) -> str:
    return _normalize_spaces(text).lower()


def extract_order_id(text: str):
    text = _normalize_spaces(text)

    uuid_match = re.search(UUID_PATTERN, text, flags=re.IGNORECASE)
    if uuid_match:
        return uuid_match.group(0)

    ord_match = re.search(ORD_PATTERN, text, flags=re.IGNORECASE)
    if ord_match:
        return ord_match.group(0)

    return None


def extract_customer_id(text: str):
    text = _normalize_spaces(text)

    cust_match = re.search(CUSTOMER_PATTERN, text, flags=re.IGNORECASE)
    if cust_match:
        return cust_match.group(0)

    return None


def _extract_number_after_keywords(text: str, keywords: List[str]) -> Optional[float]:
    for kw in keywords:
        pattern = rf"{re.escape(kw)}\s*[:=]?\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _build_eq(field: str, value: Any) -> Dict[str, Any]:
    return {field: {"$eq": value}}


def _append_if(filters: List[Dict[str, Any]], condition: bool, field: str, value: Any):
    if condition:
        filters.append(_build_eq(field, value))


def build_case_filter(query: str):
    q = _normalize_spaces(query)
    q_lower = q.lower()

    filters: List[Dict[str, Any]] = []

    order_id = extract_order_id(q)
    if order_id:
        filters.append(_build_eq("order_id", order_id))

    customer_id = extract_customer_id(q)
    if customer_id:
        filters.append(_build_eq("customer_id", customer_id))

    _append_if(filters, ("delayed" in q_lower or "delay" in q_lower), "subcategory", "Delayed")
    _append_if(filters, ("refund" in q_lower), "subcategory", "Refund")
    _append_if(filters, ("return" in q_lower), "subcategory", "Return")
    _append_if(filters, ("damaged" in q_lower or "damage" in q_lower or "broken" in q_lower), "subcategory", "Damaged")

    _append_if(filters, ("order related" in q_lower), "category", "Order Related")
    _append_if(filters, ("payment" in q_lower or "charged" in q_lower or "upi" in q_lower), "category", "Payment")
    _append_if(filters, ("complaint" in q_lower or "ticket" in q_lower), "record_type", "case")

    _append_if(filters, ("outcall" in q_lower), "channel", "Outcall")
    _append_if(filters, ("incall" in q_lower or "inbound" in q_lower), "channel", "Incall")

    _append_if(filters, ("morning" in q_lower), "shift", "Morning")
    _append_if(filters, ("evening" in q_lower), "shift", "Evening")
    _append_if(filters, ("night" in q_lower), "shift", "Night")

    if "category" in q_lower and "subcategory" not in q_lower:
        category_match = re.search(r"category\s*(?:is|=|:)?\s*([A-Za-z ]+)", q, flags=re.IGNORECASE)
        if category_match:
            value = category_match.group(1).strip()
            if value:
                filters.append(_build_eq("category", value))

    if "subcategory" in q_lower or "sub-category" in q_lower or "sub category" in q_lower:
        subcategory_match = re.search(
            r"(?:subcategory|sub-category|sub category)\s*(?:is|=|:)?\s*([A-Za-z ]+)",
            q,
            flags=re.IGNORECASE,
        )
        if subcategory_match:
            value = subcategory_match.group(1).strip()
            if value:
                filters.append(_build_eq("subcategory", value))

    if "city" in q_lower:
        city_match = re.search(r"city\s*(?:is|=|:)?\s*([A-Za-z ]+)", q, flags=re.IGNORECASE)
        if city_match:
            value = city_match.group(1).strip()
            if value:
                filters.append(_build_eq("city", value))

    if "agent" in q_lower or "handled by" in q_lower:
        agent_match = re.search(r"(?:agent|handled by)\s*(?:is|=|:)?\s*([A-Za-z .]+)", q, flags=re.IGNORECASE)
        if agent_match:
            value = agent_match.group(1).strip()
            if value:
                filters.append(_build_eq("agent_name", value))

    if "source" in q_lower:
        source_match = re.search(r"source\s*(?:is|=|:)?\s*([A-Za-z0-9 _-]+)", q, flags=re.IGNORECASE)
        if source_match:
            value = source_match.group(1).strip()
            if value:
                filters.append(_build_eq("source", value))

    if "record type" in q_lower:
        record_type_match = re.search(r"record type\s*(?:is|=|:)?\s*([A-Za-z0-9 _-]+)", q, flags=re.IGNORECASE)
        if record_type_match:
            value = record_type_match.group(1).strip()
            if value:
                filters.append(_build_eq("record_type", value))

    if "product category" in q_lower or "product type" in q_lower:
        product_match = re.search(
            r"(?:product category|product type)\s*(?:is|=|:)?\s*([A-Za-z0-9 _-]+)",
            q,
            flags=re.IGNORECASE,
        )
        if product_match:
            value = product_match.group(1).strip()
            if value:
                filters.append(_build_eq("product_category", value))

    csat_value = _extract_number_after_keywords(q, ["csat", "csat score", "score", "rating"])
    if csat_value is not None:
        filters.append(_build_eq("csat", csat_value))

    price_value = _extract_number_after_keywords(q, ["price", "item price", "cost"])
    if price_value is not None:
        filters.append(_build_eq("item_price", price_value))

    deduped = []
    seen = set()
    for f in filters:
        key = repr(sorted(f.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    if len(deduped) == 1:
        return deduped[0]
    if len(deduped) > 1:
        return {"$and": deduped}

    return None
