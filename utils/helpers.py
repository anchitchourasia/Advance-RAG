import re


UUID_PATTERN = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
ORD_PATTERN = r"\bORD[0-9A-Za-z_-]+\b"
CUSTOMER_PATTERN = r"\bCUST[0-9A-Za-z_-]+\b"


def _normalize_spaces(text: str) -> str:
    return " ".join((text or "").strip().split())


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


def build_case_filter(query: str):
    q = (query or "").strip()

    filters = []

    order_id = extract_order_id(q)
    if order_id:
        filters.append({"order_id": {"$eq": order_id}})

    q_lower = q.lower()

    if "delayed" in q_lower or "delay" in q_lower:
        filters.append({"subcategory": {"$eq": "Delayed"}})

    if "refund" in q_lower:
        filters.append({"subcategory": {"$eq": "Refund"}})

    if "return" in q_lower:
        filters.append({"subcategory": {"$eq": "Return"}})

    if "damaged" in q_lower or "damage" in q_lower or "broken" in q_lower:
        filters.append({"subcategory": {"$eq": "Damaged"}})

    if "order related" in q_lower:
        filters.append({"category": {"$eq": "Order Related"}})

    if "outcall" in q_lower:
        filters.append({"channel": {"$eq": "Outcall"}})

    if "morning" in q_lower:
        filters.append({"shift": {"$eq": "Morning"}})

    if "evening" in q_lower:
        filters.append({"shift": {"$eq": "Evening"}})

    if len(filters) == 1:
        return filters[0]
    if len(filters) > 1:
        return {"$and": filters}

    return None
