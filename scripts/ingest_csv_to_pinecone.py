import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # ← now works
import pandas as pd
import re
import time
from pathlib import Path
from config import RAW_CSV_PATH, PINECONE_CASE_NAMESPACE
from retrieval.pinecone_store import get_index, embed_texts


LIMIT_ROWS = 300

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_column(df, candidates):
    normalized = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in normalized:
            return normalized[key]
    for col in df.columns:
        col_n = norm(col)
        if any(norm(cand) in col_n for cand in candidates):
            return col
    return None

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    return "" if x.lower() == "nan" else x

def to_float_or_none(x):
    try:
        if pd.isna(x) or str(x).strip() == "":
            return None
        return float(x)
    except:
        return None

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def main():
    start = time.time()
    print("Step 1: Starting CSV ingestion...", flush=True)
    print(f"RAW_CSV_PATH = {Path(RAW_CSV_PATH).resolve()}", flush=True)

    index = get_index()
    print("Step 2: Pinecone index client ready.", flush=True)

    df = pd.read_csv(RAW_CSV_PATH).head(LIMIT_ROWS)
    print(f"Step 3: CSV loaded. Rows={len(df)}, Cols={len(df.columns)}", flush=True)

    col_map = {
        "uid": find_column(df, ["unique id", "unique_id", "id"]),
        "channel": find_column(df, ["channel_name", "channel"]),
        "category": find_column(df, ["category"]),
        "subcategory": find_column(df, ["sub-category", "sub_category", "subcategory"]),
        "customer_text": find_column(df, ["customer remarks", "customer_remark", "complaint", "issue", "issue_reported"]),
        "response_text": find_column(df, ["issue_responded in", "issue_response", "resolution", "agent_response"]),
        "survey_date": find_column(df, ["survey_response_date", "survey_res", "survey"]),
        "city": find_column(df, ["customer_city", "city", "customer city"]),
        "product_category": find_column(df, ["product_category", "product category"]),
        "item_price": find_column(df, ["item_price", "price"]),
        "agent_name": find_column(df, ["agent_name", "agent name"]),
        "csat": find_column(df, ["csat", "csat_score", "csat score"]),
        "tenure_bucket": find_column(df, ["tenure bucket", "tenure_buck"]),
        "shift": find_column(df, ["agent shift", "shift"]),
        "order_id": find_column(df, ["order_id", "order id"]),
    }

    records = []
    for idx, row in df.iterrows():
        uid = clean_text(row[col_map["uid"]]) if col_map["uid"] else f"row-{idx}"

        channel = clean_text(row[col_map["channel"]]) if col_map["channel"] else ""
        category = clean_text(row[col_map["category"]]) if col_map["category"] else ""
        subcategory = clean_text(row[col_map["subcategory"]]) if col_map["subcategory"] else ""
        customer_text = clean_text(row[col_map["customer_text"]]) if col_map["customer_text"] else ""
        response_text = clean_text(row[col_map["response_text"]]) if col_map["response_text"] else ""
        survey_date = clean_text(row[col_map["survey_date"]]) if col_map["survey_date"] else ""
        city = clean_text(row[col_map["city"]]) if col_map["city"] else ""
        product_category = clean_text(row[col_map["product_category"]]) if col_map["product_category"] else ""
        agent_name = clean_text(row[col_map["agent_name"]]) if col_map["agent_name"] else ""
        tenure_bucket = clean_text(row[col_map["tenure_bucket"]]) if col_map["tenure_bucket"] else ""
        shift = clean_text(row[col_map["shift"]]) if col_map["shift"] else ""
        order_id = clean_text(row[col_map["order_id"]]) if col_map["order_id"] else ""
        item_price = to_float_or_none(row[col_map["item_price"]]) if col_map["item_price"] else None
        csat = clean_text(row[col_map["csat"]]) if col_map["csat"] else ""

        combined_text = " | ".join([
            x for x in [
                f"Channel: {channel}" if channel else "",
                f"Category: {category}" if category else "",
                f"Subcategory: {subcategory}" if subcategory else "",
                f"Customer issue: {customer_text}" if customer_text else "",
                f"Response or status note: {response_text}" if response_text else "",
                f"City: {city}" if city else "",
                f"Product category: {product_category}" if product_category else "",
                f"Order ID: {order_id}" if order_id else "",
                f"Survey date: {survey_date}" if survey_date else "",
                f"CSAT: {csat}" if csat else "",
                f"Tenure bucket: {tenure_bucket}" if tenure_bucket else "",
                f"Agent shift: {shift}" if shift else "",
            ] if x.strip()
        ])

        if len(combined_text) < 20:
            continue

        records.append({
            "id": f"case#{uid}",
            "text": combined_text,
            "metadata": {
                "source": "customer_support_data",
                "record_type": "historical_case",
                "category": category or "unknown",
                "subcategory": subcategory or "unknown",
                "channel": channel or "unknown",
                "city": city or "unknown",
                "product_category": product_category or "unknown",
                "agent_name": agent_name or "unknown",
                "csat": csat or "unknown",
                "tenure_bucket": tenure_bucket or "unknown",
                "shift": shift or "unknown",
                "order_id": order_id or "",
                "item_price": item_price if item_price is not None else -1.0,
                "chunk_text": combined_text[:1500]
            }
        })

    print(f"Prepared records={len(records)}", flush=True)

    texts = [r["text"] for r in records]
    embed_batch_size = 32
    vectors = []

    for batch_idx, text_batch in enumerate(chunked(texts, embed_batch_size), start=1):
        embs = embed_texts(text_batch)
        for rec, emb in zip(records[(batch_idx-1)*embed_batch_size : batch_idx*embed_batch_size], embs):
            vectors.append({
                "id": rec["id"],
                "values": emb,
                "metadata": rec["metadata"]
            })
        print(f"Embedded batch {batch_idx}", flush=True)

    upsert_batch_size = 50
    total = 0
    for batch_idx, batch in enumerate(chunked(vectors, upsert_batch_size), start=1):
        index.upsert(namespace=PINECONE_CASE_NAMESPACE, vectors=batch)
        total += len(batch)
        print(f"Upsert batch {batch_idx}: total_upserted={total}", flush=True)

    print(f"Done. Upserted {total} records.", flush=True)

if __name__ == "__main__":
    main()
