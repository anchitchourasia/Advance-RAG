import sqlite3
from config import DB_PATH

def _query_one(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_order_status(order_id: str):
    return _query_one(
        "SELECT order_id, customer_id, product, status, refund_eligible FROM orders WHERE order_id = ?",
        (order_id,)
    )

def get_customer_profile(customer_id: str):
    return _query_one(
        "SELECT customer_id, name, email, tier FROM customers WHERE customer_id = ?",
        (customer_id,)
    )

def create_support_ticket(customer_id: str, issue: str, priority: str = "medium"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tickets (customer_id, issue, priority) VALUES (?, ?, ?)",
        (customer_id, issue, priority)
    )
    ticket_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "issue": issue,
        "priority": priority,
        "status": "open"
    }
