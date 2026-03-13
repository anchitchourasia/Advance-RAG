import sqlite3
from pathlib import Path
from config import DB_PATH

Path("data").mkdir(exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    tier TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT,
    product TEXT,
    status TEXT,
    refund_eligible INTEGER,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    issue TEXT,
    priority TEXT,
    status TEXT DEFAULT 'open'
)
""")

cur.executemany(
    "INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?)",
    [
        ("C101", "Aarav Sharma", "aarav@example.com", "gold"),
        ("C102", "Diya Patel", "diya@example.com", "silver"),
    ]
)

cur.executemany(
    "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?)",
    [
        ("O5001", "C101", "Wireless Headphones", "delayed", 1),
        ("O5002", "C102", "Smart Watch", "delivered", 0),
    ]
)

conn.commit()
conn.close()
