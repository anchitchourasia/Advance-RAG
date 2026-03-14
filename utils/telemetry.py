import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "telemetry.db"

_DB_LOCK = threading.Lock()


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"value": str(data)}, ensure_ascii=False)


def _to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row) if row is not None else {}


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_telemetry_db() -> None:
    with _DB_LOCK:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                thread_id TEXT,
                query TEXT,
                intent TEXT,
                status TEXT,
                has_image INTEGER DEFAULT 0,
                model_name TEXT,
                provider TEXT,
                total_latency_ms REAL DEFAULT 0,
                retrieval_latency_ms REAL DEFAULT 0,
                llm_latency_ms REAL DEFAULT 0,
                tool_latency_ms REAL DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                estimated_cost REAL DEFAULT 0,
                error_message TEXT,
                tags_json TEXT,
                metadata_json TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_name TEXT,
                status TEXT,
                latency_ms REAL DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                estimated_cost REAL DEFAULT 0,
                error_message TEXT,
                payload_json TEXT,
                FOREIGN KEY(trace_id) REFERENCES traces(trace_id)
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_traces_created_at
            ON traces(created_at)
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_traces_status
            ON traces(status)
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_traces_intent
            ON traces(intent)
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_trace_id
            ON events(trace_id)
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_created_at
            ON events(created_at)
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON events(event_type)
            """
        )

        conn.commit()
        conn.close()


def create_trace(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    query: Optional[str] = None,
    intent: Optional[str] = None,
    has_image: bool = False,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    init_telemetry_db()
    trace_id = str(uuid.uuid4())
    now = _utc_ts()

    with _DB_LOCK:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO traces (
                trace_id, created_at, updated_at, user_id, session_id, thread_id,
                query, intent, status, has_image, model_name, provider,
                tags_json, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                now,
                now,
                user_id,
                session_id,
                thread_id,
                query,
                intent,
                "running",
                1 if has_image else 0,
                model_name,
                provider,
                _safe_json(tags or []),
                _safe_json(metadata or {}),
            ),
        )
        conn.commit()
        conn.close()

    return trace_id


def update_trace(
    trace_id: str,
    *,
    intent: Optional[str] = None,
    status: Optional[str] = None,
    total_latency_ms: Optional[float] = None,
    retrieval_latency_ms: Optional[float] = None,
    llm_latency_ms: Optional[float] = None,
    tool_latency_ms: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    estimated_cost: Optional[float] = None,
    error_message: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> None:
    init_telemetry_db()
    now = _utc_ts()

    with _DB_LOCK:
        conn = get_connection()
        cur = conn.cursor()

        existing = cur.execute(
            "SELECT metadata_json, tags_json FROM traces WHERE trace_id = ?",
            (trace_id,),
        ).fetchone()

        existing_metadata = {}
        existing_tags = []

        if existing:
            try:
                existing_metadata = json.loads(existing["metadata_json"] or "{}")
            except Exception:
                existing_metadata = {}

            try:
                existing_tags = json.loads(existing["tags_json"] or "[]")
            except Exception:
                existing_tags = []

        merged_metadata = dict(existing_metadata)
        if metadata:
            merged_metadata.update(metadata)

        merged_tags = list(dict.fromkeys(existing_tags + (tags or [])))

        cur.execute(
            """
            UPDATE traces
            SET updated_at = ?,
                intent = COALESCE(?, intent),
                status = COALESCE(?, status),
                total_latency_ms = COALESCE(?, total_latency_ms),
                retrieval_latency_ms = COALESCE(?, retrieval_latency_ms),
                llm_latency_ms = COALESCE(?, llm_latency_ms),
                tool_latency_ms = COALESCE(?, tool_latency_ms),
                input_tokens = COALESCE(?, input_tokens),
                output_tokens = COALESCE(?, output_tokens),
                total_tokens = COALESCE(?, total_tokens),
                estimated_cost = COALESCE(?, estimated_cost),
                error_message = COALESCE(?, error_message),
                model_name = COALESCE(?, model_name),
                provider = COALESCE(?, provider),
                metadata_json = ?,
                tags_json = ?
            WHERE trace_id = ?
            """,
            (
                now,
                intent,
                status,
                total_latency_ms,
                retrieval_latency_ms,
                llm_latency_ms,
                tool_latency_ms,
                input_tokens,
                output_tokens,
                total_tokens,
                estimated_cost,
                error_message,
                model_name,
                provider,
                _safe_json(merged_metadata),
                _safe_json(merged_tags),
                trace_id,
            ),
        )

        conn.commit()
        conn.close()


def log_event(
    trace_id: str,
    event_type: str,
    event_name: Optional[str] = None,
    status: str = "success",
    latency_ms: float = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    estimated_cost: float = 0,
    error_message: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    init_telemetry_db()
    event_id = str(uuid.uuid4())
    now = _utc_ts()

    with _DB_LOCK:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO events (
                event_id, trace_id, created_at, event_type, event_name, status,
                latency_ms, input_tokens, output_tokens, total_tokens,
                estimated_cost, error_message, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                trace_id,
                now,
                event_type,
                event_name,
                status,
                latency_ms,
                input_tokens,
                output_tokens,
                total_tokens,
                estimated_cost,
                error_message,
                _safe_json(payload or {}),
            ),
        )
        conn.commit()
        conn.close()

    return event_id


def finalize_trace(
    trace_id: str,
    status: str = "success",
    total_latency_ms: Optional[float] = None,
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> None:
    with _DB_LOCK:
        conn = get_connection()
        cur = conn.cursor()

        agg = cur.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN event_type = 'retrieval' THEN latency_ms ELSE 0 END), 0) AS retrieval_latency_ms,
                COALESCE(SUM(CASE WHEN event_type = 'llm' THEN latency_ms ELSE 0 END), 0) AS llm_latency_ms,
                COALESCE(SUM(CASE WHEN event_type = 'tool' THEN latency_ms ELSE 0 END), 0) AS tool_latency_ms,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(SUM(estimated_cost), 0) AS estimated_cost
            FROM events
            WHERE trace_id = ?
            """,
            (trace_id,),
        ).fetchone()

        conn.close()

    update_trace(
        trace_id,
        status=status,
        total_latency_ms=total_latency_ms,
        retrieval_latency_ms=float(agg["retrieval_latency_ms"] or 0),
        llm_latency_ms=float(agg["llm_latency_ms"] or 0),
        tool_latency_ms=float(agg["tool_latency_ms"] or 0),
        input_tokens=int(agg["input_tokens"] or 0),
        output_tokens=int(agg["output_tokens"] or 0),
        total_tokens=int(agg["total_tokens"] or 0),
        estimated_cost=float(agg["estimated_cost"] or 0),
        error_message=error_message,
        metadata=metadata,
        tags=tags,
    )


@contextmanager
def trace_request(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    query: Optional[str] = None,
    intent: Optional[str] = None,
    has_image: bool = False,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    start = time.perf_counter()
    trace_id = create_trace(
        user_id=user_id,
        session_id=session_id,
        thread_id=thread_id,
        query=query,
        intent=intent,
        has_image=has_image,
        model_name=model_name,
        provider=provider,
        tags=tags,
        metadata=metadata,
    )
    try:
        yield trace_id
        total_latency_ms = (time.perf_counter() - start) * 1000
        finalize_trace(
            trace_id=trace_id,
            status="success",
            total_latency_ms=total_latency_ms,
        )
    except Exception as e:
        total_latency_ms = (time.perf_counter() - start) * 1000
        finalize_trace(
            trace_id=trace_id,
            status="error",
            total_latency_ms=total_latency_ms,
            error_message=str(e),
        )
        raise


def get_recent_traces(limit: int = 100) -> List[Dict[str, Any]]:
    init_telemetry_db()
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT *
        FROM traces
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [_to_dict(row) for row in rows]


def get_trace_events(trace_id: str) -> List[Dict[str, Any]]:
    init_telemetry_db()
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT *
        FROM events
        WHERE trace_id = ?
        ORDER BY created_at ASC
        """,
        (trace_id,),
    ).fetchall()
    conn.close()
    return [_to_dict(row) for row in rows]


def run_query(sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    init_telemetry_db()
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute(sql, params or ()).fetchall()
    conn.close()
    return [_to_dict(row) for row in rows]


def get_kpis() -> Dict[str, Any]:
    init_telemetry_db()
    conn = get_connection()
    cur = conn.cursor()

    trace_row = cur.execute(
        """
        SELECT
            COUNT(*) AS total_traces,
            COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) AS success_traces,
            COALESCE(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END), 0) AS error_traces,
            COALESCE(AVG(total_latency_ms), 0) AS avg_trace_latency_ms,
            COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
            COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COALESCE(SUM(estimated_cost), 0) AS total_cost
        FROM traces
        """
    ).fetchone()

    event_row = cur.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN event_type = 'llm' THEN 1 ELSE 0 END), 0) AS llm_calls,
            COALESCE(SUM(CASE WHEN event_type = 'tool' THEN 1 ELSE 0 END), 0) AS tool_calls,
            COALESCE(SUM(CASE WHEN event_type = 'retrieval' THEN 1 ELSE 0 END), 0) AS retrieval_calls,
            COALESCE(AVG(CASE WHEN event_type = 'llm' THEN latency_ms END), 0) AS avg_llm_latency_ms,
            COALESCE(AVG(CASE WHEN event_type = 'tool' THEN latency_ms END), 0) AS avg_tool_latency_ms,
            COALESCE(AVG(CASE WHEN event_type = 'retrieval' THEN latency_ms END), 0) AS avg_retrieval_latency_ms
        FROM events
        """
    ).fetchone()

    conn.close()

    data = {}
    data.update(_to_dict(trace_row))
    data.update(_to_dict(event_row))
    return data


def get_time_series_by_day() -> List[Dict[str, Any]]:
    sql = """
    SELECT
        substr(created_at, 1, 10) AS day,
        COUNT(*) AS trace_count,
        COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) AS success_count,
        COALESCE(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END), 0) AS error_count,
        COALESCE(AVG(total_latency_ms), 0) AS avg_latency_ms,
        COALESCE(SUM(input_tokens), 0) AS input_tokens,
        COALESCE(SUM(output_tokens), 0) AS output_tokens,
        COALESCE(SUM(total_tokens), 0) AS total_tokens,
        COALESCE(SUM(estimated_cost), 0) AS total_cost
    FROM traces
    GROUP BY substr(created_at, 1, 10)
    ORDER BY day ASC
    """
    return run_query(sql)


def get_tool_usage() -> List[Dict[str, Any]]:
    sql = """
    SELECT
        COALESCE(event_name, 'unknown') AS tool_name,
        COUNT(*) AS run_count,
        COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
        COALESCE(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END), 0) AS error_count
    FROM events
    WHERE event_type = 'tool'
    GROUP BY COALESCE(event_name, 'unknown')
    ORDER BY run_count DESC, tool_name ASC
    """
    return run_query(sql)


def get_llm_usage() -> List[Dict[str, Any]]:
    sql = """
    SELECT
        COALESCE(event_name, 'unknown') AS llm_name,
        COUNT(*) AS run_count,
        COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
        COALESCE(SUM(input_tokens), 0) AS input_tokens,
        COALESCE(SUM(output_tokens), 0) AS output_tokens,
        COALESCE(SUM(total_tokens), 0) AS total_tokens,
        COALESCE(SUM(estimated_cost), 0) AS total_cost
    FROM events
    WHERE event_type = 'llm'
    GROUP BY COALESCE(event_name, 'unknown')
    ORDER BY run_count DESC, llm_name ASC
    """
    return run_query(sql)


def get_retrieval_usage() -> List[Dict[str, Any]]:
    sql = """
    SELECT
        COALESCE(event_name, 'unknown') AS retrieval_name,
        COUNT(*) AS run_count,
        COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
        COALESCE(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END), 0) AS error_count
    FROM events
    WHERE event_type = 'retrieval'
    GROUP BY COALESCE(event_name, 'unknown')
    ORDER BY run_count DESC, retrieval_name ASC
    """
    return run_query(sql)


def get_top_intents() -> List[Dict[str, Any]]:
    sql = """
    SELECT
        COALESCE(intent, 'unknown') AS intent,
        COUNT(*) AS trace_count
    FROM traces
    GROUP BY COALESCE(intent, 'unknown')
    ORDER BY trace_count DESC, intent ASC
    """
    return run_query(sql)


def get_error_traces(limit: int = 20) -> List[Dict[str, Any]]:
    sql = """
    SELECT
        trace_id, created_at, query, intent, status, error_message, total_latency_ms
    FROM traces
    WHERE status = 'error'
    ORDER BY created_at DESC
    LIMIT ?
    """
    return run_query(sql, (limit,))


init_telemetry_db()
