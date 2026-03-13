from pathlib import Path
import sqlite3

from langgraph.graph import StateGraph, END

from config import (
    LANGGRAPH_ENABLE_PERSISTENCE,
    LANGGRAPH_CHECKPOINTER,
    LANGGRAPH_SQLITE_PATH,
)
from graph.state import AgentState
from graph.nodes import (
    prepare_context_node,
    route_node,
    choose_route,
    vision_general_node,
    build_retrieval_filters_node,
    policy_retrieval_node,
    case_retrieval_node,
    collect_retrieved_docs_node,
    tool_node,
    vision_support_node,
    general_answer_node,
    support_answer_node,
)

_DEFAULT_CHECKPOINTER = None
_SQLITE_CONN = None


def _build_default_checkpointer():
    global _DEFAULT_CHECKPOINTER, _SQLITE_CONN

    if _DEFAULT_CHECKPOINTER is not None:
        return _DEFAULT_CHECKPOINTER

    if not LANGGRAPH_ENABLE_PERSISTENCE:
        return None

    checkpointer_type = (LANGGRAPH_CHECKPOINTER or "memory").strip().lower()

    if checkpointer_type == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = Path(LANGGRAPH_SQLITE_PATH)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            if _SQLITE_CONN is None:
                _SQLITE_CONN = sqlite3.connect(str(db_path), check_same_thread=False)

            _DEFAULT_CHECKPOINTER = SqliteSaver(_SQLITE_CONN)
            return _DEFAULT_CHECKPOINTER
        except Exception:
            pass

    try:
        from langgraph.checkpoint.memory import InMemorySaver

        _DEFAULT_CHECKPOINTER = InMemorySaver()
        return _DEFAULT_CHECKPOINTER
    except Exception:
        return None


def build_graph(checkpointer=None, store=None):
    graph = StateGraph(AgentState)

    graph.add_node("prepare_context", prepare_context_node)
    graph.add_node("route", route_node)
    graph.add_node("vision_general", vision_general_node)
    graph.add_node("build_retrieval_filters", build_retrieval_filters_node)
    graph.add_node("policy_retrieval", policy_retrieval_node)
    graph.add_node("case_retrieval", case_retrieval_node)
    graph.add_node("collect_retrieved_docs", collect_retrieved_docs_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("vision_support", vision_support_node)
    graph.add_node("general_answer", general_answer_node)
    graph.add_node("support_answer", support_answer_node)

    graph.set_entry_point("prepare_context")
    graph.add_edge("prepare_context", "route")

    graph.add_conditional_edges(
        "route",
        choose_route,
        {
            "general_answer": "general_answer",
            "vision_general": "vision_general",
            "policy_retrieval": "build_retrieval_filters",
        },
    )

    graph.add_edge("vision_general", "general_answer")
    graph.add_edge("build_retrieval_filters", "policy_retrieval")
    graph.add_edge("policy_retrieval", "case_retrieval")
    graph.add_edge("case_retrieval", "collect_retrieved_docs")
    graph.add_edge("collect_retrieved_docs", "tool_node")
    graph.add_edge("tool_node", "vision_support")
    graph.add_edge("vision_support", "support_answer")

    graph.add_edge("general_answer", END)
    graph.add_edge("support_answer", END)

    resolved_checkpointer = checkpointer if checkpointer is not None else _build_default_checkpointer()

    compile_kwargs = {}
    if resolved_checkpointer is not None:
        compile_kwargs["checkpointer"] = resolved_checkpointer
    if store is not None:
        compile_kwargs["store"] = store

    return graph.compile(**compile_kwargs)
