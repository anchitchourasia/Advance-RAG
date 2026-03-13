from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    route_node,
    choose_route,
    vision_general_node,
    policy_retrieval_node,
    case_retrieval_node,
    tool_node,
    vision_support_node,
    general_answer_node,
    support_answer_node,
)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.add_node("vision_general", vision_general_node)
    graph.add_node("policy_retrieval", policy_retrieval_node)
    graph.add_node("case_retrieval", case_retrieval_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("vision_support", vision_support_node)
    graph.add_node("general_answer", general_answer_node)
    graph.add_node("support_answer", support_answer_node)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        choose_route,
        {
            "general_answer": "general_answer",
            "vision_general": "vision_general",
            "policy_retrieval": "policy_retrieval",
        },
    )

    graph.add_edge("vision_general", "general_answer")
    graph.add_edge("policy_retrieval", "case_retrieval")
    graph.add_edge("case_retrieval", "tool_node")
    graph.add_edge("tool_node", "vision_support")
    graph.add_edge("vision_support", "support_answer")

    graph.add_edge("general_answer", END)
    graph.add_edge("support_answer", END)

    return graph.compile()
