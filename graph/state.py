from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict, total=False):
    query: str
    intent: str

    order_id: Optional[str]
    customer_id: Optional[str]

    image_path: Optional[str]
    vision_output: str

    policy_hits: List[Dict[str, Any]]
    case_hits: List[Dict[str, Any]]

    tool_output: Dict[str, Any]

    final_answer: str
    chat_history: List[Dict[str, str]]

    policy_error: str
    case_error: str

    route_debug: Dict[str, Any]
    retrieval_debug: Dict[str, Any]
