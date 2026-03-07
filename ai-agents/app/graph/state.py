from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
import operator

class AgentState(TypedDict):
    """
    The shared state used by all agents in the College Management System.
    """
    # -- Input & Classification --
    user_input: str
    intent: str
    scope: str
    task_queue: List[Dict[str, Any]]
    current_task: Optional[Dict[str, Any]]
    operation_type: str
    table_name: str
    schema: Dict[str, Any]
    data: Any
    query: str
    missing_fields: List[str]
    inferred_fields: Dict[str, Any]
    validation_results: Dict[str, Any]
    execution_result: Any
    response: str
    analysis: str
    memory: Dict[str, Any]
    messages: Annotated[List[Any], operator.add]
   
