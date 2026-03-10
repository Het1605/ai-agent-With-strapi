from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
import operator

class AgentState(TypedDict):
    """
    The shared state used by all agents in the College Management System.
    """
    # -- Input & Classification --
    user_input: str
    user_query: str
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
    execution_error: Optional[str]
    interaction_phase: bool
    active_agent: Optional[str]
    interaction_attempts: int
    response: str
    analysis: str
    planned_task: str
    current_task_index: int
    intent_category: str
    ddl_operation: str
    dml_operation: str
    conversation_history: List[Dict[str, str]]
    field_registry: Dict[str, Any]
    schema_data: Dict[str, Any]
    schema_ready: bool
    debug_info: str
    interaction_request: Dict[str, Any]
    interaction_message: str
    strapi_payload: Dict[str, Any]
    strapi_endpoint: str
    user_provided_missing_data: bool
    interaction_complete: bool
    route_decision: Optional[str]
    memory: Dict[str, Any]
    messages: Annotated[List[Any], operator.add]
   
