from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.agents.supervisor.supervisor_agent import supervisor_agent
from app.agents.validation.input_validation_agent import input_validation_agent
from app.agents.classifier.scope_classifier_agent import scope_classifier_agent
from app.agents.conversation.conversation_agent import conversation_agent
from app.agents.conversation.general_qa_agent import general_qa_agent
from app.agents.planner.task_planner_agent import task_planner_agent
from app.agents.planner.intent_router_agent import intent_router_agent
from app.agents.ddl.ddl_router_agent import ddl_router_agent
from app.agents.dml.dml_router_agent import dml_router_agent
from app.agents.ddl.create_table_agent import create_table_agent
from app.agents.ddl.modify_schema_agent import modify_schema_agent
from app.agents.ddl.add_column_agent import add_column_agent
from app.agents.ddl.update_collection_agent import update_collection_agent
from app.agents.ddl.update_field_agent import update_field_agent
from app.agents.ddl.delete_field_agent import delete_field_agent
from app.agents.interaction.interaction_planner_agent import interaction_planner_agent
from app.agents.query.query_builder_agent import query_builder_agent
from app.agents.execution.execution_agent import execution_agent
from app.agents.response.response_formatter_agent import response_formatter_agent
from app.agents.routing.state_router_agent import state_router_agent
from app.memory.memory_manager import memory_manager

def router_validation(state: AgentState):
    """
    Routes based on validation success.
    """
    if state["validation_results"].get("input_validation", {}).get("is_valid") == False:
        return "formatter"
    return "classifier"

def router_scope(state: AgentState):
    """
    Routes based on determined scope.
    """
    scope = state.get("scope")
    if scope == "conversation":
        return "conversation"
    elif scope == "database":
        return "planner"
    else:
        return "general_qa"

def router_intent_category(state: AgentState):
    """
    AI-driven conditional mapping from intent category to specialized router.
    """
    category = state.get("intent_category", "")
    if category == "DDL":
        return "ddl_router"
    elif category == "DML":
        return "dml_router"
    else:
        return "formatter"

def router_ddl_completion(state: AgentState):
    """
    Routes based on whether the DDL schema data is complete.
    Shared by both CreateTableAgent and ModifySchemaAgent.
    """
    if state.get("schema_ready") == True:
        return "query_builder"
    return "interaction_planner"

def router_ddl_operation(state: AgentState):
    """
    Routes from DDLRouterAgent to the correct DDL agent based on ddl_operation.
    """
    op = state.get("ddl_operation", "DDL_CREATE_TABLE")
    if op == "DDL_MODIFY_SCHEMA":
        return "modify_schema"
    return "create_table"

def router_modify_schema_operation(state: AgentState):
    """
    Routes from ModifySchemaAgent (classifier) to the correct sub-agent
    based on state['operation'] written by ModifySchemaAgent.
    """
    op = state.get("operation")
    allowed = {"add_column", "update_collection", "update_field", "delete_field"}
    if op not in allowed:
        print(f"[router_modify_schema_operation] Unknown operation '{op}' — defaulting to 'add_column'")
        return "add_column"
    print(f"[router_modify_schema_operation] Routing to '{op}'")
    return op

def router_state_decision(state: AgentState):
    """
    Reads the route decision written by StateRouterAgent and returns the
    target node name for LangGraph's conditional edge dispatcher.
    Falls back to 'validation' if the field is missing or unrecognised.
    """
    decision = state.get("route_decision", "validation")
    allowed = {
        "create_table", "modify_schema",
        "add_column", "update_collection", "update_field", "delete_field",
        "validation"
    }
    if decision not in allowed:
        print(f"[router_state_decision] Unexpected decision '{decision}' — defaulting to 'validation'.")
        return "validation"
    print(f"[router_state_decision] Dispatching to '{decision}'")
    return decision

def create_workflow():
    """
    Builds the production-grade LangGraph workflow with AI-driven multi-agent routing.
    """
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("memory", memory_manager)
    workflow.add_node("validation", input_validation_agent)
    workflow.add_node("classifier", scope_classifier_agent)
    workflow.add_node("conversation", conversation_agent)
    workflow.add_node("general_qa", general_qa_agent)
    workflow.add_node("planner", task_planner_agent)
    workflow.add_node("intent_router", intent_router_agent)
    workflow.add_node("ddl_router", ddl_router_agent)
    workflow.add_node("dml_router", dml_router_agent)
    workflow.add_node("create_table", create_table_agent)
    workflow.add_node("modify_schema", modify_schema_agent)   # classifier/router
    workflow.add_node("add_column", add_column_agent)
    workflow.add_node("update_collection", update_collection_agent)
    workflow.add_node("update_field", update_field_agent)
    workflow.add_node("delete_field", delete_field_agent)
    workflow.add_node("interaction_planner", interaction_planner_agent)
    workflow.add_node("query_builder", query_builder_agent)
    workflow.add_node("execution", execution_agent)
    workflow.add_node("formatter", response_formatter_agent)
    workflow.add_node("state_router", state_router_agent)
    
    # Entrance
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "memory")

    # MemoryManager → StateRouterAgent (LLM-based decision)
    # StateRouterAgent only routes to:
    #   • an active DDL sub-agent   (when interaction_phase == True)
    #   • validation                (fresh request — let full pipeline run)
    # It NEVER routes to intent_router (IntentRouterAgent must only run
    # after TaskPlannerAgent has built the task queue).
    workflow.add_edge("memory", "state_router")
    workflow.add_conditional_edges(
        "state_router",
        router_state_decision,
        {
            "create_table":       "create_table",
            "modify_schema":      "modify_schema",
            "add_column":         "add_column",
            "update_collection":  "update_collection",
            "update_field":       "update_field",
            "delete_field":       "delete_field",
            "validation":         "validation",
        }
    )

    
    # Scoping Phase
    workflow.add_conditional_edges(
        "validation",
        router_validation,
        {
            "classifier": "classifier",
            "formatter": "formatter"
        }
    )
    
    workflow.add_conditional_edges(
        "classifier",
        router_scope,
        {
            "conversation": "conversation",
            "general_qa": "general_qa",
            "planner": "planner"
        }
    )
    
    # Database Operations Phase
    workflow.add_edge("planner", "intent_router")
    
    workflow.add_conditional_edges(
        "intent_router",
        router_intent_category,
        {
            "ddl_router": "ddl_router",
            "dml_router": "dml_router",
            "formatter": "formatter"
        }
    )
    
    # DDL Specialized Sub-flow
    # DDLRouterAgent → CREATE or MODIFY branch
    workflow.add_conditional_edges(
        "ddl_router",
        router_ddl_operation,
        {
            "create_table":  "create_table",
            "modify_schema": "modify_schema",
        }
    )

    # CreateTableAgent is now autonomous; always goes to query_builder
    workflow.add_edge("create_table", "query_builder")

    # ModifySchemaAgent (classifier) routes to one of the 4 sub-agents
    workflow.add_conditional_edges(
        "modify_schema",
        router_modify_schema_operation,
        {
            "add_column":        "add_column",
            "update_collection": "update_collection",
            "update_field":      "update_field",
            "delete_field":      "delete_field",
        }
    )

    # Each sub-agent shares the same completion router
    for sub_agent in ["add_column", "update_collection", "update_field", "delete_field"]:
        workflow.add_conditional_edges(
            sub_agent,
            router_ddl_completion,
            {
                "interaction_planner": "interaction_planner",
                "query_builder":       "query_builder"
            }
        )
    
    # InteractionPlannerAgent has already written the question into state["response"].
    # Terminate the graph here so the question is returned to the user.
    # On the NEXT user turn, router_interaction_phase will route directly to create_table.
    workflow.add_edge("interaction_planner", END)
    
    workflow.add_edge("query_builder", "execution")
    workflow.add_edge("execution", "formatter")
        
    # Non-database Convergence
    workflow.add_edge("conversation", "formatter")
    workflow.add_edge("general_qa", "formatter")
    
    # Final Exit
    workflow.add_edge("formatter", END)
    
    return workflow.compile()
