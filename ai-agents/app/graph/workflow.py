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
from app.agents.interaction.interaction_planner_agent import interaction_planner_agent
from app.agents.query.query_builder_agent import query_builder_agent
from app.agents.execution.execution_agent import execution_agent
from app.agents.response.response_formatter_agent import response_formatter_agent
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
    scope = state.get("scope", "general")
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
    Routes based on whether the schema is complete.
    """
    if state.get("schema_ready") == True:
        return "query_builder"
    return "interaction_planner"

def router_interaction_loop(state: AgentState):
    """
    Controls the interaction loop between InteractionPlannerAgent and CreateTableAgent.
    """
    if state.get("user_provided_missing_data") == True:
        return "create_table"
    return "formatter"

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
    workflow.add_node("interaction_planner", interaction_planner_agent)
    workflow.add_node("query_builder", query_builder_agent)
    workflow.add_node("execution", execution_agent)
    workflow.add_node("formatter", response_formatter_agent)
    
    # Entrance
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "memory")
    workflow.add_edge("memory", "validation")
    
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
    workflow.add_edge("ddl_router", "create_table")
    
    workflow.add_conditional_edges(
        "create_table",
        router_ddl_completion,
        {
            "interaction_planner": "interaction_planner",
            "query_builder": "query_builder"
        }
    )
    
    # Interaction Loop
    workflow.add_conditional_edges(
        "interaction_planner",
        router_interaction_loop,
        {
            "create_table": "create_table",
            "formatter": "formatter"
        }
    )
    
    workflow.add_edge("query_builder", "execution")
    workflow.add_edge("execution", "formatter")
        
    # Non-database Convergence
    workflow.add_edge("conversation", "formatter")
    workflow.add_edge("general_qa", "formatter")
    
    # Final Exit
    workflow.add_edge("formatter", END)
    
    return workflow.compile()
