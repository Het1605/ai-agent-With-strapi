from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.graph.state import AgentState
from app.agents.supervisor.supervisor_agent import supervisor_agent
from app.agents.validation.input_validation_agent import input_validation_agent
from app.agents.classifier.scope_classifier_agent import scope_classifier_agent
from app.agents.conversation.conversation_agent import conversation_agent
from app.agents.conversation.general_qa_agent import general_qa_agent
from app.agents.planner.intent_router_agent import intent_router_agent
from app.agents.ddl.ddl_router_agent import ddl_router_agent
from app.agents.dml.dml_router_agent import dml_router_agent
from app.agents.ddl.CreateTableAgents.requirement_agent import requirement_agent
from app.agents.ddl.CreateTableAgents.planning_agent import planning_agent
from app.agents.ddl.CreateTableAgents.schema_designer_agent import schema_designer_agent
from app.agents.ddl.ModifySchemaAgents.schema_context_loader_agent import schema_context_loader_agent
from app.agents.ddl.ModifySchemaAgents.modify_schema_intent_agent import modify_schema_intent_agent
from app.agents.ddl.ModifySchemaAgents.schema_planner_agent import schema_planner_agent as modify_schema_planner_agent
from app.agents.ddl.ModifySchemaAgents.schema_designer_agent import schema_designer_agent as modify_schema_designer_agent
from app.agents.ddl.ModifySchemaAgents.modify_schema_visualization_agent import modify_schema_visualization_agent
from app.agents.ddl.ModifySchemaAgents.user_approval_agent import user_approval_agent as modify_schema_user_approval_agent
from app.agents.ddl.ModifySchemaAgents.approval_decision_agent import approval_decision_agent as modify_schema_approval_decision_agent
from app.agents.ddl.ModifySchemaAgents.user_reprompt_agent import user_reprompt_agent as modify_schema_user_reprompt_agent
from app.agents.query.query_builder_agent import query_builder_agent
from app.agents.execution.execution_agent import execution_agent
from app.agents.error_handling_agents.execution_monitor_agent import execution_monitor_agent
from app.agents.error_handling_agents.log_analyzer_agent import log_analyzer_agent
from app.agents.error_handling_agents.schema_validator_agent import schema_validator_agent
from app.agents.error_handling_agents.dependency_checker_agent import dependency_checker_agent
from app.agents.error_handling_agents.naming_validator_agent import naming_validator_agent
from app.agents.error_handling_agents.error_classifier_agent import error_classifier_agent
from app.agents.error_handling_agents.error_recovery_agent import error_recovery_agent
from app.agents.error_handling_agents.retry_execution_agent import retry_execution_agent
from app.agents.response.response_formatter_agent import response_formatter_agent
from app.agents.routing.state_router_agent import state_router_agent
from app.agents.ddl.CreateTableAgents.schema_visualization_agent import schema_visualization_agent
from app.agents.ddl.CreateTableAgents.schema_optimizer_agent import schema_optimizer_agent
from app.agents.ddl.CreateTableAgents.user_approval_agent import user_approval_agent
from app.agents.ddl.CreateTableAgents.user_reprompt_agent import user_reprompt_agent
from app.agents.ddl.CreateTableAgents.approval_decision_agent import approval_decision_router
from app.agents.ddl.CreateTableAgents.schema_execution_planner_agent import schema_execution_planner_agent
from app.memory.memory_manager import memory_manager
from app.graph.router import (
    router_validation,
    router_scope,
    router_intent_category,
    router_ddl_operation,
    router_error_classifier,
    router_approval_decision,
    router_modify_schema_approval
)

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
    workflow.add_node("intent_router", intent_router_agent)
    workflow.add_node("ddl_router", ddl_router_agent)
    workflow.add_node("dml_router", dml_router_agent)
    workflow.add_node("requirement", requirement_agent)
    workflow.add_node("planning", planning_agent)
    workflow.add_node("schema_designer", schema_designer_agent)
    workflow.add_node("schema_context_loader", schema_context_loader_agent)
    workflow.add_node("modify_schema_intent", modify_schema_intent_agent)
    workflow.add_node("modify_schema_planner", modify_schema_planner_agent)
    workflow.add_node("modify_schema_designer", modify_schema_designer_agent)
    workflow.add_node("modify_schema_visualization", modify_schema_visualization_agent)
    workflow.add_node("modify_schema_user_approval", modify_schema_user_approval_agent)
    workflow.add_node("modify_schema_approval_decision", modify_schema_approval_decision_agent)
    workflow.add_node("modify_schema_user_reprompt", modify_schema_user_reprompt_agent)
    workflow.add_node("query_builder", query_builder_agent)
    workflow.add_node("execution", execution_agent)
    workflow.add_node("execution_monitor", execution_monitor_agent)
    workflow.add_node("log_analyzer", log_analyzer_agent)
    workflow.add_node("schema_validator", schema_validator_agent)
    workflow.add_node("dependency_checker", dependency_checker_agent)
    workflow.add_node("naming_validator", naming_validator_agent)
    workflow.add_node("error_classifier", error_classifier_agent)
    workflow.add_node("error_recovery", error_recovery_agent)
    workflow.add_node("retry_execution", retry_execution_agent)
    workflow.add_node("formatter", response_formatter_agent)
    workflow.add_node("state_router", state_router_agent)
    workflow.add_node("schema_optimizer", schema_optimizer_agent)
    workflow.add_node("schema_visualization", schema_visualization_agent)
    workflow.add_node("user_approval", user_approval_agent)
    workflow.add_node("user_reprompt", user_reprompt_agent)
    workflow.add_node("approval_decision", approval_decision_router)
    workflow.add_node("schema_execution_planner", schema_execution_planner_agent)
    
    # Entrance
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "memory")

    # MemoryManager → validation
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
            "intent_router": "intent_router"
        }
    )
    
    # Database Operations Phase
    
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
    # DDL Specialized Sub-flow
    workflow.add_edge("ddl_router", "schema_context_loader")
    
    workflow.add_conditional_edges(
        "schema_context_loader",
        router_ddl_operation,
        {
            "requirement":  "requirement",
            "modify_schema_intent": "modify_schema_intent"
        }
    )

    # DDL Create Path: Requirement -> Planning -> Designer -> Visualization -> Approval
    workflow.add_edge("requirement", "planning")
    workflow.add_edge("planning", "schema_designer")
    workflow.add_edge("schema_designer", "schema_optimizer")
    workflow.add_edge("schema_optimizer", "schema_visualization")
    workflow.add_edge("schema_visualization", "user_approval")
    
    # After HIB Interrupt resumes:
    workflow.add_edge("user_approval", "approval_decision")
    
    # Approval Decision Router Branches (Create path):
    workflow.add_conditional_edges(
        "approval_decision",
        router_approval_decision,
        {
            "schema_execution_planner": "schema_execution_planner",
            "planning": "planning",
            "user_reprompt": "user_reprompt"
        }
    )

    workflow.add_edge("user_reprompt", "user_approval")

    workflow.add_edge("schema_execution_planner", "query_builder")

    # Modify Schema Path
    workflow.add_edge("modify_schema_intent", "modify_schema_planner")
    workflow.add_edge("modify_schema_planner", "modify_schema_designer")
    workflow.add_edge("modify_schema_designer", "modify_schema_visualization")
    workflow.add_edge("modify_schema_visualization", "modify_schema_user_approval")
    workflow.add_edge("modify_schema_user_approval", "modify_schema_approval_decision")

    workflow.add_conditional_edges(
        "modify_schema_approval_decision",
        router_modify_schema_approval,
        {
            "query_builder": "query_builder",
            "modify_schema_planner": "modify_schema_planner",
            "modify_schema_user_reprompt": "modify_schema_user_reprompt"
        }
    )
    workflow.add_edge("modify_schema_user_reprompt", "modify_schema_user_approval")
    
    workflow.add_edge("query_builder", "execution")
    workflow.add_edge("execution", "formatter")

    # Execution Monitoring (Analysis First Pattern)
    workflow.add_edge("execution_monitor", "log_analyzer")
    workflow.add_edge("execution_monitor", "schema_validator")
    workflow.add_edge("execution_monitor", "dependency_checker")
    workflow.add_edge("execution_monitor", "naming_validator")

    # Convergence from parallel analysis
    workflow.add_edge("log_analyzer", "error_classifier")
    workflow.add_edge("schema_validator", "error_classifier")
    workflow.add_edge("dependency_checker", "error_classifier")
    workflow.add_edge("naming_validator", "error_classifier")

    # Classification & Healing Routing
    workflow.add_conditional_edges(
        "error_classifier",
        router_error_classifier,
        {
            "error_recovery": "error_recovery",
            "formatter": "formatter"
        }
    )

    # Sequential recovery loop
    workflow.add_edge("error_classifier", "error_recovery")
    workflow.add_edge("error_recovery", "retry_execution")
    workflow.add_edge("retry_execution", "query_builder")
        
    # Non-database Convergence
    workflow.add_edge("conversation", "formatter")
    workflow.add_edge("general_qa", "formatter")
    
    # Final Exit
    workflow.add_edge("formatter", END)
    
    return workflow.compile(checkpointer=MemorySaver())
