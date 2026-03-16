from app.graph.state import AgentState

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
        return "intent_router"
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

def router_ddl_operation(state: AgentState):
    """
    Routes from DDLRouterAgent to the correct DDL agent based on ddl_operation.
    """
    op = state.get("ddl_operation", "DDL_CREATE_TABLE")
    if op == "DDL_MODIFY_SCHEMA":
        return "modify_schema_intent"
    return "requirement"

def router_error_classifier(state: AgentState):
    """
    Routes from ErrorClassifierAgent based on diagnostic findings.
    If execution_error exists, routes to recovery. Otherwise, proceeds to formatter.
    """
    if state.get("execution_error"):
        return "error_recovery"
    return "formatter"

def router_approval_decision(state: AgentState):
    """
    Approval Decision Router Branches (Create path).
    """
    status = state.get("approval_status", "INVALID")
    if status == "APPROVE":
        return "schema_execution_planner"
    elif status == "MODIFY":
        return "planning"
    else:
        return "user_reprompt" # Use the reprompter to set response then pause

def router_modify_schema_approval(state: AgentState):
    """
    Approval Decision Router Branches (Modify path).
    """
    status = state.get("approval_status", "INVALID")
    if status == "APPROVE":
        return "query_builder"
    elif status == "MODIFY":
        return "modify_schema_planner"
    else:
        return "modify_schema_user_reprompt"
