from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def ddl_router_agent(state: AgentState) -> AgentState:
    """
    DDLRouterAgent: AI agent that determines the exact DDL operation using LLM reasoning.
    """
    print("\n----- ENTERING DDLRouterAgent -----")
    
    task_queue = state.get("task_queue", [])
    current_index = state.get("current_task_index", 0)
    current_task = task_queue[current_index] if current_index < len(task_queue) else {}
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_prompt = (
        "You are a DDL Specialist Agent. "
        "Analyze the provided task and determine the specific DDL operation type.\n"
        "Possible types: DDL_CREATE_TABLE, DDL_MODIFY_SCHEMA, DDL_DELETE_TABLE.\n\n"
        "Respond ONLY with the operation type constant."
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Task Context: {current_task}")
    ])
    
    operation = response.content.strip()
    state["ddl_operation"] = operation
    
    print(f"DDLRouterAgent: Determined operation -> {operation}")
    
    # Store reasoning
    state["analysis"] = (state.get("analysis") or "") + f"\nDDL Router: Resolved operation as {operation}."
    
    return state
