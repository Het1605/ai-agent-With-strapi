from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def dml_router_agent(state: AgentState) -> AgentState:
    """
    DMLRouterAgent: AI agent that determines the exact DML operation using LLM reasoning.
    """
    print("\n----- ENTERING DMLRouterAgent -----")
    
    task_queue = state.get("task_queue", [])
    current_index = state.get("current_task_index", 0)
    current_task = task_queue[current_index] if current_index < len(task_queue) else {}
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_prompt = (
        "You are a DML Specialist Agent. "
        "Analyze the provided task and determine the specific DML operation type.\n"
        "Possible types: DML_SELECT, DML_INSERT, DML_UPDATE, DML_DELETE.\n\n"
        "Respond ONLY with the operation type constant."
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Task Context: {current_task}")
    ])
    
    operation = response.content.strip()
    state["dml_operation"] = operation
    
    print(f"DMLRouterAgent: Determined operation -> {operation}")
    
    # Store reasoning
    state["analysis"] = (state.get("analysis") or "") + f"\nDML Router: Resolved operation as {operation}."
    
    return state
