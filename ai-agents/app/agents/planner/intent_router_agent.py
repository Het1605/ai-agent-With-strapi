from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def intent_router_agent(state: AgentState) -> AgentState:
    """
    IntentRouterAgent: AI agent that analyzes the current task and categorizes it
    into DDL or DML operations using LLM reasoning.
    """
    task_queue = state.get("task_queue", [])
    current_index = state.get("current_task_index", 0)
    
    print("\n----- ENTERING IntentRouterAgent -----")
    print(f"Current Task Index: {current_index}")
    
    if current_index >= len(task_queue):
        print("IntentRouterAgent: Task queue empty, routing to formatter.")
        state["intent_category"] = "FORMATTER" # Changed from COMPLETE to FORMATTER
        return state

    current_task = task_queue[current_index]
    task_type = current_task.get("task_type", "UNKNOWN")
    print(f"Task Type: {task_type}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_prompt = (
        "You are an Intent Analysis Agent for a Database System. "
        "Your role is to categorize a specific database task type into one of two categories:\n"
        "1. 'DDL': For Data Definition Language tasks (creating, modifying, or deleting tables/schemas).\n"
        "2. 'DML': For Data Manipulation Language tasks (selecting, inserting, updating, or deleting records).\n\n"
        "Respond ONLY with the category name: DDL or DML."
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Categorize this task type: {task_type}")
    ])
    
    category = response.content.strip().upper()
    state["intent_category"] = category
    
    print(f"IntentRouterAgent: Categorized as {category}")
    
    # Store reasoning in analysis
    state["analysis"] = (state.get("analysis") or "") + f"\nIntent Router: Task categorized as {category}."
    
    return state
