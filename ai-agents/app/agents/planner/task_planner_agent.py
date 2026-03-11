from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def task_planner_agent(state: AgentState) -> AgentState:
    """
    TaskPlannerAgent: Analyzes user queries and generates a structured queue of tasks.
    The agent uses LLM reasoning to decompose requests into executable database steps.
    """
    print("\n----- ENTERING TaskPlannerAgent (Planning Queue) -----")
    
    if state.get("interaction_phase") == True:
        print("TaskPlannerAgent: Bypassing planning during interaction phase.")
        return state
        
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])
    current_planned_task = state.get("planned_task", "")
    
    # Format history for prompt
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    
    system_prompt = (
        "You are a Database Task Planner for a College Management System. "
        "Decompose the user's request into a list of atomic executable database tasks.\n\n"
        "CONTEXT (Previous Logic):\n"
        f"Conversation History:\n{history_str}\n"
        f"Previously Planned Task: {current_planned_task}\n\n"
        "INSTRUCTIONS:\n"
        "1. If the user is providing missing details (like column types or names) for a previously discussed task, "
        "ensure the task queue reflects the CONTINUATION of that task.\n"
        "2. Allowed Task Types (ONLY these two DDL types exist):\n"
        "- DDL_CREATE_TABLE: Create a brand new collection/table from scratch.\n"
        "  Examples: 'create table orders', 'create collection product', 'make new student table'\n"
        "- DDL_MODIFY_SCHEMA: Any modification to an existing schema.\n"
        "  Examples: 'add column price', 'delete column email', 'rename column email to email_address',\n"
        "           'make field required', 'make field unique', 'set default value',\n"
        "           'delete collection product', 'rename collection orders to purchase'\n"
        "  IMPORTANT: Collection/table DELETION is DDL_MODIFY_SCHEMA, NOT a separate delete type.\n"
        "- DML_SELECT: Retrieve records.\n"
        "- DML_INSERT: Create new records.\n"
        "- DML_UPDATE: Update existing records.\n"
        "- DML_DELETE: Delete records (row-level, NOT table deletion).\n\n"
        "CRITICAL: DDL_DELETE_TABLE does NOT exist. Use DDL_MODIFY_SCHEMA for collection deletion.\n\n"
        "Respond ONLY with a valid JSON array of objects. Each object must have:\n"
        "- 'task_type': One of the allowed types above.\n"
        "- 'target': The table name or 'system'.\n"
        "- 'description': A brief explanation of what this step does.\n\n"
        "Example JSON Output:\n"
        '[\n'
        '  {"task_type": "DDL_CREATE_TABLE", "target": "students", "description": "Create the students table"}\n'
        ']'
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])

    print("Task_planner_response",response)
    
    try:
        # Clean and parse JSON response
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        task_list = json.loads(clean_content)

        print("Task List:",task_list)
        
        # Ensure it's a list
        if not isinstance(task_list, list):
            task_list = []
            
        state["task_queue"] = task_list
        state["current_task_index"] = 0

        # Safety normalization: DDL_DELETE_TABLE is not supported.
        # Any such task must be silently rewritten to DDL_MODIFY_SCHEMA.
        for task in task_list:
            if task.get("task_type") == "DDL_DELETE_TABLE":
                print(f"[TaskPlannerAgent] Rewriting DDL_DELETE_TABLE → DDL_MODIFY_SCHEMA for target '{task.get('target')}'")
                task["task_type"] = "DDL_MODIFY_SCHEMA"

        # If this is a fresh DDL_CREATE_TABLE plan, reset schema_data.
        # We are guaranteed to be outside interaction_phase here — the early-return
        # at the top of this function already handles the interaction loop bypass.
        is_create_table = any(t.get("task_type") == "DDL_CREATE_TABLE" for t in task_list)
        if is_create_table:
            state["schema_data"] = {"table_name": None, "columns": []}
            state["schema_ready"] = False
            state["missing_fields"] = []
            state["interaction_attempts"] = 0
            print("[Schema Reset] Starting new table creation task — schema_data cleared.")
        
        # Update internal analysis
        tasks_summary = ", ".join([t.get("task_type") for t in task_list])
        state["analysis"] = (state.get("analysis") or "") + f"\nTask Queue Planned: {tasks_summary}"
        state["intent"] = "database_multi_step_planning"
        
        print(f"TaskPlannerAgent: Generated {len(task_list)} tasks.")
        for i, task in enumerate(task_list):
            print(f"  [{i}] {task.get('task_type')} on '{task.get('target')}'")
            
    except Exception as e:
        print(f"Error in TaskPlannerAgent JSON parsing: {e}")
        state["task_queue"] = []
        state["current_task_index"] = 0
        state["analysis"] = f"Planning failed: {str(e)}"
        
    # IMPORTANT: We no longer set state["response"] here!
    return state
