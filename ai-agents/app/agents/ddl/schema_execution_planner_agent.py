from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def schema_execution_planner_agent(state: AgentState) -> AgentState:
    """
    SchemaExecutionPlannerAgent: Prepares the finalized execution order.
    Uses LLM reasoning for topological sorting.
    """
    print("\n----- ENTERING SchemaExecutionPlannerAgent (AI Planning) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    schema_plan = state.get("schema_plan")
    
    if not schema_plan or not schema_plan.get("tables"):
        print("[SchemaExecutionPlannerAgent] No tables found in plan.")
        return state

    system_prompt = (
        "You are a Database Execution Planner.\n"
        "Your goal is to prepare an ordered execution plan for creating Strapi collections.\n\n"
        "RESPONSIBILITIES:\n"
        "1. Dependency Analysis: Identify which tables reference each other via 'relation' fields.\n"
        "2. Topological Sorting: Determine the correct creation order. Parent tables (relation targets) MUST be created before dependent tables.\n"
        "3. Payload Preparation: Output the final list of table objects in the exact order they should be executed.\n\n"
        "REASONING:\n"
        "- If table 'enrollment' has a relation field targeting 'student', then 'student' MUST come first.\n"
        "- Strapi UIDs follow 'api::[singular].[singular]' format.\n\n"
        "JSON OUTPUT REQUIREMENTS:\n"
        "- Respond ONLY with the sorted JSON array of table objects.\n"
        "- No markdown, no headers, just the array: [{\"table_name\": \"...\", \"columns\": [...]}, ...]"
    )

    human_msg = f"Approved Schema Plan JSON:\n{json.dumps(schema_plan, indent=2)}"

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        sorted_tables = json.loads(clean)
        
        # In current system, QueryBuilder expects schema_data["tables"]
        state["schema_data"] = {"tables": sorted_tables}
        state["schema_ready"] = True
        
        print(f"[SchemaExecutionPlannerAgent] Created execution plan for {len(sorted_tables)} tables.")
        for i, t in enumerate(sorted_tables, 1):
            print(f" {i}. {t['table_name']}")
            
    except Exception as e:
        print(f"[SchemaExecutionPlannerAgent] Plan parsing error: {e}. Falling back to unsorted plan.")
        state["schema_data"] = schema_plan
        state["schema_ready"] = True

    return state
