import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def planning_agent(state: AgentState) -> AgentState:
    """
    PlanningAgent: High-level architectural planning.
    Determines core and optional modules for the database.
    """
    print("\n----- ENTERING PlanningAgent (Architecture) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    reqs = state.get("requirements", {})
    
    system_prompt = (
        "You are a Database Architect.\n"
        "Based on the system requirements, suggest the high-level modules for the database.\n\n"
        "FORMAT:\n"
        "{\n"
        "  \"system_domain\": \"...\",\n"
        "  \"core_modules\": [\"module1\", \"module2\"...],\n"
        "  \"optional_modules\": [\"moduleA\", \"moduleB\"...]\n"
        "}\n\n"
        "Respond ONLY with valid JSON."
    )
    
    human_msg = f"Requirements: {json.dumps(reqs)}"
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])
    
    try:
        plan = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        state["architecture_plan"] = plan
        print(f"[PlanningAgent] Modules suggested: {len(plan.get('core_modules', []))} core, {len(plan.get('optional_modules', []))} optional.")
    except Exception as e:
        print(f"[PlanningAgent] Error: {e}")
        state["architecture_plan"] = {
            "system_domain": reqs.get("system_domain", "generic"),
            "core_modules": ["data_records"],
            "optional_modules": []
        }
        
    return state
