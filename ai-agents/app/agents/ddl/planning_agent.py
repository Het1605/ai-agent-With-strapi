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
    field_registry = state.get("field_registry", {})
    history = state.get("conversation_history", [])
    previous_plan = state.get("architecture_plan", {})
    user_modification = state.get("user_input", "") # The latest feedback
    
    system_prompt = (
        "You are a world-class Database Architect.\n\n"

        "Your job is to design or REVISE the conceptual architecture of a database system.\n"
        "You must think like an enterprise system designer.\n\n"

        "CONTEXT HANDLING:\n"
        "- If it is a new design: Identify core modules and entities from scratch.\n"
        "- If it is a revision: Read the 'Previous Architecture Plan' and 'User Modification Request'.\n"
        "- If the user provides their own architecture or specific entities, respect and refine them.\n"
        "- You can extend, merge, or remove modules and entities based on feedback.\n\n"

        "Based on the requirements and history, identify:\n"
        "1. Core business modules\n"
        "2. Core entities inside each module\n"
        "3. Optional modules that enhance the system\n\n"

        "IMPORTANT DESIGN PRINCIPLES:\n"
        "- Think like a real production system.\n"
        "- Each module should contain multiple related entities.\n"
        "- Avoid overly simplistic systems.\n"
        "- Include supporting entities such as logs, reviews, transactions, or usage tables where appropriate.\n"
        "- Systems should typically contain 10–30 tables depending on complexity.\n\n"

        "OUTPUT FORMAT (JSON ONLY):\n"
        "{\n"
        "  \"system_domain\": \"...\",\n"
        "  \"modules\": [\n"
        "    {\n"
        "      \"name\": \"module_name\",\n"
        "      \"entities\": [\"entity1\", \"entity2\", \"entity3\"]\n"
        "    }\n"
        "  ],\n"
        "  \"optional_modules\": [\n"
        "    {\n"
        "      \"name\": \"optional_module\",\n"
        "      \"entities\": [\"entityA\", \"entityB\"]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "Respond ONLY with valid JSON."
    )
    
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    human_msg = (
        f"Conversation History:\n{history_str}\n\n"
        f"Requirements: {json.dumps(reqs)}\n"
        f"Previous Architecture Plan: {json.dumps(previous_plan)}\n"
        f"User Modification Request: {user_modification}\n"
        f"Available Field Registry: {json.dumps(field_registry)}"
    )
    
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
