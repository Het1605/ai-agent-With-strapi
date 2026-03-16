import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def requirement_agent(state: AgentState) -> AgentState:
    """
    RequirementAgent: Structured requirement extraction.
    Analyzes user intent and domain context.
    """
    print("\n----- ENTERING RequirementAgent (AI Planning) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])[-3:]
    
    system_prompt = (
        "You are a Database Requirements Analyst.\n"
        "Your goal is to parse the user's request into structured requirements.\n\n"
        "GUIDELINES:\n"
        "- intent_type: full_database_design, multi_table_design, single_table_design, or architecture_suggestion.\n"
        "- system_domain: The business area (e.g., ecommerce, school, hospital, generic_database).\n"
        "- interaction_mode: suggest_and_design (if user is vague), design_only (if user is specific), or explanation_only.\n"
        "- execution_required: true (since this is a DDL workflow).\n\n"
        "OUTPUT REQUIREMENT:\n"
        "- Respond ONLY with a valid JSON object."
    )
    
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    human_msg = f"History:\n{history_str}\n\nRequest: {user_input}"
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    print("requirnment:",response)
    
    try:
        reqs = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        state["requirements"] = reqs
        print(f"[RequirementAgent] Extracted domain: {reqs.get('system_domain')}")
    except Exception as e:
        print(f"[RequirementAgent] Error parsing JSON: {e}")
        state["requirements"] = {
            "intent_type": "single_table_design",
            "system_domain": "generic_database",
            "interaction_mode": "suggest_and_design",
            "execution_required": True
        }
    
    return state
