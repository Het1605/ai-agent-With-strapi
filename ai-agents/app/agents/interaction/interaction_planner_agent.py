from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def interaction_planner_agent(state: AgentState) -> AgentState:
    """
    InteractionPlannerAgent: AI agent that manages the user interaction loop 
    when required schema information is missing.
    """
    print("\n----- ENTERING InteractionPlannerAgent (AI Reasoning) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    missing_fields = state.get("missing_fields", [])
    schema_data = state.get("schema_data", {})
    user_query = state.get("user_query", "")
    
    system_prompt = (
        "You are an Interaction Design Specialist for an AI Database System. "
        "Your role is to analyze missing schema requirements and generate a friendly, "
        "helpful request for the user to provide that information.\n\n"
        "Guidelines:\n"
        "- Be specific about what is missing (e.g., column names, data types).\n"
        "- Explain why this information is needed if appropriate.\n"
        "- Provide examples to help the user understand the requested format.\n"
        "- DO NOT use jargon; keep it natural.\n\n"
        "Output a JSON object with two fields:\n"
        "1. 'interaction_request': Meta-data about the missing fields.\n"
        "2. 'interaction_message': The actual message to show to the user."
    )
    
    human_msg = (
        f"Missing Fields: {missing_fields}\n"
        f"Current Schema: {json.dumps(schema_data)}\n"
        f"User Original Query: {user_query}"
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])
    
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
        
        state["interaction_request"] = result.get("interaction_request", {})
        state["interaction_message"] = result.get("interaction_message", "I need a bit more information to proceed.")
        
        print(f"InteractionPlannerAgent: Generated request message -> {state['interaction_message']}")
        
        # Also store in analysis for ResponseFormatter
        state["analysis"] = (state.get("analysis") or "") + f"\nInteraction Planner: {state['interaction_message']}"
        
    except Exception as e:
        print(f"Error in InteractionPlannerAgent reasoning: {e}")
        state["interaction_message"] = "I'm having trouble understanding what's missing. Could you please provide more details about the table you want to create?"
        state["interaction_request"] = {"error": str(e)}
        
    return state
