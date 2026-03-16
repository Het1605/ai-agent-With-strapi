from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def user_reprompt_agent(state: AgentState) -> AgentState:
    """
    UserRepromptAgent (Modify Schema Flow): Handles cases where the user's 
    approval input was marked INVALID. Politely asks them to clarify if they 
    approve the modification or need further adjustments.
    """
    print("\n----- ENTERING ModifySchemaUserRepromptAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    user_input = state.get("user_input", "")
    schema_design = state.get("modify_schema_design", {})
    
    system_prompt = """
    You are a friendly AI Database Assistant.
    You previously presented a schema modification plan to the user, but their response was unclear, irrelevant, or invalid.
    
    Your task is to politely explain that you didn't understand, and ask them to either:
    1. Approve the changes.
    2. Provide feedback/instructions to modify the changes.

    --------------------------------------------------
    RULES:
    --------------------------------------------------
    - Be concise and polite.
    - DO NOT alter the schema plan.
    - DO NOT generate new modification explanations.
    - DO NOT use placeholders like [table name].

    Example: "I couldn't clearly understand your response. If you're satisfied with the proposed schema updates, you can approve them and I will proceed with applying the changes. If you'd like to adjust anything in the modification plan, just describe the changes and I'll update the design."
    """

    context_message = f"""
    PREVIOUS PROPOSAL WE ARE WAITING ON:
    {schema_design.get("operations", "[]")}
    
    THEIR INVALID RESPONSE:
    "{user_input}"
    """

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_message)
        ])
        
        state["response"] = response.content.strip()
        state["interaction_message"] = state["response"]
        print("[ModifySchemaUserRepromptAgent] successfully drafted reprompt message.")
        
    except Exception as e:
        print(f"[ModifySchemaUserRepromptAgent] Unexpected error: {e}")
        fallback_msg = "I didn't quite catch that. Would you like me to proceed with modifying the schema, or would you like to make changes to the plan?"
        state["response"] = fallback_msg
        state["interaction_message"] = fallback_msg

    return state
