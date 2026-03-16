from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def approval_decision_agent(state: AgentState) -> AgentState:
    """
    ApprovalDecisionAgent (Modify Schema Flow): Categorizes user response after 
    the human-in-the-loop schema modification preview.
    Determines if the user APPROVED, requested to MODIFY, or provided INVALID input.
    """
    print("\n----- ENTERING ModifySchemaApprovalDecisionAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])
    schema_design = state.get("modify_schema_design", {})
    
    if not user_input.strip():
        print("[ModifySchemaApprovalDecisionAgent] Empty input. Status -> INVALID")
        state["approval_status"] = "INVALID"
        return state

    system_prompt = """
    You are a strict Intent Classifier.
    Your task is to classify a user's response to a schema modification preview.
    
    You must output exactly ONE word:
    APPROVE
    MODIFY
    INVALID

    --------------------------------------------------
    RULES:
    --------------------------------------------------
    1. Output APPROVE if the user accepts the proposed schema modifications without any changes.
       (e.g., "yes", "approve", "looks good", "proceed", "go ahead", "apply changes")
       
    2. Output MODIFY if the user requests changes, additions, or deductions to the schema plan.
       (e.g., "add another column", "no wait, change the type to string", "also update the email field", "cancel the deletion")
       
    3. Output INVALID if the input is purely conversational, off-topic, or completely ambiguous and un-actionable.
       (e.g., "how are you", "what time is it", "ok wait", "dfjkldsf")
       
    No explanations. Just the single word.
    """

    context_message = f"""
    PROPOSED DESIGN THEY ARE RESPONDING TO:
    {schema_design.get("operations", "[]")}
    
    USER'S RESPONSE:
    "{user_input}"
    """

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_message)
        ])
        
        status = response.content.strip().upper()
        
        # Enforce enum values
        if status not in ["APPROVE", "MODIFY", "INVALID"]:
            print(f"[ModifySchemaApprovalDecisionAgent] Warning: LLM output '{status}'. Coercing to INVALID.")
            status = "INVALID"
            
        state["approval_status"] = status
        print(f"[ModifySchemaApprovalDecisionAgent] Decision: {status}")
        
    except Exception as e:
        print(f"[ModifySchemaApprovalDecisionAgent] Unexpected error: {e}")
        state["approval_status"] = "INVALID"

    return state
