from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def approval_decision_router(state: AgentState) -> AgentState:
    """
    ApprovalDecisionRouter: Uses LLM reasoning to classify user intent.
    Routes to execution, redesign, or reprompt.
    """
    print("\n----- ENTERING ApprovalDecisionRouter (Semantic Classification) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input = state.get("user_input", "")
    
    if not user_input:
        print("[ApprovalDecisionRouter] No user input found. Defaulting to INVALID.")
        state["approval_status"] = "INVALID"
        return state

    system_prompt = (
        "You are an Intent Classification Engine.\n"
        "Your goal is to semantically analyze user feedback regarding a database schema proposal.\n\n"
        "CATEGORIES:\n"
        "1. APPROVE: User agrees to proceed with the design. (e.g., 'yes', 'do it', 'approve', 'looks good', 'create it')\n"
        "2. MODIFY: User wants to change something or provide more details. (e.g., 'add phone field', 'remove salary', 'change name to department', 'Spanish translation')\n"
        "3. INVALID: The response is unrelated, gibberish, or a generic greeting. (e.g., 'hello', 'ping', 'tell me a story', 'what time is it')\n\n"
        "REASONING GUIDELINES:\n"
        "- Be multilingual: 'Haan' (Hindi) or 'Si' (Spanish) should be APPROVE.\n"
        "- Be tolerant: 'make name required' should be MODIFY.\n\n"
        "OUTPUT REQUIREMENT:\n"
        "- Respond ONLY with the category name: APPROVE, MODIFY, or INVALID."
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Feedback: {user_input}")
    ])

    decision = response.content.strip().upper()
    if decision not in ["APPROVE", "MODIFY", "INVALID"]:
        decision = "INVALID"

    print(f"[ApprovalDecisionRouter] Classification Decision: {decision}")
    state["approval_status"] = decision
    
    return state
