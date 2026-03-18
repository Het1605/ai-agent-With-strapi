import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def approval_decision_router(state: AgentState) -> AgentState:
    """
    ApprovalDecisionRouter: Context-aware semantic intent classifier.
    Uses conversation history + latest user input to classify intent.
    """
    print("\n----- ENTERING ApprovalDecisionRouter (Semantic Classification) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input = state.get("user_input", "")
    conversation_history = state.get("conversation_history", [])
    
    if not user_input:
        print("[ApprovalDecisionRouter] No user input found. Defaulting to INVALID.")
        state["approval_status"] = "INVALID"
        return state

    # Build recent history context (last 10 messages)
    recent_history = conversation_history[-10:] if conversation_history else []
    history_str = ""
    if recent_history:
        history_lines = []
        for msg in recent_history:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            # Truncate long system responses for efficiency
            if len(content) > 500:
                content = content[:500] + "..."
            history_lines.append(f"{role}: {content}")
        history_str = "\n".join(history_lines)

    system_prompt = """You are a Production-Grade Intent Classification Engine.

                    Your ONLY job: classify the user's LATEST message into one of THREE categories based on the FULL conversation context.

                    --------------------------------------------------
                    CATEGORIES
                    --------------------------------------------------

                    1. APPROVE
                    The user wants to proceed with the current schema/design as-is.
                    
                    Direct signals: "yes", "ok", "do it", "approve", "create it", "proceed", "go ahead", "confirmed"
                    Indirect signals: "looks fine", "seems good", "perfect", "nice work", "that works", "I like it"
                    Multilingual: "haan", "ok hai", "si", "oui", "ja", "theek hai", "sahi hai", "ban jao"

                    2. MODIFY
                    The user wants ANY change, improvement, or has ANY dissatisfaction — even vague.
                    
                    Direct changes: "add email field", "remove salary", "change the name", "make it required"
                    Vague requests: "make it better", "optimize this", "something is missing", "not complete"
                    Complaints: "this is wrong", "I don't like this", "needs improvement"
                    Additions: "also add...", "what about...", "can you include..."
                    Structural: "change the relations", "split this table", "merge these"

                    3. INVALID
                    ONLY when the message is completely unrelated to ANY database/schema context.
                    
                    Only use for: pure greetings with NO prior context, random gibberish, completely off-topic questions

                    --------------------------------------------------
                    CRITICAL DECISION RULES
                    --------------------------------------------------

                    RULE 1: If the conversation is ALREADY about schema design (check history), then ANY follow-up is either APPROVE or MODIFY. NEVER INVALID.

                    RULE 2: If unsure between MODIFY vs INVALID → choose MODIFY.
                    RULE 3: If unsure between APPROVE vs MODIFY → choose MODIFY (safer).
                    RULE 4: INVALID should be EXTREMELY RARE. Only for truly unrelated input with NO schema context.

                    --------------------------------------------------
                    OUTPUT
                    --------------------------------------------------

                    Respond with ONLY one word: APPROVE, MODIFY, or INVALID

"""

    # Build the human message with full context
    context_parts = []
    if history_str:
        context_parts.append(f"CONVERSATION HISTORY (recent):\n{history_str}")
    context_parts.append(f"\nLATEST USER INPUT:\n{user_input}")

    human_content = "\n".join(context_parts)

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ])

    decision = response.content.strip().upper()
    
    # Safety: extract just the classification word if LLM adds extra text
    for keyword in ["APPROVE", "MODIFY", "INVALID"]:
        if keyword in decision:
            decision = keyword
            break
    
    if decision not in ["APPROVE", "MODIFY", "INVALID"]:
        # Default to MODIFY if uncertain (safer than INVALID)
        decision = "MODIFY"

    print(f"[ApprovalDecisionRouter] Classification Decision: {decision}")
    state["approval_status"] = decision
    
    return state
