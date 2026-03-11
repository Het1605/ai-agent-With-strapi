from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json


async def input_validation_agent(state: AgentState) -> AgentState:
    """
    InputValidationAgent: context-aware, spelling-tolerant input validation.

    Passes conversation history to the LLM so that short follow-up answers
    (e.g. 'product', 'decimal', 'yes') are recognised as valid responses
    rather than invalid standalone tokens.
    """
    print("----- ENTERING InputValidationAgent -----")

    llm        = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input = state.get("user_input", "")

    # Build a readable conversation history block (last 6 turns)
    history     = state.get("conversation_history") or []
    history_str = "\n".join(
        f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
        for m in history[-6:]
    ) or "(no previous conversation)"

    system_prompt = (
        "You are an Input Validation Agent for an AI Database Assistant.\n\n"
        "Your job is to determine whether the user's message should be processed.\n\n"
        "You will receive:\n"
        "  1. Conversation History — use this to understand if the user is answering a previous question.\n"
        "  2. Current User Input   — the message to validate.\n\n"
        "Output ONLY a JSON object with exactly this structure:\n"
        '{"is_valid": boolean, "reasoning": "technical explanation", "suggested_error_message": "user-friendly explanation"}\n\n'
        "VALIDATION RULES\n"
        "1. Be EXTREMELY tolerant of normal user behaviour.\n"
        "2. ACCEPT spelling mistakes: 'creat table', 'add colum', 'updte feild' → valid.\n"
        "3. ACCEPT short follow-up answers when the user is replying to a previous question:\n"
        "   'product', 'employee', 'yes', 'no', 'decimal', 'price', 'age' → always valid.\n"
        "4. ACCEPT schema fragments: 'name age price', 'salary integer', 'price decimal' → valid.\n"
        "5. ACCEPT greetings: 'hello', 'hi', 'good morning' → valid.\n"
        "6. ACCEPT partial database instructions:\n"
        "   'create table student', 'add column price', 'rename field email' → valid.\n"
        "7. REJECT ONLY inputs that are clearly meaningless:\n"
        "   - empty string\n"
        "   - random keyboard spam: 'asdfasdfasdf'\n"
        "   - corrupted/binary text: '#@!$#!$#!$'\n\n"
        "CRITICAL: NEVER reject input because it is short, misspelled, a schema fragment, or a follow-up answer.\n\n"
        "Example VALID response:   {\"is_valid\": true,  \"reasoning\": \"Valid follow-up answer.\", \"suggested_error_message\": \"\"}\n"
        "Example INVALID response: {\"is_valid\": false, \"reasoning\": \"Meaningless spam.\", \"suggested_error_message\": \"Sorry, I couldn't understand that. Please rephrase.\"}"
    )

    human_msg = (
        f"[Conversation History]\n{history_str}\n\n"
        f"[Current Input]\n{user_input}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    try:
        clean  = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

        state["analysis"] = f"Validation Result: {result.get('reasoning')}"
        state["validation_results"]["input_validation"] = {
            "is_valid":         result.get("is_valid", True),
            "suggested_message": result.get("suggested_error_message", "")
        }

        print(f"Validation decision: {'Valid' if result.get('is_valid') else 'Invalid'}")
        print(f"Reasoning: {result.get('reasoning')}")

    except Exception as e:
        print(f"Error in InputValidationAgent: {e} — defaulting to valid.")
        state["validation_results"]["input_validation"] = {"is_valid": True}
        state["analysis"] = "Validation system error, defaulting to valid."

    return state
