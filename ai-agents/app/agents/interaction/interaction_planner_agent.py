from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def interaction_planner_agent(state: AgentState) -> AgentState:
    """
    InteractionPlannerAgent — Generic, dynamic interaction handler.

    Called whenever any upstream agent detects missing information.
    Reads state["active_agent"] and state["missing_fields"] to generate
    a targeted question, then sets the interaction state so StateRouterAgent
    can resume the correct agent on the next user turn.
    """
    print("\n----- ENTERING InteractionPlannerAgent -----")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # ── Read context from state ────────────────────────────────────────
    active_agent   = state.get("active_agent")
    missing_fields = state.get("missing_fields") or []
    schema_data    = state.get("schema_data") or {}
    user_input     = state.get("user_input") or ""
    execution_error = state.get("execution_error")

    print("active_agent",active_agent)


    state["interaction_attempts"] = state.get("interaction_attempts", 0) + 1

    # Safety valve — prevent infinite loops
    if state["interaction_attempts"] > 5:
        message = (
            "I'm having trouble collecting the required information. "
            "Could you please start over with a clearer request, for example: "
            "'create collection orders with name and price'?"
        )
        state["response"]          = message
        state["interaction_phase"] = True
        state["active_agent"]      = active_agent
        print(f"[InteractionPlannerAgent] Max attempts reached — asking user to restart.")
        return state

    # ── Build prompt context ───────────────────────────────────────────
    if execution_error:
        context_detail = (
            f"An execution error occurred: {execution_error}\n"
            "Ask the user how they would like to resolve this."
        )
        # Clear the error so it is not re-processed next turn
        state["execution_error"] = None
    else:
        context_detail = (
            f"Agent: {active_agent}\n"
            f"Missing fields: {json.dumps(missing_fields)}\n"
            f"Current schema / collected data: {json.dumps(schema_data)}\n"
            f"Last user message: {user_input!r}"
        )

    system_prompt = (
        "You are the InteractionPlannerAgent in a multi-agent database management system.\n\n"
        "An upstream agent attempted to complete a task but could not because some required "
        "information is still missing from the user's request.\n\n"
        "Your job:\n"
        "1. Understand which agent needs more data and what is missing.\n"
        "2. Generate ONE clear, friendly question to collect the first missing piece of information.\n"
        "3. Do NOT ask for multiple things at once.\n"
        "4. Do NOT modify or invent any schema fields.\n\n"
        "Return ONLY a valid JSON object with this exact structure:\n"
        "{\n"
        '  "interaction": true,\n'
        '  "next_agent": "<agent_that_should_continue>",\n'
        '  "missing_fields": ["field1", "field2"],\n'
        '  "message": "<question to show the user>"\n'
        "}\n"
        "Output ONLY valid JSON. No extra text."
    )

    human_prompt = (
        f"Context:\n{context_detail}\n\n"
        "Generate the interaction JSON now."
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    # ── Parse LLM response ─────────────────────────────────────────────
    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

        next_agent     = result.get("next_agent", active_agent)
        missing_out    = result.get("missing_fields", missing_fields)
        message        = result.get("message", "Could you provide more information?")

    except Exception as e:
        print(f"[InteractionPlannerAgent] JSON parse error: {e} — using fallback question.")
        next_agent  = active_agent
        missing_out = missing_fields
        message     = (
            f"I need a bit more information to continue. "
            f"Could you please provide: {', '.join(missing_fields)}?"
        )

    # ── Update state so StateRouterAgent routes correctly next turn ────
    state["response"]          = message
    state["interaction_phase"] = True
    state["active_agent"]      = next_agent
    state["missing_fields"]    = missing_out

    print(f"[InteractionPlannerAgent] next_agent    : {next_agent}")
    print(f"[InteractionPlannerAgent] missing_fields: {missing_out}")
    print(f"[InteractionPlannerAgent] message       : {message}")

    return state
