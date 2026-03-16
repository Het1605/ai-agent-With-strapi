from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def user_reprompt_agent(state: AgentState) -> AgentState:
    """
    UserRepromptAgent: AI reasoning for invalid feedback.
    Sets state["response"] then proceeds to the interrupt node.
    """
    print("----- ENTERING UserRepromptAgent (AI Reasoning) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    system_prompt = (
        "You are a helpful database architect assistant.\n"
        "The user just sent an unrelated or unclear message instead of approving/modifying the schema.\n"
        "Your task is to politely guide them back to the decision.\n\n"
        "GUIDELINES:\n"
        "- Be very brief and polite.\n"
        "- Explain that you need a decision on the proposed schema to proceed.\n"
        "- Give clear examples of how they can respond (e.g., 'approve', 'add field X', 'remove table Y').\n"
        "- Do NOT repeat the entire schema design here, just the call to action."
    )
    
    context = (
        f"Original Design Preview:\n{state.get('interaction_message')}\n\n"
        f"User's invalid response: {state.get('user_input')}"
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=context)
    ])
    
    # Store the reprompt message in state["response"]
    state["response"] = response.content.strip()
    
    return state
