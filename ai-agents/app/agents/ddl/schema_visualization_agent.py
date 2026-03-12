from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def schema_visualization_agent(state: AgentState) -> AgentState:
    """
    SchemaVisualizationAgent: Uses LLM reasoning to generate a pretty markdown preview.
    """
    print("\n----- ENTERING SchemaVisualizationAgent (Preview Reasoning) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    schema_plan = state.get("schema_plan")
    
    if not schema_plan or not schema_plan.get("tables"):
        state["interaction_message"] = "No new tables were designed. Is there anything else you'd like to do?"
        return state

    system_prompt = (
        "You are a Technical UX Writer and Database Designer.\n"
        "Your task is to convert a JSON database schema plan into a beautiful, human-readable markdown preview.\n\n"
        "DESIGN PRINCIPLES:\n"
        "1. Clarity: Explain what each table and field is for.\n"
        "2. Visual Appeal: Use emojis (🏗️, 📋, 🔗), bullet points, and headers.\n"
        "3. Engagement: Ask the user clearly if they want to proceed or modify the design.\n"
        "4. Multilingual: If the user request was in a specific language (Spanish, Hindi, etc.), use that language for the explanation.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- Respond ONLY with the markdown content.\n"
        "- Do not include markdown code blocks around your entire response.\n"
        "- Always include a call-to-action at the end asking for 'APPROVE' or 'MODIFY'."
    )

    human_msg = (
        f"User Original Request: {state.get('user_input_history', [{}])[0].get('content', 'N/A')}\n"
        f"Schema Plan JSON:\n{json.dumps(schema_plan, indent=2)}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    state["interaction_message"] = response.content.strip()
    # Ensure the preview is visible to the user by committing it to 'response'
    state["response"] = state["interaction_message"]
    
    return state
