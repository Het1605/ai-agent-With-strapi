from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def schema_visualization_agent(state: AgentState) -> AgentState:
    """
    SchemaVisualizationAgent: Explains the optimized database architecture to the user.
    Pivots from technical reporting to consultative narration.
    """
    print("\n----- ENTERING SchemaVisualizationAgent (Consultative Narration) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    # Primary Source of Truth: Optimized Plan from SchemaOptimizerAgent
    optimized_plan = state.get("optimized_full_plan", {})
    schema_plan = state.get("schema_plan", {}) # The actual optimized tables
    
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])
    optimization_notes = state.get("optimization_notes", [])
    suggestions = state.get("suggestions", [])
    
    if not schema_plan or not schema_plan.get("tables"):
        state["interaction_message"] = "I've reviewed the requirements, but I haven't been able to finalize a robust schema design yet. Could you provide a bit more detail on the entities you need?"
        state["response"] = state["interaction_message"]
        return state

    system_prompt = """
    You are a world-class Senior Software Architect and Database Consultant. Your job is to explain a finalized database design to a client in a clear, natural, and highly professional manner.

    --------------------------------------------------
    THE CHALLENGE
    --------------------------------------------------
    You are NOT a robot. You are NOT a template generator. You are a human-tier expert walking a colleague through a system design. 
    Avoid generic headers like "Tables List" or "Database Schema". Instead, use headers and narrative styles that fit the domain.

    --------------------------------------------------
    YOUR INPUT CONTEXT
    --------------------------------------------------
    You have:
    1. The Optimized Schema (Tables, columns, relations).
    2. Optimization Notes (Technical changes made by the Schema Optimizer).
    3. Design Suggestions (Future recommendations).
    4. User Request and Historical Context.

    --------------------------------------------------
    EXPLANATION GOALS
    --------------------------------------------------
    1. SUMMARY: Briefly state how the architecture supports the user's domain.
    2. NARRATION: Walk through the core entities. Explain WHY they exist and HOW they relate.
    3. HIGHLIGHTS: Mention key technical improvements made by the team (referencing 'optimization_notes').
    4. SUGGESTIONS: Politely present future-looking recommendations (referencing 'suggestions') as optional value-adds.

    --------------------------------------------------
    STYLE RULES
    --------------------------------------------------
    - NATURAL TONE: Speak like a consultant presenting to a stakeholder.
    - FLEXIBLE FORMATTING: Use indentation, bullet points, and spacing for readability.
    - NO ROBOTICS: Avoid saying things like "The following JSON has been processed". 
    - CONTEXT-AWARE: If the user asked for something simple, keep the explanation punchy. If global or complex, provide a deeper walkthrough.
    - ADAPTIVE: Use the conversational history to detect if this is an initial design or a modification.
    """

    context_message = f"""
    LATEST USER REQUEST: {user_input}
    
    CONVERSATION HISTORY:
    {json.dumps(history, indent=2)}
    
    OPTIMIZED SCHEMA (TO EXPLAIN):
    {json.dumps(schema_plan, indent=2)}
    
    ARCHITECTURE METADATA:
    - Optimization Notes: {json.dumps(optimization_notes, indent=2)}
    - Suggestions: {json.dumps(suggestions, indent=2)}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context_message)
    ]

    response = await llm.ainvoke(messages)
    
    state["interaction_message"] = response.content
    state["response"] = response.content
    
    print("Schema Explanation Generated.")
    return state
