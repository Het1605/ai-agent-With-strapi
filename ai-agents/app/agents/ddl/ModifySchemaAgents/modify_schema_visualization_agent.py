import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def modify_schema_visualization_agent(state: AgentState) -> AgentState:
    """
    ModifySchemaVisualizationAgent: Translates structured schema modification
    designs into a natural, UX-friendly explanation for the user to approve.
    """
    print("\n----- ENTERING ModifySchemaVisualizationAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])
    schema_design = state.get("modify_schema_design", {})

    print("schema_design:",schema_design)
    
    if not schema_design or not schema_design.get("operations"):
        print("[ModifySchemaVisualizationAgent] No schema design found to visualize.")
        state["interaction_message"] = "I couldn't find any schema modifications to apply based on your request. Could you clarify what exactly you want to change?"
        state["response"] = state["interaction_message"]
        return state

    system_prompt = """
    You are a highly capable Senior Software Architect communicating database schema modifications to a stakeholder.
    Your task is to translate a structured JSON schema design into a very clear, natural, and UX-friendly explanation that dynamically adapts to the scope of changes.

    --------------------------------------------------
    YOUR OBJECTIVE
    --------------------------------------------------
    1. Translate the JSON into a readable summary.
    2. Analyze the operations to determine the best structure. 
       - If there's only a single column change, keep the response short and conversational.
       - If there are multiple operations across multiple tables, group the explanation by Table and Operation clearly using headings.
    3. Briefly explain the purpose of new columns or the impact of deleted/updated columns based on normal tech assumptions.
    4. Conclude by naturally asking the user for their approval. Do not use a static templated generic response.

    --------------------------------------------------
    ITERATION AWARENESS
    --------------------------------------------------
    - If the Conversation History implies this is the FIRST explanation of a new request, outline all changes clearly.
    - If the Conversation History implies this is an ITERATION (e.g., the user said "Make target_date required"), do NOT repeat the entire explanation. Instead, say something like "Based on your feedback, I've updated the schema..." and dynamically focus on the adjustment alongside the overall readiness of the design.

    --------------------------------------------------
    FORMATTING RULES
    --------------------------------------------------
    - NEVER follow a rigid visual template if it doesn't fit the context. Adapt the formatting (bullet points, paragraphs).
    - NEVER invent columns or tables that are not in the JSON.
    - NEVER modify or invent schema changes. You strictly explain what the Designer produced.
    - NEVER show raw JSON or code snippets to the user.
    - End the response with a natural, varied question asking if they want to proceed/execute/apply the modifications.
    """

    context_message = f"""
    STRUCTURED SCHEMA DESIGN TO EXPLAIN:
    {json.dumps(schema_design, indent=2)}
    
    LATEST USER INPUT:
    "{user_input}"
    
    CONVERSATION HISTORY:
    {json.dumps(history, indent=2)}
    """

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_message)
        ])
        
        explanation = response.content.strip()
        state["interaction_message"] = explanation
        state["response"] = explanation
        print("[ModifySchemaVisualizationAgent] Successfully generated schema visualization.")
        
    except Exception as e:
        print(f"[ModifySchemaVisualizationAgent] Unexpected error: {e}")
        error_msg = "I've drafted the schema modifications, but I encountered an issue presenting them. Could you try asking again?"
        state["interaction_message"] = error_msg
        state["response"] = error_msg

    return state
