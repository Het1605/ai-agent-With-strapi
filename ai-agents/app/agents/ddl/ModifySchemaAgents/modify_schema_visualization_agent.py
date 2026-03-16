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
    
    if not schema_design or not schema_design.get("operations"):
        print("[ModifySchemaVisualizationAgent] No schema design found to visualize.")
        state["interaction_message"] = "I couldn't find any schema modifications to apply based on your request. Could you clarify what exactly you want to change?"
        state["response"] = state["interaction_message"]
        return state

    system_prompt = """
    You are a Senior Software Architect communicating database schema modifications to a stakeholder.
    Your task is to translate a structured JSON schema design into a very clear, natural, and UX-friendly explanation.

    --------------------------------------------------
    THE INPUT
    --------------------------------------------------
    You will receive:
    1. The structured Schema Design (containing add_column, delete_column, update_column, update_collection).
    2. The User's latest request.
    3. Conversation history (to detect if this is a first draft or an iteration).

    --------------------------------------------------
    YOUR OBJECTIVE
    --------------------------------------------------
    1. Translate the JSON into a readable summary.
    2. Group the explanation by Table and Operation.
    3. Briefly explain the purpose of new columns or the impact of deleted/updated columns.
    4. At the end, ask the user for their approval in a natural way.

    --------------------------------------------------
    FORMATTING RULES
    --------------------------------------------------
    - Use Markdown efficiently (headings, bullet points, bold text).
    - NEVER invent columns or tables that are not in the JSON.
    - NEVER show raw JSON or code snippets to the user.
    - Be conversational but highly professional.
    - Adapt based on context:
      * If this is the FIRST explanation, outline all changes clearly.
      * If this is an ITERATION (the user just asked for an adjustment), say something like "Based on your feedback, I've adjusted..." and highlight what changed.
      
    --------------------------------------------------
    EXAMPLE OUTPUT STYLE
    --------------------------------------------------
    ### 🏢 Department Table Updates

    The following modifications will be made to the `departments` table:

    **Columns to Add:**
    * **`budget`** (Decimal)
    * **`manager_id`** (Integer) — Linked to the responsible manager.

    **Columns to Remove:**
    * **`temporary_code`**

    ### 👤 Employee Table Updates

    **Columns to Modify:**
    * **`salary`** will now be a required field.

    ---
    Would you like me to proceed with these changes, or is there anything else you want to adjust?
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
