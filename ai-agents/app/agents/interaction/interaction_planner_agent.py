from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def interaction_planner_agent(state: AgentState) -> AgentState:
    """
    InteractionPlannerAgent: Manages the interaction loop. 
    It either asks for missing data OR processes the provided data and prepares to loop back.
    """
    print("\n----- ENTERING InteractionPlannerAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_input = state.get("user_input", "")
    missing_fields = state.get("missing_fields", [])
    schema_data = state.get("schema_data", {})
    history = state.get("conversation_history", [])
    
    # Check if the user just provided something that looks like it's for the schema
    # We use a bit of reasoning here to see if the current input is a response to the missing fields
    
    analysis_prompt = (
        "You are an Interaction Analyst. Determine if the user's latest message is providing "
        f"information to fill these missing schema fields: {missing_fields}.\n\n"
        f"Latest Message: {user_input}\n"
        "Respond with a JSON object: {'is_providing_data': boolean, 'extracted_info': {column_definitions}}"
    )
    
    response = await llm.ainvoke([SystemMessage(content=analysis_prompt)])
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(clean_content)
        
        if analysis.get("is_providing_data"):
            print("InteractionPlannerAgent: User provided missing data.")
            state["user_provided_missing_data"] = True
            # Update schema_data with extracted info (CreateTableAgent will refine this)
            # This is a bit of a shortcut, but it signals to the router to go back to create_table
            return state
        
    except Exception as e:
        print(f"Error analyzing interaction: {e}")

    # If we are here, we need to ASK the user for information
    state["user_provided_missing_data"] = False
    
    request_prompt = (
        "You are a Database Assistant. A user wants to create a table but is missing some information.\n"
        f"Missing Fields: {missing_fields}\n"
        f"Current Schema: {json.dumps(schema_data)}\n\n"
        "Generate a friendly, professional question to ask the user for this missing information."
    )
    
    request_response = await llm.ainvoke([SystemMessage(content=request_prompt)])
    state["response"] = request_response.content.strip()
    
    print(f"InteractionPlannerAgent: Asking user -> {state['response']}")
    
    return state
