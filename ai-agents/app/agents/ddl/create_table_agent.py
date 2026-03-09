from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def create_table_agent(state: AgentState) -> AgentState:
    """
    CreateTableAgent: AI agent that uses LLM reasoning to validate and extract table schema.
    It ensures all column properties (min, max, required, etc.) are captured.
    """
    print("\n----- ENTERING CreateTableAgent (AI Reasoning) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_query = state.get("user_query", "")
    current_schema = state.get("schema_data", {})
    field_registry = state.get("field_registry", {})
    history = state.get("conversation_history", [])
    
    # Format history for prompt
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-10:]])
    
    system_prompt = (
        "You are a Database Schema Specialist. Your role is to analyze a user query and the conversation history "
        "to determine the complete structure for creating a new database table.\n\n"
        "Required Information:\n"
        "1. table_name: A clear, singular name for the table.\n"
        "2. columns: A list of column objects with 'name', 'type', and all relevant constraints.\n\n"
        f"Available Strapi Field Registry: {json.dumps(field_registry)}\n\n"
        "Instructions:\n"
        "- Use the CONVERSATION HISTORY to resolve fields that were previously missing.\n"
        "- Extract all possible constraints from the user request (e.g., 'age between 18 and 90' -> min: 18, max: 90).\n"
        "- Detect 'required', 'unique', 'default', 'enum', etc., based on natural language clues.\n"
        "- If the request is vague or still missing critical data, list what is missing in 'missing_fields'.\n"
        "- Output a JSON object with two fields:\n"
        "  1. 'schema_data': The complete schema object including all column properties.\n"
        "  2. 'missing_fields': An array of missing requirements.\n\n"
        "Example Output Structure:\n"
        "{\n"
        "  \"schema_data\": {\n"
        "    \"table_name\": \"student\",\n"
        "    \"columns\": [\n"
        "      {\"name\": \"name\", \"type\": \"string\", \"required\": true},\n"
        "      {\"name\": \"age\", \"type\": \"integer\", \"min\": 18, \"max\": 90}\n"
        "    ]\n"
        "  },\n"
        "  \"missing_fields\": []\n"
        "}"
    )
    
    human_msg = (
        f"Conversation History:\n{history_str}\n\n"
        f"Current User Message: {user_query}\n"
        f"Existing Schema State: {json.dumps(current_schema)}"
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])
    
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
        
        state["schema_data"] = result.get("schema_data", {})
        missing = result.get("missing_fields", [])
        state["missing_fields"] = missing
        
        if not missing:
            state["schema_ready"] = True
            state["debug_info"] = "Schema is complete and ready for query building."
        else:
            state["schema_ready"] = False
            state["debug_info"] = f"Missing schema information detected: {', '.join(missing)}"
            
    except Exception as e:
        print(f"Error in CreateTableAgent reasoning: {e}")
        state["schema_ready"] = False
        state["missing_fields"] = ["internal_parsing_error"]
        
    return state
