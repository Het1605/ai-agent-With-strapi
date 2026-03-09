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
    current_schema = state.get("schema_data", {"table_name": None, "columns": []})
    field_registry = state.get("field_registry", {})
    
    system_prompt = (
        "You are a Database Schema Specialist. Your role is to analyze a new user message and extract ONLY the newly provided schema constraints, fields, or table name.\n\n"
        "Required Information for a full schema:\n"
        "1. table_name: A clear, singular name for the table.\n"
        "2. columns: A list of column objects with 'name', 'type', and relevant constraints.\n\n"
        f"Available Strapi Field Registry: {json.dumps(field_registry)}\n\n"
        "Instructions:\n"
        "- Extract new constraints, column names, or table names from the Current User Message.\n"
        "- Do NOT worry about preserving previous columns. Just output what is NEW in this specific message.\n"
        "- If the user provides a table name, output it.\n"
        "- Detect 'required', 'unique', 'default', 'enum', etc., based on natural language clues.\n"
        "- Output a JSON object with 'extracted_data' (acting as a partial schema_data object with a 'table_name' if found, and a 'columns' list of new or modified columns).\n"
        "Output ONLY valid JSON."
    )
    
    human_msg = (
        f"Current User Message: {user_query}\n"
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])
    
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
        extracted = result.get("extracted_data", {})
        
        # 1. Read existing state
        # 2. Extract new info (done by LLM)
        # 3. Merge new fragment into schema_data
        # 4. Preserve previously collected fields
        
        # Merge table name
        if extracted.get("table_name"):
            current_schema["table_name"] = extracted.get("table_name")
            
        # Merge columns
        existing_cols = {col["name"]: col for col in current_schema.get("columns", [])}
        for new_col in extracted.get("columns", []):
            name = new_col.get("name")
            if not name:
                continue
            if name in existing_cols:
                existing_cols[name].update(new_col)
            else:
                existing_cols[name] = new_col
                
        current_schema["columns"] = list(existing_cols.values())
        state["schema_data"] = current_schema
        
        # Compute missing fields dynamically
        missing = []
        if not current_schema.get("table_name"):
            missing.append("table_name")
        if not current_schema.get("columns") or len(current_schema["columns"]) == 0:
            missing.append("columns (at least one column is required)")
        else:
            for c in current_schema["columns"]:
                if "type" not in c:
                    missing.append(f"data type for column '{c.get('name')}'")
        
        state["missing_fields"] = missing
        
        if not missing:
            state["schema_ready"] = True
            state["interaction_phase"] = False
            state["active_agent"] = None
            state["interaction_attempts"] = 0
            state["debug_info"] = "Schema is complete and ready for query building."
        else:
            state["schema_ready"] = False
            state["debug_info"] = f"Missing schema information detected: {', '.join(missing)}"
            
    except Exception as e:
        print(f"Error in CreateTableAgent reasoning: {e}")
        state["schema_ready"] = False
        state["missing_fields"] = ["internal_parsing_error"]
        
    return state
