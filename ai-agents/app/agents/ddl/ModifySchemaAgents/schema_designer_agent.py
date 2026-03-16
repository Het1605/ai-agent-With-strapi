import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def schema_designer_agent(state: AgentState) -> AgentState:
    """
    ModifySchemaDesignerAgent: Converts modification plans into precise,
    fully-structured database schema changes.
    """
    print("\n----- ENTERING ModifySchemaDesignerAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])
    schema_plan = state.get("modify_schema_plan", {})
    previous_design = state.get("modify_schema_design", {})
    field_registry = state.get("field_registry", {})
    
    if not schema_plan or not schema_plan.get("operations"):
        print("[ModifySchemaDesignerAgent] No operations planned.")
        state["modify_schema_design"] = {"operations": []}
        return state

    system_prompt = """
    You are a Senior Database Schema Designer.
    Your task is to convert high-level schema modification plans into fully detailed schema changes.

    You will receive:
    1. The target Modification Plan (from the Planner Agent)
    2. The previous Schema Design (if this is an iterative refinement)
    3. The Field Registry mapping allowed types and constraints
    4. The Conversation History and Latest User Input

    --------------------------------------------------
    YOUR OBJECTIVE
    --------------------------------------------------
    Generate a highly structured JSON document detailing exactly HOW to modify the schema.
    Determine the optimal standard data types, constraints (required, unique, default), and specific structural changes.

    Supported Intents:
    1. add_column
    2. delete_column
    3. update_column
    4. update_collection

    --------------------------------------------------
    OUTPUT STRUCTURE
    --------------------------------------------------
    Return ONLY a valid JSON object starting with `{ "operations": [ ... ] }`.
    Do not wrap it in markdown. Do not provide explanations.

    Example for add_column:
    {
      "intent": "add_column",
      "table": "employees",
      "columns": [
        {
          "name": "salary",
          "type": "decimal",
          "required": false,
          "unique": false,
          "default": null
        }
      ]
    }

    Example for delete_column:
    {
      "intent": "delete_column",
      "table": "employees",
      "columns": [
        {"name": "leave_balance"}
      ]
    }

    Example for update_column:
    {
      "intent": "update_column",
      "table": "employees",
      "columns": [
        {
          "name": "salary",
          "changes": {
            "type": "decimal",
            "required": true
          }
        }
      ]
    }

    Example for update_collection:
    {
      "intent": "update_collection",
      "table": "employees",
      "changes": {
        "rename_to": "staff_members"
      }
    }

    --------------------------------------------------
    RULES
    --------------------------------------------------
    1. Group outputs inside an "operations" array.
    2. Follow the Planner's requested structure.
    3. Modify existing definitions if the user iteratively refined their request ("actually, make salary required").
    4. Pick the most realistic data types defined in conventional systems.
    """

    context_message = f"""
    FIELD REGISTRY:
    {json.dumps(field_registry, indent=2) if field_registry else 'None provided, use standard types (string, integer, boolean, date, etc).'}
    
    LATEST USER INPUT:
    "{user_input}"
    
    MODIFICATION PLAN (from Planner):
    {json.dumps(schema_plan, indent=2)}
    
    PREVIOUS SCHEMA DESIGN (if Iterative):
    {json.dumps(previous_design, indent=2)}
    
    CONVERSATION HISTORY:
    {json.dumps(history, indent=2)}
    """

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_message)
        ])
        
        content = response.content.replace("```json", "").replace("```", "").strip()
        design = json.loads(content)
        
        if "operations" not in design:
            design = {"operations": []}
            
        state["modify_schema_design"] = design
        print(f"[ModifySchemaDesignerAgent] Designed {len(design.get('operations', []))} operations.")
        
    except json.JSONDecodeError as e:
        print(f"[ModifySchemaDesignerAgent] JSON parse error: {e}")
        if not state.get("modify_schema_design"):
            state["modify_schema_design"] = {"operations": []}
    except Exception as e:
        print(f"[ModifySchemaDesignerAgent] Unexpected error: {e}")
        if not state.get("modify_schema_design"):
            state["modify_schema_design"] = {"operations": []}

    return state
