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
    existing_collections = state.get("existing_collections", [])
    existing_schema = state.get("existing_schema", {})

    print("schema_plan :",schema_plan)

    print("previous_design:",previous_design)

    
    if not schema_plan or not schema_plan.get("operations"):
        print("[ModifySchemaDesignerAgent] No operations planned.")
        state["modify_schema_design"] = {"operations": []}
        return state

    system_prompt = """
    You are a Senior Database Architect and Schema Designer.
    Your task is to convert high-level schema modification plans into fully detailed, production-ready schema changes.

    You will receive:
    1. The target Modification Plan (from the Planner Agent)
    2. The previous Schema Design (if this is an iterative refinement)
    3. The Field Registry mapping allowed types and all valid constraints/attributes
    4. The Conversation History and Latest User Input

    --------------------------------------------------
    YOUR OBJECTIVE & STRICT SCHEMA AWARENESS RULES
    --------------------------------------------------
    You are to produce the final, specific JSON describing exactly how columns should be added, deleted, or updated.

    1. STRICT SCHEMA USAGE: You MUST ONLY use columns from EXISTING COLLECTIONS SCHEMA. 
       If a column is not present in existing_schema, DO NOT include it in delete or update operations.
       If adding a column, ensure it does not already exist.

    2. FIELD REGISTRY USAGE: Translate natural language into specific database definitions.
       - Use `field_registry` to determine valid types and constraints.
       - If type="string", use constraints like `minLength`, `maxLength`, `regex`.
       - If type="decimal", use `min`, `max`.
       - If type="media", use `multiple`, `allowedTypes`.
       - If type="relation", use `relation`, `target`, `targetAttribute`.
       - DO NOT guess attributes. Only use what is listed in the Field Registry.

    3. GENERATE REALISTIC DATABASE FIELDS: Make logical assumptions about constraints.
       - A "salary" column should likely be `type: decimal`, `min: 0`, `required: true`.
       - An "email" column should likely be `type: email`, `unique: true`, `required: true`.
       - A "name" column should likely be `type: string`, `minLength: 2`, `maxLength: 100`.

    4. DO NOT INCLUDE NULL OR UNUSED ATTRIBUTES: 
       - Omit any attribute that is null, false, or irrelevant. Do NOT output `"minLength": null`.

    5. SUPPORT MULTIPLE OPERATIONS: The input may contain modifications for multiple tables, multiple columns, and multiple intents. Address them all.

    --------------------------------------------------
    OUTPUT FORMAT
    --------------------------------------------------
    Return ONLY a valid JSON object starting with `{ "operations": [ ... ] }`.
    No explanations, no wrapper text, no markdown block quotes around the JSON unless strictly necessary.

    Example structure:
    {
      "operations": [
        {
          "intent": "add_column",
          "table": "employees",
          "columns": [
            {
              "name": "salary",
              "type": "decimal",
              "min": 0,
              "required": true
            },
            {
              "name": "email",
              "type": "email",
              "unique": true,
              "required": true
            }
          ]
        },
        {
          "intent": "delete_column",
          "table": "employees",
          "columns": [
            {"name": "leave_balance"}
          ]
        },
        {
          "intent": "update_column",
          "table": "employees",
          "columns": [
            {
              "name": "status",
              "changes": {
                "type": "enumeration",
                "enum": ["active", "inactive"]
              }
            }
          ]
        },
        {
          "intent": "update_collection",
          "table": "employees",
          "changes": {
            "displayName": "Staff"
          }
        }
      ]
    }
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

    EXISTING COLLECTIONS SCHEMA:
    {json.dumps(existing_schema, indent=2) if existing_schema else "[]"}
    
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

        print("Designer Content:",content)

        print("Designer Design:",design)
        
        
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
