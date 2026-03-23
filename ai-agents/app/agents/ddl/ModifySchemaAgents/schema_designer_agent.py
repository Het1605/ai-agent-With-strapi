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
    
    memory = state.get("modify_schema_memory", {})
    previous_design = memory.get("latest_design", {})
    field_registry = state.get("field_registry", {})
    existing_schema = state.get("existing_schema", {})

    print("schema_plan :",schema_plan)

    print("previous_design:",previous_design)

    
    if not schema_plan or not schema_plan.get("operations"):
        print("[ModifySchemaDesignerAgent] No operations planned.")
        state["modify_schema_design"] = {"operations": []}
        return state

    system_prompt = """
    You are a Senior Middleware Schema Engineer acting as the ModifySchemaDesignerAgent.
    Your sole purpose is to act as a DETERMINISTIC TRANSFORMER of the Modification Plan provided by the Planner Agent.

    You will receive:
    1. The target Modification Plan (from the Planner Agent, representing the final intended state)
    2. The entire modify_schema_memory (containing the history of previous designs)
    3. The Field Registry (valid data types/constraints)
    4. The Conversation History
    5. The existing_schema

    --------------------------------------------------
    YOUR CORE RESPONSIBILITIES
    --------------------------------------------------
    1. ABSOLUTE COMPLIANCE TO PLANNER OUTPUT
       - The Planner Agent has already done the heavy lifting of figuring out memory merges and user intent.
       - The Modification Plan is the SINGLE SOURCE OF TRUTH.
       - You MUST NOT drop, reduce, or ignore ANY field or operation provided by the Planner.
       - If the Planner provides 6 columns in its `columns_to_add` array, you MUST output ALL 6 columns fully expanded.

    2. INTELLIGENT FIELD EXPANSION
       - Your job isn't to think about what the user wants; your job is to engineer the database architecture for exactly what the Planner output.
       - Use the `field_registry` to assign robust, production-ready types and constraints mathematically.
       - Example: "salary" -> `type: decimal`, `min: 0`, `required: true`.
       - Example: "email" -> `type: email`, `unique: true`, `required: true`.
       - Do not simply default to "string". Infer the optimal format from the column name.

    3. STRICT DATA VALIDATION
       - Ensure your output strictly complies with `existing_schema`. Do not act upon fields unless they are logically valid based on the constraints.
       - Omit purely null or unused attributes (e.g., omit `"minLength": null`).

    --------------------------------------------------
    STRICT NAMING RULES (RESERVED KEYWORDS)
    --------------------------------------------------
    - THE FIELD NAME "status" is strictly PROHIBITED.
    - Always use contextual naming: `{table_name_singular}_status`.
    - If the Planner accidentally included a field named "status", you MUST rename it.

    --------------------------------------------------
    COLLECTION-LEVEL RULES
    --------------------------------------------------
    - If `update_collection` specifies `{"delete": true}`, pass it through exactly as defined.
    - If `update_collection` specifies `{"delete": true}`, pass it through exactly as defined.
    - If it specifies `{"displayName": "New Name"}`, pass it through exactly.
    - Never infer collection deletions on your own.

    --------------------------------------------------
    STRICT RELATIONSHIP RULES (SINGLE-SIDE ONLY - HARD CONSTRAINT)
    --------------------------------------------------
    - 🚨 CRITICAL: Enforce STRICT SINGLE-SIDE relation definition.
    - NO MUTUAL REFERENCES: If Table A references Table B, Table B **MUST NOT** reference Table A.
    - PREFERRED DIRECTION (CHILD-SIDE): Always design relations on the CHILD side of the relationship.
      - If adding a relation, place it in the table that "belongs to" the other (e.g., `manyToOne`).
      - Every relationship MUST exist only once and in only ONE direction (CHILD to PARENT).
    - ONE-TO-ONE RESOLUTION: Keep exactly one side of a one-to-one relationship.

    --------------------------------------------------
    OUTPUT FORMAT (STRICT JSON)
    --------------------------------------------------
    Return ONLY a valid JSON object starting with `{ "operations": [ ... ] }`.
    No explanations, no wrapper blocks.

    Example constraint format mapping for Planner's "salary" and "email":
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
        }
      ]
    }
    """

    context_message = f"""
    FIELD REGISTRY:
    {json.dumps(field_registry, indent=2) if field_registry else 'None provided, use standard types.'}
    
    LATEST USER INPUT:
    "{user_input}"
    
    MODIFICATION PLAN (from Planner - SINGLE SOURCE OF TRUTH):
    {json.dumps(schema_plan, indent=2)}
    
    SYSTEM MEMORY (History of Previous Designs):
    {json.dumps(memory, indent=2)}

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
        
        if "operations" not in design:
            design = {"operations": []}
            
        if memory:
            memory["designer_history"].append(design)
            memory["latest_design"] = design
            state["modify_schema_memory"] = memory
            
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
