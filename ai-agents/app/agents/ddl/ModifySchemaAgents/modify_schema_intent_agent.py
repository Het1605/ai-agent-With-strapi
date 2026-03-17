import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def modify_schema_intent_agent(state: AgentState) -> AgentState:
    """
    ModifySchemaIntentAgent: Analyzes the user's request and detects ALL 
    schema modification operations mentioned.
    """
    print("\n----- ENTERING ModifySchemaIntentAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])
    existing_collections = state.get("existing_collections", [])
    existing_schema = state.get("existing_schema", {})
    
    if not user_input:
        print("[ModifySchemaIntentAgent] No user input found.")
        state["modify_operations"] = []
        return state

    system_prompt = """
    You are a Schema Modification Intent Classifier.
    Your job is to detect:
    1. The type of schema operation (intent)
    2. The target table
    3. The user's instruction (details)
    
    --------------------------------------------------
    CRITICAL CONTEXT
    --------------------------------------------------
    You are given:
    - user_input
    - conversation_history
    - existing_collections (valid tables)
    - existing_schema (actual database structure)
    
    You MUST use existing_schema to understand whether the request is valid and schema-related.
    
    --------------------------------------------------
    IMPORTANT BEHAVIOR
    --------------------------------------------------
    You MUST NOT return empty operations if the request is related to schema.
    If you return empty operations, the system pipeline breaks.
    This is STRICTLY FORBIDDEN.
    
    --------------------------------------------------
    SUPPORTED INTENTS
    --------------------------------------------------
    You MUST classify into one of:
    1. add_column
    2. delete_column
    3. update_column
    4. update_collection
    
    --------------------------------------------------
    OUTPUT FORMAT (STRICT)
    --------------------------------------------------
    Return ONLY JSON:
    {
      "operations": [
        {
          "intent": "add_column | delete_column | update_column | update_collection",
          "table": "table_name or null",
          "details": "cleaned user instruction"
        }
      ]
    }
    
    --------------------------------------------------
    STRICT RULES
    --------------------------------------------------
    1. ALWAYS return at least one operation for schema-related input
    2. NEVER return empty operations unless input is completely unrelated
    3. DO NOT extract columns
    4. DO NOT design schema
    5. DO NOT hallucinate tables
    6. Only match tables from existing_collections
    7. Use the conversation history ONLY for context, DO NOT re-extract completed past operations.
    
    --------------------------------------------------
    TABLE DETECTION (IMPORTANT)
    --------------------------------------------------
    - Match user input with existing_collections
    - Use fuzzy normalization (e.g., "employees" -> "employee", "employee leave" -> "employee-leave")
    - If table exists in existing_schema -> HIGH CONFIDENCE
    - If not found -> table = null
    
    --------------------------------------------------
    SCHEMA-AWARE DECISION MAKING
    --------------------------------------------------
    Use existing_schema to:
    - Confirm the system contains tables
    - Increase confidence that request is schema-related
    - Avoid returning empty output
    Even if table is unclear -> still return intent
    
    --------------------------------------------------
    VAGUE REQUEST HANDLING
    --------------------------------------------------
    User: "remove unnecessary columns"
    You MUST return:
    {
      "operations": [
        {
          "intent": "delete_column",
          "table": null,
          "details": "remove unnecessary columns"
        }
      ]
    }
    DO NOT return empty.
    
    --------------------------------------------------
    MULTI-OPERATION SUPPORT
    --------------------------------------------------
    User: "add fields to employee and remove columns from order"
    Return:
    {
      "operations": [
        {
          "intent": "add_column",
          "table": "employee",
          "details": "add fields"
        },
        {
          "intent": "delete_column",
          "table": "order",
          "details": "remove columns"
        }
      ]
    }
    
    --------------------------------------------------
    FAIL-SAFE
    --------------------------------------------------
    If intent is weak:
    Return:
    {
      "operations": [
        {
          "intent": "update_collection",
          "table": null,
          "details": "<user_input>"
        }
      ]
    }
    
    --------------------------------------------------
    GOAL
    --------------------------------------------------
    You must behave like a robust intent extractor:
    - Always produce output
    - Never break pipeline
    - Use schema for grounding
    - Never hallucinate
    """

    context_message = f"""
    EXISTING COLLECTIONS:
    {json.dumps(existing_collections, indent=2) if existing_collections else '[]'}
    
    EXISTING SCHEMA:
    {json.dumps(existing_schema, indent=2) if existing_schema else '{}'}
    
    CONVERSATION HISTORY:
    {json.dumps(history, indent=2)}
    
    USER INPUT:
    "{user_input}"
    """

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_message)
        ])
        
        # Parse the JSON array
        content = response.content.replace("```json", "").replace("```", "").strip()

        print("Intent_agent:",content)

        parsed_json = json.loads(content)
        operations = parsed_json.get("operations", [])

        print("operations:",operations)
        
        if not isinstance(operations, list):
            print("[ModifySchemaIntentAgent] Warning: Output was not a list. Defaulting to empty list.")
            operations = []
            
        # Optional validation
        valid_intents = {"add_column", "delete_column", "update_column", "update_collection"}
        validated_operations = []
        for op in operations:
            if op.get("intent") in valid_intents:
                validated_operations.append(op)
            else:
                print(f"[ModifySchemaIntentAgent] Skipping invalid operation: {op}")
        
        state["modify_operations"] = validated_operations
        print(f"[ModifySchemaIntentAgent] Detected {len(validated_operations)} operations: {validated_operations}")
        
    except json.JSONDecodeError as e:
        print(f"[ModifySchemaIntentAgent] JSON parse error: {e}")
        state["modify_operations"] = []
    except Exception as e:
        print(f"[ModifySchemaIntentAgent] Unexpected error: {e}")
        state["modify_operations"] = []

    return state
