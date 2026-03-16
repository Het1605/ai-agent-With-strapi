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
    
    if not user_input:
        print("[ModifySchemaIntentAgent] No user input found.")
        state["modify_operations"] = []
        return state

    system_prompt = """
    You are an AI Schema Modification Intent Detector.
    Your goal is to analyze the LATEST user input and extract ONLY the NEW schema modification operations requested in this turn.
    
    The user may request multiple operations in a single message.
    
    --------------------------------------------------
    SUPPORTED INTENTS
    --------------------------------------------------
    Only these four intents are allowed:
    1. add_column
    2. delete_column
    3. update_column
    4. update_collection
    
    Do NOT invent new intent types.
    
    --------------------------------------------------
    RESPONSIBILITY
    --------------------------------------------------
    1. Extract ONLY NEW schema modification operations newly mentioned in the latest user input.
    2. Use the conversation history ONLY for context or resolving pronouns (e.g., 'table'), but DO NOT re-extract operations that were already requested and completed in the history.
    3. Identify the target collection/table for each new operation.
    4. Provide a brief detail of the requested action.
    5. Do NOT attempt to design the schema fields here.
    6. Do NOT guess column types.
    
    --------------------------------------------------
    EXPECTED OUTPUT FORMAT (STRICT JSON ARRAY)
    --------------------------------------------------
    Return ONLY a valid JSON array of objects. Do not include markdown blocks or other text.
    
    Format:
    [
      {
        "intent": "add_column",
        "table": "employees",
        "details": "add salary column"
      },
      {
        "intent": "delete_column",
        "table": "employees",
        "details": "remove leave_balance column"
      }
    ]
    """

    context_message = f"""
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

        operations = json.loads(content)

        print("operations:",operations)
        
        if not isinstance(operations, list):
            print("[ModifySchemaIntentAgent] Warning: Output was not a list. Defaulting to empty list.")
            operations = []
            
        # Optional validation
        valid_intents = {"add_column", "delete_column", "update_column", "update_collection"}
        validated_operations = []
        for op in operations:
            if op.get("intent") in valid_intents and "table" in op:
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
