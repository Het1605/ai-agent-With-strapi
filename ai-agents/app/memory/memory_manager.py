from app.graph.state import AgentState
from app.memory.conversation_memory import initialize_conversation_memory, add_user_message
from app.memory.schema_memory import attach_schema_memory_to_state

async def memory_manager(state: AgentState) -> AgentState:
    """
    MemoryManager: Prepares the context for the workflow.
    Runs immediately after SupervisorAgent.
    """
    print("\n----- ENTERING MemoryManager -----")
    
    # 1. Initialize conversation history
    state = initialize_conversation_memory(state)

   # print("initialize_conversation_memory:",state)
    
    # 2. Add current user message to history
    # (The user request says to add it here)
    state = add_user_message(state)

   # print("add_user_message:",state)
    
    # 3. Load and attach schema metadata
    print("[MemoryManager] Loading schema context (existing collections)...")
    state = attach_schema_memory_to_state(state)
    print(f"[MemoryManager] Existing collections in state: {len(state.get('existing_collections', []))}")

   # print("attach_schema_memory_to_state :",state)
    
    print(f"MemoryManager: Conversation history size -> {len(state['conversation_history'])}")
    print(f"MemoryManager: Field registry loaded -> {bool(state['field_registry'])}")
    
    return state
