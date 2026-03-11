from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.graph.workflow import create_workflow

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Compile the workflow shell
app_workflow = create_workflow()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

SESSIONS = {}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    FastAPI endpoint to trigger the LLM-driven AI Agent workflow.
    """
    session_id = "default_session"
    
    # Load existing state or initialize new
    state = SESSIONS.get(session_id, {
        "user_input": "",
        "user_query": "",
        "intent": "",
        "scope": "",
        "task_queue": [],
        "current_task": None,
        "operation_type": "",
        "table_name": "",
        "schema": {},
        "data": None,
        "query": "",
        "missing_fields": [],
        "inferred_fields": {},
        "validation_results": {},
        "execution_result": None,
        "execution_error": None,
        "response": "",
        "analysis": "",
        "planned_task": "",
        "current_task_index": 0,
        "intent_category": "",
        "ddl_operation": "",
        "dml_operation": "",
        "modify_operation": {},
        "operation": "",
        "conversation_history": [],
        "field_registry": {},
        "schema_data": {"table_name": None, "columns": []},
        "schema_ready": False,
        "debug_info": "",
        "interaction_request": {},
        "interaction_message": "",
        "strapi_payload": {},
        "strapi_endpoint": "",
        "user_provided_missing_data": False,
        "interaction_complete": False,
        "interaction_phase": False,
        "active_agent": None,
        "interaction_attempts": 0,
        "route_decision": None,
        "memory": {},
        "messages": []
    })
    
    # Update state for the current turn
    state["user_input"] = request.message
    state["user_query"] = request.message
    
    # Reset some per-turn flags
    state["response"] = ""
    state["analysis"] = ""
    state["debug_info"] = ""
    
    try:
        # Run the workflow asynchronously using ainvoke
        final_state = await app_workflow.ainvoke(state)
        SESSIONS[session_id] = final_state
        return {"response": final_state.get("response", "I encountered an error while processing your request.")}
    except Exception as e:
        print(f"Error executing workflow: {e}")
        return {"response": f"System error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
