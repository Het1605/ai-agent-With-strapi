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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    FastAPI endpoint to trigger the LLM-driven AI Agent workflow.
    """
    # Initialize the production state
    initial_state = {
        "user_input": request.message,
        "user_query": request.message,
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
        "response": "",
        "analysis": "",
        "planned_task": "",
        "current_task_index": 0,
        "intent_category": "",
        "ddl_operation": "",
        "dml_operation": "",
        "conversation_history": [],
        "field_registry": {},
        "memory": {},
        "messages": []
    }
    
    try:
        # Run the workflow asynchronously using ainvoke
        final_state = await app_workflow.ainvoke(initial_state)
        return {"response": final_state.get("response", "I encountered an error while processing your request.")}
    except Exception as e:
        print(f"Error executing workflow: {e}")
        return {"response": f"System error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
