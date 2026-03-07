from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.graph.workflow import create_workflow

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Compile the workflow once at startup
app_workflow = create_workflow()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    FastAPI endpoint that triggers the LangGraph SupervisorAgent.
    """
    # Initialize the state
    initial_state = {
        "user_input": request.message,
        "response": ""
    }
    
    # Run the workflow
    final_state = app_workflow.invoke(initial_state)
    
    return {"response": final_state.get("response", "No response generated.")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
