import os
import json
import requests
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRAPI_BASE_URL = os.getenv("STRAPI_BASE_URL")

# Define State
class AgentState(TypedDict):
    user_query: str
    schema: Optional[dict]
    validation_results: Optional[str]
    api_response: Optional[dict]
    error: Optional[str]

# Node 1: Generate Schema
def generate_schema(state: AgentState):
    print("--- GENERATING SCHEMA ---")
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    
    prompt = f"""
    You are a Strapi v5 Schema Architect.
    Convert the following user query into a simplified logical JSON schema.
    
    User Query: "{state['user_query']}"
    
    Return ONLY valid JSON with this structure:
    {{
      "collectionName": "plural_name",
      "fields": [
        {{
          "name": "field_name",
          "type": "string | integer | decimal | float | biginteger | boolean | date | datetime | email | enumeration | relation | uid | media | json | password | blocks | richtext",
          "required": true/false,
          "unique": true/false,
          "default": "value",
          "private": true/false,
          "configurable": true/false,
          "minLength": 3,      // string/text/password/uid
          "maxLength": 30,     // string/text/password/uid
          "regex": "^[a-z]+$", // string/text/password/uid
          "min": 1,            // numeric
          "max": 100,          // numeric
          "enum": ["opt1", "opt2"], // only for enumeration
          "relation": "manyToOne | oneToMany | manyToMany", // only for relation
          "target": "api::collection.collection", // only for relation
          "multiple": true/false // only for media
        }}
      ]
    }}
    
    Rules:
    - Use plural names for collectionName (e.g., "students").
    - target UIDs MUST be in the format "api::singularName.singularName" (e.g., "api::student.student").
    - Be strict with types and validations.
    """
    
    response = llm.invoke(prompt)
    content = response.content.strip().replace("```json", "").replace("```", "").strip()
    
    try:
        schema = json.loads(content)
        return {"schema": schema}
    except Exception as e:
        return {"error": f"Failed to parse LLM response: {str(e)}"}

# Node 2: Validate Schema
def validate_schema(state: AgentState):
    print("--- VALIDATING SCHEMA ---")
    schema = state.get("schema")
    if not schema:
        return {"error": "No schema present to validate"}
    
    # Basic validation logic
    if "collectionName" not in schema or "fields" not in schema:
        return {"error": "Invalid schema structure: missing collectionName or fields"}
    
    for field in schema["fields"]:
        if "name" not in field or "type" not in field:
            return {"error": f"Invalid field definition: {field}"}
            
    return {"validation_results": "Schema is valid"}

# Node 3: Call Strapi Bridge
def call_strapi_bridge(state: AgentState):
    print("--- CALLING STRAPI BRIDGE ---")
    if state.get("error"):
        return state

    try:
        url = f"{STRAPI_BASE_URL}/api/ai-schema/create-collection"
        response = requests.post(url, json=state["schema"])
        response.raise_for_status()
        return {"api_response": response.json()}
    except Exception as e:
        return {"error": f"Bridge API request failed: {str(e)}"}

# Define Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("generate_schema", generate_schema)
workflow.add_node("validate_schema", validate_schema)
workflow.add_node("call_strapi_bridge", call_strapi_bridge)

# Set Entry Point
workflow.set_entry_point("generate_schema")

# Define Edges
workflow.add_edge("generate_schema", "validate_schema")
workflow.add_edge("validate_schema", "call_strapi_bridge")
workflow.add_edge("call_strapi_bridge", END)

# Compile
app = workflow.compile()

if __name__ == "__main__":
    print("\nAI Strapi Schema Agent (LangGraph)")
    print("====================================")
    user_input = input("\nDescribe the collection you want to create: ")
    
    initial_state = {
        "user_query": user_input,
        "schema": None,
        "validation_results": None,
        "api_response": None,
        "error": None
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    print("\n--- FINAL RESULT ---")
    if result.get("error"):
        print(f"ERROR: {result['error']}")
    elif result.get("api_response"):
        print(f"SUCCESS: {json.dumps(result['api_response'], indent=2)}")
    else:
        print("Workflow finished with no response.")
