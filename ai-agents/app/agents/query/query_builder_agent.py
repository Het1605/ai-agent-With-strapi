from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def query_builder_agent(state: AgentState) -> AgentState:
    """
    QueryBuilderAgent: AI agent that converts the structured schema_data into 
    the exact JSON payload for the Strapi bridge, ensuring no properties are lost.
    """
    print("\n----- ENTERING QueryBuilderAgent (Preservation Mode) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    schema_data = state.get("schema_data", {})
    table_name = schema_data.get("table_name", "untitled_collection")
    
    # Deterministic Naming Logic
    singular_name = table_name.lower().replace(" ", "_")
    plural_name = f"{singular_name}s"
    collection_name = plural_name
    display_name = singular_name.capitalize()
    
    system_prompt = (
        "You are a Strapi Field Mapping Expert. Your role is to convert a list of logical columns "
        "into the exact Strapi 'fields' array, ensuring NO properties are lost.\n\n"
        "CRITICAL INSTRUCTION:\n"
        "You must preserve 'min', 'max', 'default', 'unique', 'required', 'enum', 'relation', "
        "'target', 'allowedTypes', and any other custom attributes provided in the input.\n\n"
        "Expected Output Format:\n"
        "{\n"
        "  \"fields\": [\n"
        "    {\"name\": \"...\", \"type\": \"...\", \"required\": ..., \"min\": ..., \"max\": ..., ...}\n"
        "  ]\n"
        "}\n\n"
        "Output ONLY the final JSON object containing the 'fields' array."
    )
    
    human_msg = f"Columns to Transform: {json.dumps(schema_data.get('columns', []), indent=2)}"
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])
    
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
        fields = result.get("fields", [])
        
        # Construct final payload programmatically
        payload = {
            "collectionName": collection_name,
            "singularName": singular_name,
            "pluralName": plural_name,
            "displayName": display_name,
            "fields": fields
        }
        
        state["strapi_payload"] = payload
        print(f"QueryBuilderAgent: Generated deterministic payload for {collection_name} with {len(fields)} fields.")
        
    except Exception as e:
        print(f"Error in QueryBuilderAgent reasoning: {e}")
        state["strapi_payload"] = {}
        
    return state
