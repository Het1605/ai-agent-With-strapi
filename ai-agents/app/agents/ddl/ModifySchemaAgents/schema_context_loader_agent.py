import httpx
from typing import Dict, Any
from app.graph.state import AgentState

async def schema_context_loader_agent(state: AgentState) -> AgentState:
    """
    SchemaContextLoaderAgent: Fetches the real existing structure of all collections 
    from Strapi and injects it into the state before operations are planned.
    """
    print("\n----- ENTERING SchemaContextLoaderAgent -----")
    
    base_url = "http://strapi:1337/api/ai-schema"
    existing_schema: Dict[str, Any] = {}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 1. Fetch available collections from field registry
            registry_res = await client.get(f"{base_url}/field-registry")
            if registry_res.status_code == 200:
                data = registry_res.json()
                collections = data.get("collections", [])

                print("collections:",collections)
                
                # 2. Extract schema structure for each collection
                for col_name in collections:
                    schema_res = await client.get(f"{base_url}/content-type-schema/{col_name}")
                    if schema_res.status_code == 200:
                        existing_schema[col_name] = schema_res.json()
                    else:
                        print(f"[SchemaContextLoader] Warning: Failed to fetch schema for {col_name}")
            else:
                print(f"[SchemaContextLoader] Error: Failed to fetch field registry: {registry_res.status_code}")
                
    except Exception as e:
        print(f"[SchemaContextLoader] Unexpected error fetching schema: {e}")
        
    state["existing_schema"] = existing_schema
    print("existing_schema:",existing_schema)
    print(f"[SchemaContextLoader] Loaded schema for {len(existing_schema)} collections.")
    
    return state
