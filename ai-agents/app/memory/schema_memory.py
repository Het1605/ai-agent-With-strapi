import requests
from app.graph.state import AgentState

def load_field_registry():
    """
    Fetches the field capabilities and collection list from the Strapi runtime endpoint.
    """
    url = "http://strapi:1337/api/ai-schema/field-registry"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"SchemaMemory: Failed to fetch registry. Status: {response.status_code}")
            return {}
    except Exception as e:
        print(f"SchemaMemory: Error fetching field registry: {str(e)}")
        return {}

def attach_schema_memory_to_state(state: AgentState) -> AgentState:
    """
    Refreshes the schema memory (field registry and collections) on every request.
    """
    print("SchemaMemory: Refreshing schema context from Strapi...")
    full_registry = load_field_registry()
    
    state["field_registry"]     = full_registry.get("fields", {})
    state["existing_collections"] = full_registry.get("collections", [])
    
    if not isinstance(state["existing_collections"], list):
        print(f"SchemaMemory: Warning! 'collections' from registry is not a list: {type(state['existing_collections'])}")
        state["existing_collections"] = []

    print(f"[SchemaLoader] Total Collections Found: {len(state['existing_collections'])}")
    if state["existing_collections"]:
        print(f"[SchemaLoader] Collections: {', '.join(state['existing_collections'])}")
    else:
        print("[SchemaLoader] No collections found.")

    return state
