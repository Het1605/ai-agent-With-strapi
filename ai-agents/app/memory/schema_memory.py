import requests
from app.graph.state import AgentState

def load_field_registry():
    """
    Fetches the field capabilities from the Strapi runtime endpoint.
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
    Attaches the field registry to the shared state if not already present.
    """
    if not state.get("field_registry"):
        print("SchemaMemory: Loading field registry from Strapi...")
        state["field_registry"] = load_field_registry()
    return state
