from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def query_builder_agent(state: AgentState) -> AgentState:
    """
    QueryBuilderAgent: Converts schema_data into the exact Strapi bridge payload.
    Supports both DDL_CREATE_TABLE and DDL_MODIFY_SCHEMA operations.
    """
    print("\n----- ENTERING QueryBuilderAgent -----")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    schema_data    = state.get("schema_data", {})
    ddl_operation  = state.get("ddl_operation")
    table_name     = schema_data.get("table_name")

    system_prompt = (
        "You are a Strapi Field Mapping Expert. Convert the provided columns list "
        "into a Strapi-compatible 'fields' array.\n\n"
        "RULES:\n"
        "- Preserve ALL properties: type, enum, relation, target, min, max, required, unique, default, etc.\n"
        "- Do NOT add or remove any properties.\n"
        "- Output ONLY valid JSON: {\"fields\": [...]}"
    )

    human_msg = f"Columns to map: {json.dumps(schema_data.get('columns', []), indent=2)}"

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    print('query builder response',response)

    try:
        clean   = response.content.replace("```json", "").replace("```", "").strip()
        result  = json.loads(clean)
        fields  = result.get("fields", [])

        singular_name   = table_name.lower().replace(" ", "_")
        # Remove trailing 's' for simple plural → singular (matches Strapi bridge rule)
        if singular_name.endswith('s') and not singular_name.endswith('ss'):
            singular_name = singular_name[:-1]
        plural_name     = f"{singular_name}s"
        display_name    = singular_name.capitalize()

        if ddl_operation == "DDL_MODIFY_SCHEMA":
            # Modify-schema payload: collectionName is the singular name
            payload = {
                "collectionName": singular_name,
                "fields": fields
            }
            endpoint = "http://strapi:1337/api/ai-schema/modify-schema"
            print(f"QueryBuilderAgent: MODIFY payload for '{singular_name}' with {len(fields)} field(s).")
        else:
            # Create-collection payload (default)
            payload = {
                "collectionName": plural_name,
                "singularName":   singular_name,
                "pluralName":     plural_name,
                "displayName":    display_name,
                "fields":         fields
            }
            endpoint = "http://strapi:1337/api/ai-schema/create-collection"
            print(f"QueryBuilderAgent: CREATE payload for '{plural_name}' with {len(fields)} field(s).")

        state["strapi_payload"]  = payload
        state["strapi_endpoint"] = endpoint

    except Exception as e:
        print(f"QueryBuilderAgent: Error — {e}")
        state["strapi_payload"]  = {}
        state["strapi_endpoint"] = ""

    return state
