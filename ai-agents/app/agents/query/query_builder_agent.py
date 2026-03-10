from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json


async def query_builder_agent(state: AgentState) -> AgentState:
    """
    QueryBuilderAgent: Converts schema_data into the exact Strapi bridge payload.

    Handles:
      DDL_CREATE_TABLE  → POST /api/ai-schema/create-collection
      DDL_MODIFY_SCHEMA → POST /api/ai-schema/modify-schema  (operation-aware)
    """
    print("\n----- ENTERING QueryBuilderAgent -----")

    llm           = ChatOpenAI(model="gpt-4o", temperature=0)
    schema_data   = state.get("schema_data", {})
    ddl_operation = state.get("ddl_operation", "DDL_CREATE_TABLE")
    operation     = state.get("operation", "")       # sub-operation for modify-schema
    table_name    = schema_data.get("table_name", "untitled")

    # ── Singular name helper ──────────────────────────────────────────
    singular_name = table_name.lower().replace(" ", "_")
    if singular_name.endswith('s') and not singular_name.endswith('ss'):
        singular_name = singular_name[:-1]
    plural_name  = f"{singular_name}s"
    display_name = singular_name.capitalize()

    # ── DDL_CREATE_TABLE path ─────────────────────────────────────────
    if ddl_operation != "DDL_MODIFY_SCHEMA":
        system_prompt = (
            "You are a Strapi Field Mapping Expert. Convert the provided columns list "
            "into a Strapi-compatible 'fields' array.\n"
            "RULES: Preserve ALL properties (type, enum, relation, target, min, max, required, unique, default).\n"
            "Output ONLY valid JSON: {\"fields\": [...]}"
        )
        human_msg = f"Columns to map: {json.dumps(schema_data.get('columns', []), indent=2)}"
        response  = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg)
        ])
        try:
            clean  = response.content.replace("```json", "").replace("```", "").strip()
            fields = json.loads(clean).get("fields", [])
        except Exception as e:
            print(f"QueryBuilderAgent: Field mapping error — {e}")
            fields = []

        payload  = {
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
        return state

    # ── DDL_MODIFY_SCHEMA path — operation-specific ───────────────────
    endpoint = "http://strapi:1337/api/ai-schema/modify-schema"

    if operation == "add_column":
        # Use LLM to map columns to Strapi field format
        system_prompt = (
            "Convert the provided columns list into a Strapi-compatible 'fields' array.\n"
            "RULES: Preserve ALL properties.\n"
            "Output ONLY valid JSON: {\"fields\": [...]}"
        )
        human_msg = f"Columns to map: {json.dumps(schema_data.get('columns', []), indent=2)}"
        response  = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg)
        ])
        try:
            clean  = response.content.replace("```json", "").replace("```", "").strip()
            fields = json.loads(clean).get("fields", [])
        except Exception:
            fields = schema_data.get("columns", [])

        payload = {
            "operation":  "add_column",
            "collection": singular_name,
            "data":       {"fields": fields}
        }
        print(f"QueryBuilderAgent: ADD_COLUMN payload for '{singular_name}' — {len(fields)} field(s).")

    elif operation == "update_collection":
        if schema_data.get("delete") == True:
            data = {"delete": True}
        else:
            data = {}
            if schema_data.get("new_display_name"):
                data["displayName"] = schema_data["new_display_name"]

        payload = {
            "operation":  "update_collection",
            "collection": singular_name,
            "data":       data
        }
        print(f"QueryBuilderAgent: UPDATE_COLLECTION payload for '{singular_name}' — data={data}.")

    elif operation == "update_field":
        payload = {
            "operation":  "update_field",
            "collection": singular_name,
            "data": {
                "field":   schema_data.get("field_name", ""),
                "updates": schema_data.get("updates", {})
            }
        }
        print(f"QueryBuilderAgent: UPDATE_FIELD payload — field='{schema_data.get('field_name')}'.")

    elif operation == "delete_field":
        payload = {
            "operation":  "delete_field",
            "collection": singular_name,
            "data": {
                "field": schema_data.get("field_name", "")
            }
        }
        print(f"QueryBuilderAgent: DELETE_FIELD payload — field='{schema_data.get('field_name')}'.")

    else:
        # Unknown sub-operation — pass through what we have
        payload = {
            "operation":  operation,
            "collection": singular_name,
            "data":       schema_data
        }
        print(f"QueryBuilderAgent: Unknown modify operation '{operation}' — forwarding raw data.")

    state["strapi_payload"]  = payload
    state["strapi_endpoint"] = endpoint
    return state
