from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json


async def query_builder_agent(state: AgentState) -> AgentState:
    """
    QueryBuilderAgent: Converts schema_data into executable Strapi payloads.
    
    Now implements naming normalization (underscores to hyphens) for Strapi compatibility.
    """
    print("\n----- ENTERING QueryBuilderAgent -----")

    llm           = ChatOpenAI(model="gpt-4o", temperature=0)
    schema_data   = state.get("schema_data", {})
    ddl_operation = state.get("ddl_operation", "DDL_CREATE_TABLE")
    operation     = state.get("operation", "")
    
    state["execution_payloads"] = []

    # ── DDL_MODIFY_SCHEMA path ────────────────────────────────────────
    if ddl_operation == "DDL_MODIFY_SCHEMA":
        payload = await _handle_modify_schema_payload(state, llm, schema_data, operation)
        if payload:
            state["execution_payloads"] = [payload]
            state["strapi_payload"] = payload
        return state

    # ── DDL_CREATE_TABLE path (Senior Architect support) ─────────────
    tables = schema_data.get("tables", [])
    if not tables:
        if schema_data.get("table_name"):
            tables = [schema_data]
        else:
            state["execution_error"] = "No table data available in schema_data['tables'] to build creation payload."
            return state

    payloads = []
    for table in tables:
        table_name = table.get("table_name", "untitled")
        columns    = table.get("columns", [])

        # ── Naming Normalization (Fix Problem 2) ───────────────────────
        # Strapi requires hyphen-based slugs for content-type keys.
        # Conversion: employee_leave -> employee-leave
        raw_slug = table_name.lower().replace(" ", "_")
        normalized_slug = raw_slug.replace("_", "-")
        
        print(f"[QueryBuilder] Normalized slug: {table_name} → {normalized_slug}")

        # Singularization (basic)
        singular_name = normalized_slug
        if singular_name.endswith('s') and not singular_name.endswith('ss'):
            singular_name = singular_name[:-1]
            
        plural_name  = f"{singular_name}s"
        display_name = table_name.replace("_", " ").title()

        # ── Strapi Compatibility Safety Filter ─────────────────────────
        reserved_fields = {"id", "createdAt", "updatedAt", "publishedAt"}
        filtered_columns = [
            col for col in columns
            if col.get("name") not in reserved_fields
        ]

        system_prompt = (
            "You are a Strapi Field Mapping Expert. Convert the provided columns list "
            "into a Strapi-compatible 'fields' array.\n"
            "Output ONLY valid JSON: {\"fields\": [...]}"
        )
        human_msg = f"Columns to map for table '{table_name}': {json.dumps(filtered_columns, indent=2)}"
        
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg)
        ])
        
        try:
            clean  = response.content.replace("```json", "").replace("```", "").strip()
            fields = json.loads(clean).get("fields", [])
        except Exception as e:
            print(f"QueryBuilderAgent: Field mapping error for '{table_name}' — {e}")
            fields = []

        payloads.append({
            "operation":      "create_collection",
            "collectionName": plural_name,
            "singularName":   singular_name,
            "pluralName":     plural_name,
            "displayName":    display_name,
            "fields":         fields
        })

    state["execution_payloads"] = payloads
    if payloads:
        state["strapi_payload"] = payloads[0]
    
    state["strapi_endpoint"] = "http://strapi:1337/api/ai-schema/create-collection"
    
    print(f"QueryBuilderAgent: Generated {len(payloads)} executable payload(s).")
    return state


async def _handle_modify_schema_payload(state, llm, schema_data, operation) -> dict:
    table_name = schema_data.get("table_name", "untitled")
    # Normalize for modify operations too
    singular_name = table_name.lower().replace(" ", "_").replace("_", "-")
    if singular_name.endswith('s') and not singular_name.endswith('ss'):
        singular_name = singular_name[:-1]

    if operation == "add_column":
        system_prompt = "Convert to Strapi 'fields' array. Output ONLY JSON: {\"fields\": [...]}"
        human_msg = f"Columns: {json.dumps(schema_data.get('columns', []), indent=2)}"
        response  = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=human_msg)])
        try:
            clean  = response.content.replace("```json", "").replace("```", "").strip()
            fields = json.loads(clean).get("fields", [])
        except Exception:
            fields = schema_data.get("columns", [])

        return {
            "operation":  "add_column",
            "collection": singular_name,
            "data":       {"fields": fields}
        }

    elif operation == "update_collection":
        data = {"delete": True} if schema_data.get("delete") else {}
        if schema_data.get("new_display_name"):
            data["displayName"] = schema_data["new_display_name"]
        return {
            "operation":  "update_collection",
            "collection": singular_name,
            "data":       data
        }

    elif operation == "update_field":
        return {
            "operation":  "update_field",
            "collection": singular_name,
            "data": {
                "field":   schema_data.get("field_name", ""),
                "updates": schema_data.get("updates", {})
            }
        }

    elif operation == "delete_field":
        return {
            "operation":  "delete_field",
            "collection": singular_name,
            "data": {"field": schema_data.get("field_name", "")}
        }

    return {"operation": operation, "collection": singular_name, "data": schema_data}
