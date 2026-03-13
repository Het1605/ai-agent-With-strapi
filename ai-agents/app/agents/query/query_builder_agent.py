import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.utils.naming import to_strapi_slug, get_strapi_uid

async def query_builder_agent(state: AgentState) -> AgentState:
    """
    QueryBuilderAgent: Converts schema_data into executable Strapi payloads.
    Lean version: delegates naming and networking.
    """
    print("\n----- ENTERING QueryBuilderAgent (Lean) -----")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    schema_data = state.get("schema_data", {})
    ddl_operation = state.get("ddl_operation", "DDL_CREATE_TABLE")
    operation = state.get("operation", "")
    
    state["execution_payloads"] = []

    print("schema_data",schema_data)

    # ── 1. DDL_MODIFY_SCHEMA path ────────────────────────────────────
    if ddl_operation == "DDL_MODIFY_SCHEMA":
        payload = await _handle_modify_schema_payload(state, llm, schema_data, operation)
        if payload:
            state["execution_payloads"] = [payload]
        return state

    # ── 2. DDL_CREATE_TABLE path ─────────────────────────────────────
    tables = schema_data.get("tables", [])
    if not tables:
        state["execution_error"] = "No table data available."
        return state

    payloads = []
    for table in tables:
        table_name = table.get("table_name", "untitled")
        columns = table.get("columns", [])

        # Direct metadata mapping from DesignerAgent
        singular = table.get("singular_name")
        plural = table.get("plural_name")
        slug = table.get("slug")
        display = table.get("display_name")

        # Minimal safety fallback (no naive singularization)
        if not all([singular, plural, slug, display]):
            slug = to_strapi_slug(table_name)
            singular = slug
            plural = slug
            display = table_name.replace("_", " ").title()

        # Pre-process columns: Resolve relation targets to full UIDs using schema names
        processed_columns = []
        for col in columns:
            if col.get("type") == "relation" and col.get("target"):
                # Try to find the target table's singular name in the schema
                target_table = col["target"]
                target_singular = to_strapi_slug(target_table) # fallback
                for t in tables:
                    if t.get("table_name") == target_table and t.get("singular_name"):
                        target_singular = t["singular_name"]
                        break
                col["target"] = get_strapi_uid(target_singular)
            processed_columns.append(col)

        # Map to Strapi fields via LLM
        system_prompt = "Convert columns to Strapi 'fields' array. Output ONLY JSON: {\"fields\": [...]}"
        human_msg = f"Table '{table_name}' Columns: {json.dumps(processed_columns)}"
        
        response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=human_msg)])
        try:
            fields = json.loads(response.content.strip().replace("```json", "").replace("```", "")).get("fields", [])
        except:
            fields = []

        payloads.append({
            "operation":      "create_collection",
            "collectionName": plural,
            "singularName":   singular,
            "pluralName":     plural,
            "displayName":    display,
            "fields":         fields
        })

    state["execution_payloads"] = payloads
    print(f"[QueryBuilder] Generated {len(payloads)} creation payload(s).")
    return state

async def _handle_modify_schema_payload(state, llm, schema_data, operation) -> dict:
    table_name = schema_data.get("table_name", "untitled")
    
    # Metadata mapping if available
    singular = schema_data.get("singular_name")
    if not singular:
        # Final fallback: lower-kebab the table name if no singular metadata
        singular = to_strapi_slug(table_name)

    if operation == "add_column":
        # Resolve UIDs for any new relations
        cols = schema_data.get("columns", [])
        for c in cols:
            if c.get("type") == "relation" and c.get("target"):
                # For modifications, we might not have the full table list to look up UIDs.
                # Use to_strapi_slug as a basic singular fallback for the target.
                target_table = c["target"]
                target_singular = to_strapi_slug(target_table)
                c["target"] = get_strapi_uid(target_singular)

        system_prompt = "Convert to Strapi 'fields' array. Output ONLY JSON: {\"fields\": [...]}"
        human_msg = f"Columns: {json.dumps(cols)}"
        response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=human_msg)])
        try:
            fields = json.loads(response.content.strip().replace("```json", "").replace("```", "")).get("fields", [])
        except:
            fields = cols
        return {"operation": "add_column", "collection": singular, "data": {"fields": fields}}

    elif operation == "update_collection":
        data = {"delete": True} if schema_data.get("delete") else {}
        if schema_data.get("new_display_name"):
            data["displayName"] = schema_data["new_display_name"]
        return {"operation": "update_collection", "collection": singular, "data": data}

    elif operation == "update_field":
        return {
            "operation": "update_field",
            "collection": singular,
            "data": {
                "field": schema_data.get("field_name", ""),
                "updates": schema_data.get("updates", {})
            }
        }

    elif operation == "delete_field":
        return {
            "operation": "delete_field",
            "collection": singular,
            "data": {"field": schema_data.get("field_name", "")}
        }

    return {"operation": operation, "collection": singular, "data": schema_data}
