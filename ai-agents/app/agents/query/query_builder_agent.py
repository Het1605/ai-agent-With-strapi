import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def query_builder_agent(state: AgentState) -> AgentState:
    """
    QueryBuilderAgent: Converts schema_data into executable Strapi payloads.
    Lean version: delegates naming and networking.
    """
    print("\n----- ENTERING QueryBuilderAgent (Lean) -----")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    schema_data = state.get("schema_data", {})
    existing_collections = state.get("existing_collections", [])
    ddl_operation = state.get("ddl_operation", "DDL_CREATE_TABLE")
    operation = state.get("operation", "")
    
    state["execution_payloads"] = []

    print(f"schema_data: {schema_data}")
    print(f"existing_collections count: {len(existing_collections)}")

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
    # Build a lookup set for existing singular names
    # Safety: handle both list of strings (actual) and list of dicts (fallback)
    existing_singulars = set()
    for item in existing_collections:
        if isinstance(item, str):
            existing_singulars.add(item)
        elif isinstance(item, dict):
            existing_singulars.add(item.get("singular_name"))

    for table in tables:
        table_name = table.get("table_name", "untitled")
        singular = table.get("singular_name")
        plural = table.get("plural_name")
        slug = table.get("slug")
        display = table.get("display_name")
        columns = table.get("columns", [])

        # Absolute Authority Enforcement
        if not singular or not plural or not slug or not display:
            state["execution_error"] = f"Missing naming metadata for table '{table_name}'. Architect must provide singular_name, plural_name, slug, and display_name."
            return state

        # Final Safety Guard: Prevent duplicate creation if table exists in SchemaMemory
        if singular in existing_singulars:
            print(f"[QueryBuilder] SAFETY SKIP: Table '{singular}' already exists in database. Skipping creation.")
            continue

        if singular == plural:
            state["execution_error"] = f"Naming conflict for table '{table_name}': singular_name and plural_name must be different (found '{singular}')."
            return state

        print(f"[QueryBuilder] Generating creation payload for '{singular}'...")

        # Pre-process columns: Resolve relation targets to full UIDs
        processed_columns = []
        for col in columns:
            if col.get("type") == "relation" and col.get("target"):
                target_table = col["target"]
                target_singular = None
                
                # Search in current batch
                for t in tables:
                    if t.get("table_name") == target_table or t.get("slug") == target_table:
                        target_singular = t.get("singular_name")
                        break
                
                # Search in existing collections (Schema Awareness)
                if not target_singular:
                    if target_table in existing_singulars:
                        target_singular = target_table

                if not target_singular:
                    print(f"[QueryBuilder] RELATION ERROR: Could not resolve target '{target_table}'")
                    state["execution_error"] = f"Relation Error: Could not resolve singular_name for target table '{target_table}'."
                    return state

                # Direct UID formatting using the authoritative singular_name
                col["target"] = f"api::{target_singular}.{target_singular}"
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
    print(f"[QueryBuilder] Generated {len(payloads)} execution payload(s).")
    return state

async def _handle_modify_schema_payload(state, llm, schema_data, operation) -> dict:
    table_name = schema_data.get("table_name", "untitled")
    
    # Metadata mapping if available
    singular = schema_data.get("singular_name")
    if not singular:
        state["execution_error"] = f"Missing 'singular_name' for operation '{operation}' on table '{table_name}'."
        return None

    print(f"[QueryBuilder] Modification '{operation}' on collection '{singular}'")

    if operation == "add_column":
        # Resolve UIDs for any new relations
        cols = schema_data.get("columns", [])
        for c in cols:
            if c.get("type") == "relation" and c.get("target"):
                target_table = c["target"]
                print(f"[QueryBuilder] Relation Target: {target_table}")
                # We expect the target to be a known singular name (UID base)
                c["target"] = f"api::{target_table}.{target_table}"

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
            
        # Absolute Authority: Include pluralName update if provided
        if schema_data.get("new_plural_name"):
            data["pluralName"] = schema_data["new_plural_name"]
            
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
