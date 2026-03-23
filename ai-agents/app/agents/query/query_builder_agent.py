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
    
    state["execution_payloads"] = []

    print(f"schema_data: {schema_data}")

    # ── 1. DDL_MODIFY_SCHEMA path ────────────────────────────────────
    if ddl_operation == "DDL_MODIFY_SCHEMA":
        modify_design = state.get("modify_schema_design", {})
        operations = modify_design.get("operations", [])
        payloads = []

        print("modify_design:",modify_design)

        print("operations:",operations)
        
        for op in operations:
            intent = op.get("intent")
            table = op.get("table", "untitled").lower().replace("_", "-").replace(" ", "-") # Ensure kebab-case
            
            # Relation Resolution for columns
            if "columns" in op:
                for col in op["columns"]:
                    if col.get("type") == "relation" and col.get("target") and not col["target"].startswith("api::"):
                        raw_target = col["target"]
                        target_id = raw_target.lower().replace("_", "-").replace(" ", "-")
                        # Check existing collections for exact match
                        for existing in existing_collections:
                            if isinstance(existing, str) and (existing == target_id or existing == raw_target):
                                target_id = existing
                                break
                            elif isinstance(existing, dict) and (existing.get("singular_name") == target_id or existing.get("singular_name") == raw_target):
                                target_id = existing.get("singular_name")
                                break
                        col["target"] = f"api::{target_id}.{target_id}"
                        
                    if "changes" in col:
                        updates = col["changes"]
                        if updates.get("type") == "relation" and updates.get("target") and not updates["target"].startswith("api::"):
                            raw_target = updates["target"]
                            target_id = raw_target.lower().replace("_", "-").replace(" ", "-")
                            # Check existing collections for exact match
                            for existing in existing_collections:
                                if isinstance(existing, str) and (existing == target_id or existing == raw_target):
                                    target_id = existing
                                    break
                                elif isinstance(existing, dict) and (existing.get("singular_name") == target_id or existing.get("singular_name") == raw_target):
                                    target_id = existing.get("singular_name")
                                    break
                            updates["target"] = f"api::{target_id}.{target_id}"

            if intent == "add_column":
                print("Enter in add_column:")
                payloads.append({
                    "operation": "add_column",
                    "collection": table,
                    "data": {"fields": op.get("columns", [])}
                })

                print(payloads)

            elif intent == "delete_column":
                for col in op.get("columns", []):
                    payloads.append({
                        "operation": "delete_column",
                        "collection": table,
                        "data": {"field": col.get("name")}
                    })
            elif intent == "update_column":
                print("Enter in Update_column")
                for col in op.get("columns", []):
                    payloads.append({
                        "operation": "update_column",
                        "collection": table,
                        "data": {
                            "field": col.get("name"),
                            "updates": col.get("changes", {})
                        }
                    })
                print(payloads)
                
            elif intent == "update_collection":
                payloads.append({
                    "operation": "update_collection",
                    "collection": table,
                    "data": op.get("details", {})
                })

        # Sort payloads by priority: update_collection, add_column, update_column, delete_column
        priority = {"update_collection": 1, "add_column": 2, "update_column": 3, "delete_column": 4}
        payloads.sort(key=lambda p: priority.get(p.get("operation"), 99))
        
        state["execution_payloads"] = payloads
        print(f"[QueryBuilder] Generated {len(payloads)} MODIFY execution payload(s) in sorted order.")
        print("payload:",payloads)
        return state

    # ── 2. DDL_CREATE_TABLE path ─────────────────────────────────────
    tables = schema_data.get("tables", [])
    if not tables:
        state["execution_error"] = "No table data available."
        return state
    
    print("tables:",tables)

    payloads = []
    # Build a lookup set for existing singular names
    # Safety: handle both list of strings (actual) and list of dicts (fallback)
    existing_singulars = set()
    for item in existing_collections:
        if isinstance(item, str):
            existing_singulars.add(item)
        elif isinstance(item, dict):
            existing_singulars.add(item.get("singular_name"))
    
    print("existing_singulars:",existing_singulars)

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
        # Note: We now check against SLUG (kebab-case) for Strapi identity
        if slug in existing_singulars:
            print(f"[QueryBuilder] SAFETY SKIP: Table '{slug}' already exists in database. Skipping creation.")
            continue

        if singular == plural:
            state["execution_error"] = f"Naming conflict for table '{table_name}': singular_name and plural_name must be different (found '{singular}')."
            return state

        print(f"[QueryBuilder] Generating creation payload for '{slug}'...")

        # Pre-process columns: Resolve relation targets to full UIDs
        processed_columns = []
        for col in columns:
            if col.get("type") == "relation" and col.get("target"):
                target_table = col["target"]
                target_authoritative_id = None
                
                # Search in current batch using table_name, slug, or singular_name
                for t in tables:
                    print("Entering in for loop")
                    if t.get("table_name") == target_table or t.get("slug") == target_table or t.get("singular_name") == target_table:
                        print("Entering in If block")
                        target_authoritative_id = t.get("slug") # RULE 1: slug is authoritative
                        print("target_authoritative_id",target_authoritative_id)
                        break
                
                # Search in existing collections (Schema Awareness)
                if not target_authoritative_id:

                    print("Entering in second if block")
                    # The LLM might have provided the full UID (e.g. api::employee.employee)
                    clean_target = target_table
                    if target_table.startswith("api::"):
                        print("entering in 1st if block")
                        clean_target = target_table.split(".")[1]
                        
                    if clean_target in existing_singulars:
                        print("entering in 2nd if block")
                        target_authoritative_id = clean_target
                    
                    print("clean_target:",clean_target)
                    print("target_authoritative_id:",target_authoritative_id)

                if not target_authoritative_id:
                    print(f"[QueryBuilder] RELATION ERROR: Could not resolve target '{target_table}'")
                    state["execution_error"] = f"Relation Error: Could not resolve authoritative ID for target table '{target_table}'."
                    return state

                # Direct UID formatting using the authoritative SLUG (kebab-case)
                col["target"] = f"api::{target_authoritative_id}.{target_authoritative_id}"

                print("Uids:",col["target"])
            processed_columns.append(col)

        # Map to Strapi fields via LLM
        system_prompt = (
            "Convert technical columns to Strapi 'fields' array.\n"
            "RULES:\n"
            "1. PRESERVE all constraints: 'required', 'unique', 'minLength', 'maxLength', 'min', 'max', 'private', 'configurable', 'default'.\n"
            "2. CLEAN OUTPUT: Omit any attribute that is 'false', 'null', or missing from the input.\n"
            "3. RELATIONS: Use the 'relation', 'target', 'inversedBy', and 'mappedBy' fields exactly as provided.\n"
            "Output ONLY JSON: {\"fields\": [...]}"
        )
        human_msg = f"Table '{table_name}' Columns: {json.dumps(processed_columns)}"
        
        response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=human_msg)])
        try:
            print("Entering in Try Block:")

            fields = json.loads(response.content.strip().replace("```json", "").replace("```", "")).get("fields", [])

            print("fields:",fields)
        except:
            print("entering in Catch Block")
            fields = []
            

        payloads.append({
            "operation":      "create_collection",
            "collectionName": plural,
            "singularName":   slug, # RULE 2: singularName MUST be slug (kebab-case)
            "pluralName":     plural,
            "displayName":    display,
            "fields":         fields
        })

    state["execution_payloads"] = payloads
    print(f"[QueryBuilder] Generated {len(payloads)} execution payload(s).")

    print("payload:",payloads)

    return state


