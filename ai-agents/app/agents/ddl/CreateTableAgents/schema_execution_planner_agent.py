from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def schema_execution_planner_agent(state: AgentState) -> AgentState:
    """
    SchemaExecutionPlannerAgent: Production-grade execution order planner.
    Performs dependency analysis, duplicate filtering, and topological sorting.
    """
    print("\n----- ENTERING SchemaExecutionPlannerAgent (AI Planning) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    schema_plan = state.get("schema_plan")
    existing_collections = state.get("existing_collections", [])
    conversation_history = state.get("conversation_history", [])

    print(f"Existing collections for planning: {len(existing_collections)}")
    
    if not schema_plan or not schema_plan.get("tables"):
        print("[SchemaExecutionPlannerAgent] No tables found in plan.")
        return state

    tables = schema_plan.get("tables", [])

    # ── FAST PATH: Single table, no sorting needed ──
    if len(tables) == 1:
        slug = tables[0].get("slug", "")
        if slug in existing_collections:
            print(f"[SchemaExecutionPlannerAgent] Single table '{slug}' already exists. Skipping.")
            state["schema_data"] = {"tables": []}
            state["schema_ready"] = True
            return state
        
        print(f"[SchemaExecutionPlannerAgent] Single table '{slug}' — no sorting needed.")
        state["schema_data"] = {"tables": tables}
        state["schema_ready"] = True
        return state

    # ── PRE-FILTER: Remove tables that already exist ──
    filtered_tables = []
    for t in tables:
        slug = t.get("slug", "")
        singular = t.get("singular_name", "")
        table_name = t.get("table_name", "")
        if slug in existing_collections or singular in existing_collections or table_name in existing_collections:
            print(f"[SchemaExecutionPlannerAgent] SKIP existing: '{slug}'")
        else:
            filtered_tables.append(t)

    if not filtered_tables:
        print("[SchemaExecutionPlannerAgent] All tables already exist. Nothing to create.")
        state["schema_data"] = {"tables": []}
        state["schema_ready"] = True
        return state

    # ── CHECK: Any relations at all? ──
    has_relations = False
    for t in filtered_tables:
        for col in t.get("columns", []):
            if col.get("type") == "relation":
                has_relations = True
                break
        if has_relations:
            break

    if not has_relations:
        print("[SchemaExecutionPlannerAgent] No relations found — using original order.")
        state["schema_data"] = {"tables": filtered_tables}
        state["schema_ready"] = True
        print(f"[SchemaExecutionPlannerAgent] Created execution plan for {len(filtered_tables)} tables.")
        for i, t in enumerate(filtered_tables, 1):
            print(f" {i}. {t.get('table_name', 'unknown')}")
        return state

    # ── LLM-BASED TOPOLOGICAL SORTING ──
    # Build recent history context (last 5 messages for iteration awareness)
    recent_history = conversation_history[-5:] if conversation_history else []
    history_str = json.dumps(recent_history, indent=2) if recent_history else "[]"

    system_prompt = f"""
    
                    You are a Senior Backend Deployment Engineer.

                    Your ONLY job: take the input tables and return them in the CORRECT CREATION ORDER.

                    --------------------------------------------------
                    AUTHORITATIVE CONTEXT: EXISTING DATABASE
                    --------------------------------------------------
                    The following tables ALREADY EXIST in the database (DO NOT include these in output):
                    {json.dumps(existing_collections)}

                    --------------------------------------------------
                    CONVERSATION CONTEXT
                    --------------------------------------------------
                    Recent conversation (for iteration awareness):
                    {history_str}

                    --------------------------------------------------
                    YOUR RESPONSIBILITIES
                    --------------------------------------------------

                    1. DEPENDENCY ANALYSIS
                    - Analyze ALL 'relation' columns in each table.
                    - Identify which table each relation's 'target' points to.
                    - Map parent → child dependencies.

                    2. DUPLICATE PREVENTION
                    - Compare every table's 'slug', 'singular_name', and 'table_name' against the existing collections list.
                    - If a table already exists → REMOVE it from output entirely.

                    3. TOPOLOGICAL SORTING
                    - Parent/target tables MUST appear BEFORE dependent tables.
                    - If table A has a relation targeting table B, then B comes first (unless B already exists in database).
                    - If a relation targets a table that already exists in the database, that dependency is satisfied — no ordering constraint needed.

                    4. CIRCULAR DEPENDENCY RESOLUTION
                    - If A depends on B and B depends on A, choose the safest order (the one with fewer dependencies goes first).

                    --------------------------------------------------
                    STRICT RULES
                    --------------------------------------------------

                    ❌ Do NOT modify any table schema, columns, names, or metadata.
                    ❌ Do NOT add new tables.
                    ❌ Do NOT remove columns or fields.
                    ❌ Do NOT include tables that exist in the EXISTING DATABASE list.

                    ✅ ONLY reorder the tables.
                    ✅ PRESERVE every field exactly: table_name, slug, singular_name, plural_name, display_name, columns.

                    --------------------------------------------------
                    OUTPUT
                    --------------------------------------------------

                    Return ONLY a valid JSON array of the filtered and sorted table objects.
                    No markdown. No explanation. No code blocks. Just the raw JSON array.

            """

    human_msg = f"Tables to plan execution order for:\n{json.dumps(filtered_tables, indent=2)}"

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        sorted_tables = json.loads(clean)

        # Safety: validate each table still has required metadata
        valid_tables = []
        for t in sorted_tables:
            if t.get("slug") and t.get("singular_name") and t.get("plural_name"):
                # Final duplicate check
                if t["slug"] not in existing_collections:
                    valid_tables.append(t)
                else:
                    print(f"[SchemaExecutionPlannerAgent] POST-FILTER: Removed existing '{t['slug']}'")
            else:
                print(f"[SchemaExecutionPlannerAgent] WARNING: Table missing metadata, skipping: {t.get('table_name', 'unknown')}")

        state["schema_data"] = {"tables": valid_tables}
        state["schema_ready"] = True
        
        print(f"[SchemaExecutionPlannerAgent] Created execution plan for {len(valid_tables)} tables.")
        for i, t in enumerate(valid_tables, 1):
            print(f" {i}. {t.get('table_name', 'unknown')}")
            
    except Exception as e:
        print(f"[SchemaExecutionPlannerAgent] Plan parsing error: {e}. Falling back to pre-filtered plan.")
        state["schema_data"] = {"tables": filtered_tables}
        state["schema_ready"] = True

    return state
