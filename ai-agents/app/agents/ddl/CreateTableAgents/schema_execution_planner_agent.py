from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def schema_execution_planner_agent(state: AgentState) -> AgentState:
    """
    SchemaExecutionPlannerAgent: Pure Dependency-Based Ordering Engine.
    Its ONLY job is to determine the correct execution order of the optimized schema.
    It does NOT filter, skip, or modify the tables.
    """
    print("\n----- ENTERING SchemaExecutionPlannerAgent (Pure Ordering Engine) -----")

    schema_plan = state.get("schema_plan")
    
    if not schema_plan or not schema_plan.get("tables"):
        print("[SchemaExecutionPlannerAgent] No tables found in plan.")
        return state

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = """
                    You are a Senior Backend Deployment Engineer specialized in Database Execution Ordering.

                    Your ONLY responsibility is to determine the CORRECT EXECUTION ORDER of the tables provided.

                    --------------------------------------------------
                    🎯 YOUR CORE TASK
                    --------------------------------------------------
                    You MUST reorder the input tables so that parent tables (those that are targets of relations) appear BEFORE the dependent tables (those that contain the relation fields).

                    --------------------------------------------------
                    🚨 CRITICAL RULES
                    --------------------------------------------------
                    1. KEEP ALL TABLES: You MUST include EVERY table from the input in your output. 
                    2. NO FILTERING: Do NOT remove any tables, even if you think they exist or are duplicates. (That was handled by previous agents).
                    3. NO MODIFICATION: Do NOT change any field, column, or metadata within the table objects.
                    4. DEPENDENCY ANALYSIS:
                    - Identify columns where 'type' == "relation".
                    - Find the 'target' table for those relations.
                    - If Table A depends on Table B, then Table B MUST come BEFORE Table A.

                    --------------------------------------------------
                    🔁 SPECIAL CASES
                    --------------------------------------------------
                    - If no relations exist → Return the tables in their original order.
                    - If only one table exists → Return it as-is.
                    - Circular Dependency → If A depends on B and B depends on A, choose a stable order (e.g., the one with fewer total dependencies first).

                    --------------------------------------------------
                    OUTPUT
                    --------------------------------------------------
                    Return ONLY a valid JSON array of the tables in their correct execution order.
                    No markdown. No code blocks. No explanations. Just the raw JSON array.
            """

    human_msg = f"Tables to order by dependency:\n{json.dumps(schema_plan.get('tables'), indent=2)}"

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    try:
        # Strip potential markdown and parse
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        ordered_tables = json.loads(clean_content)

        print("clean_content:",clean_content)
        print("ordered_tables:",ordered_tables)

        # Update state with the ordered tables
        state["schema_data"] = {"tables": ordered_tables}
        state["schema_ready"] = True

        print(f"[SchemaExecutionPlannerAgent] Determined execution order for {len(ordered_tables)} tables.")
        for i, t in enumerate(ordered_tables, 1):
            print(f" {i}. {t.get('table_name', 'unknown')}")

    except Exception as e:
        print(f"[SchemaExecutionPlannerAgent] Order parsing error: {e}. Falling back to original plan order.")
        state["schema_data"] = schema_plan
        state["schema_ready"] = True

    return state
