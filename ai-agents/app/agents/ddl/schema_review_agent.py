from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def schema_review_agent(state: AgentState) -> AgentState:
    """
    SchemaReviewAgent: Uses LLM reasoning to perform architectural linting.
    Improves quality, normalizes naming, and validates relations.
    """
    print("\n----- ENTERING SchemaReviewAgent (Architectural Reasoning) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    schema_plan = state.get("schema_plan")
    
    if not schema_plan or not schema_plan.get("tables"):
        print("[SchemaReviewAgent] No schema plan found to review.")
        return state

    system_prompt = (
        "You are a Senior Database Architect and Quality Engineer.\n"
        "Your goal is to review and refine a proposed Strapi database schema plan.\n\n"
        "RESPONSIBILITIES:\n"
        "1. Architectural Quality: Ensure the schema is realistic and normalized.\n"
        "2. Naming Conventions: Enforce lowercase snake_case for all table and field names.\n"
        "3. Relation Integrity: Verify that relation 'target' UIDs point to either existing tables or tables defined in the plan.\n"
        "4. Redundancy Detection: Identify and merge duplicate tables or fields.\n"
        "5. Constraint Advice: Ensure reasonable fields (e.g. status) are present for system entities.\n\n"
        "JSON OUTPUT REQUIREMENTS:\n"
        "- Respond ONLY with the modified JSON schema_plan.\n"
        "- Maintain the structure: {\"tables\": [...]}\n\n"
        "DATABASE CONTEXT:\n"
        f"Existing Tables in DB: {state.get('existing_collections', [])}"
    )

    human_msg = f"Current Schema Plan:\n{json.dumps(schema_plan, indent=2)}"

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        state["schema_plan"] = json.loads(clean)
        print(state["schema_plan"])
        print(f"[SchemaReviewAgent] Schema plan refined by LLM.")
    except Exception as e:
        print(f"[SchemaReviewAgent] LLM parsing error: {e}. Keeping original plan.")

    return state
