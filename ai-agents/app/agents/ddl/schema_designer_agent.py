import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def schema_designer_agent(state: AgentState) -> AgentState:
    """
    SchemaDesignerAgent: Generates the actual database schema (JSON).
    Focuses on Strapi-compatible types and normalized relations.
    """
    print("\n----- ENTERING SchemaDesignerAgent (Design) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    plan = state.get("architecture_plan", {})
    user_input = state.get("user_input", "") # For iterative modifications
    
    system_prompt = (
        "You are a Senior Database Designer.\n"
        "Convert the architecture plan into a detailed JSON schema.\n\n"
        "RULES:\n"
        "1. Do NOT include system fields like id, createdAt, updatedAt.\n"
        "2. Use Strapi-compatible types: string, text, integer, float, decimal, date, datetime, boolean, enumeration, email, password, json, media, relation.\n"
        "3. Relations must use this format:\n"
        "   {\"name\": \"fieldName\", \"type\": \"relation\", \"relation\": \"oneToMany|manyToOne|manyToMany|oneToOne\", \"target\": \"target_table\"}\n"
        "4. Output must be a stable, normalized JSON structure.\n\n"
        "SCHEMA FORMAT:\n"
        "{\n"
        "  \"tables\": [\n"
        "    {\n"
        "      \"table_name\": \"...\",\n"
        "      \"columns\": [\n"
        "        {\"name\": \"...\", \"type\": \"...\", \"required\": true/false, \"unique\": true/false, \"default\": null}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Respond ONLY with valid JSON."
    )
    
    human_msg = f"Plan: {json.dumps(plan)}\nUser Modification Request (if any): {user_input}"
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])
    
    try:
        schema = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        state["schema_plan"] = schema
        state["schema_ready"] = True
        print(f"[SchemaDesignerAgent] Generated {len(schema.get('tables', []))} tables.")
    except Exception as e:
        print(f"[SchemaDesignerAgent] Error: {e}")
        state["schema_ready"] = False
        
    return state
