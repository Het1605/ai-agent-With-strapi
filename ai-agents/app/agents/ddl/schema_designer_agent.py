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
    field_registry = state.get("field_registry", {})
    history = state.get("conversation_history", [])
    previous_schema = state.get("schema_plan", {})
    user_modification = state.get("user_input", "") # For iterative modifications
    
    system_prompt = (
        "You are a Senior Database Architect designing a production-grade database schema.\n\n"

        "Your task is to convert the architecture plan into a detailed database schema. "
        "If this is a REVISION, refer to the 'Previous Schema Plan' and 'Conversation History' to ensure consistency while applying the new architectural changes.\n\n"

        "YOUR CORE RESPONSIBILITY: NAMING STRATEGY\n"
        "You must decide the most linguistically correct and professional names for each table.\n"
        "1. table_name: Internal identifier (snake_case).\n"
        "2. singular_name: Correct English singular form (one record).\n"
        "3. plural_name: Correct English plural form (the collection name).\n"
        "4. slug: Kebab-case version for API usage.\n"
        "5. display_name: Human-readable title case.\n\n"

        "STRICT NAMING RULES:\n"
        "- ALWAYS use correct English pluralization (e.g., 'category' -> 'categories', 'person' -> 'people').\n"
        "- NEVER make singular_name and plural_name identical (e.g., if the word is 'Status', use 'status' for singular and 'statuses' for plural).\n"
        "- Singular represents ONE entity; Plural represents the COLLECTION.\n"
        "- Slug must be lowercase kebab-case.\n"
        "- Display name should be professional Title Case.\n\n"

        "EXAMPLES:\n"
        "- Category: {singular: 'category', plural: 'categories', slug: 'category', display: 'Category'}\n"
        "- Status: {singular: 'status', plural: 'statuses', slug: 'status', display: 'Status'}\n"
        "- Address: {singular: 'address', plural: 'addresses', slug: 'address', display: 'Address'}\n\n"

        "DESIGN PRINCIPLES:\n"
        "1. Each table should normally contain 8–10 realistic business columns unless the entity is very simple.also consider more column if needed\n"
        "2. Include meaningful business fields.\n"
        "3. Include reference fields for relations.\n"
        "4. Include status fields where appropriate.\n"
        "5. Include metadata fields where useful.\n"
        "6. Include lookup tables when needed.\n"
        "7. Include join tables for many-to-many relations.\n"
        "8. Avoid overly minimal schemas.\n"
        "9. Use the field registry as a reference for commonly used fields when designing tables.\n\n"

        "IMPORTANT RULES:\n"
        "- DO NOT include system fields like id, createdAt, updatedAt (Strapi generates them automatically).\n"
        "- Columns must represent real business data.\n"
        "- Relations must use the format below.\n\n"

        "VALID TYPES:\n"
        "string, text, integer, float, decimal, date, datetime, boolean, enumeration, email, password, json, media, relation\n\n"

        "RELATION FORMAT:\n"
        "{\"name\": \"fieldName\", \"type\": \"relation\", \"relation\": \"oneToMany|manyToOne|manyToMany|oneToOne\", \"target\": \"target_table\"}\n\n"

        "SCHEMA FORMAT:\n"
        "{\n"
        "  \"tables\": [\n"
        "    {\n"
        "      \"table_name\": \"...\",\n"
        "      \"singular_name\": \"...\",\n"
        "      \"plural_name\": \"...\",\n"
        "      \"slug\": \"...\",\n"
        "      \"display_name\": \"...\",\n"
        "      \"columns\": [\n"
        "        {\"name\": \"...\", \"type\": \"...\", \"required\": true/false, \"unique\": true/false, \"default\": null}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "Each table should include a realistic set of columns describing the entity. Use the field registry as inspiration for commonly used attributes.\n\n"

        "Respond ONLY with valid JSON."
    )
            
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    human_msg = (
        f"Conversation History:\n{history_str}\n\n"
        f"Current Architecture Plan: {json.dumps(plan)}\n"
        f"Previous Schema Plan: {json.dumps(previous_schema)}\n"
        f"User Modification Request: {user_modification}\n"
        f"Available Field Registry: {json.dumps(field_registry)}"
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])
    
    try:
        schema = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        
        # Validation Rule: singular != plural
        for table in schema.get("tables", []):
            s = table.get("singular_name")
            p = table.get("plural_name")
            if s and p and s == p:
                print(f"[SchemaDesignerAgent] WARNING: Singular == Plural for '{s}'. Fixing...")
                table["plural_name"] = f"{p}s" # Simple safety fixup

        state["schema_plan"] = schema
        state["schema_ready"] = True
        print(f"[SchemaDesignerAgent] Generated {len(schema.get('tables', []))} tables with AI-driven naming.")
        print("Schemas:", state["schema_plan"])
    except Exception as e:
        print(f"[SchemaDesignerAgent] Error: {e}")
        state["schema_ready"] = False
        
    return state
