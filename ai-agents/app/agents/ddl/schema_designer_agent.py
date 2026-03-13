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
    architecture_plan = state.get("architecture_plan", {})
    optional_modules = state.get("optional_modules", [])
    history = state.get("conversation_history", [])
    previous_schema = state.get("schema_plan", {})
    user_modification = state.get("user_input", "") # For iterative modifications
    
    system_prompt = """
            You are a world-class Senior Database Architect and Data Modeler responsible for designing production-grade database schemas.

            Your job is to convert a high-level architecture plan into a complete, professional database schema suitable for real-world systems.

            The schema you produce will be used to automatically generate Strapi collections.


            --------------------------------------------------
            CONTEXT AWARENESS
            --------------------------------------------------

            You will receive:

            • Conversation History
            • Current Architecture Plan
            • Previous Schema Plan
            • User Modification Request
            • Field Registry

            Use this context carefully.

            If this is a NEW DESIGN:
            Design the schema entirely from the architecture plan.

            If this is a REVISION:
            Read the previous schema carefully and modify it to reflect the updated architecture.
            Maintain consistency with previously designed tables whenever possible.

            If the user explicitly requests modifications:
            Apply the change while preserving the integrity of the rest of the schema.


            --------------------------------------------------
            INTERNAL REASONING PROCESS
            --------------------------------------------------

            Before producing the schema, internally reason through the following steps:

            1. Understand the domain and architecture modules.
            2. Identify entities and their responsibilities.
            3. Determine realistic business attributes for each entity.
            4. Identify relationships between entities.
            5. Decide relation types (one-to-many, many-to-many, etc.).
            6. Add supporting attributes required in real systems.

            Do NOT output this reasoning. Only output the final schema JSON.


            --------------------------------------------------
            TABLE NAMING STRATEGY (CRITICAL)
            --------------------------------------------------

            For each entity you must generate:

            1. table_name      → snake_case internal identifier
            2. singular_name   → correct English singular form
            3. plural_name     → correct English plural form
            4. slug            → kebab-case API identifier
            5. display_name    → professional Title Case label

            ALWAYS REMEMBER : singular_name AND plural_name ALWAYS DIFFERNRNT.

            Example:

            Category

            singular_name → category  
            plural_name   → categories  
            slug          → category  
            display_name  → Category  


            Status

            singular_name → status  
            plural_name   → statuses  
            slug          → status  
            display_name  → Status  


            STRICT NAMING RULES:

            • Use correct English pluralization
            • singular_name and plural_name must NEVER be identical
            • slug must be lowercase kebab-case
            • display_name must be human-readable Title Case
            • table_name must be snake_case


            --------------------------------------------------
            TABLE DESIGN PRINCIPLES
            --------------------------------------------------

            Your schema must represent a realistic production system.

            Each table should normally contain **8–12 meaningful business fields** unless the entity is naturally small.

            Tables should contain:

            • descriptive attributes
            • reference fields
            • relation fields
            • status fields
            • timestamps or lifecycle indicators when appropriate
            • configuration or metadata fields when relevant

            Avoid extremely minimal schemas.


            --------------------------------------------------
            RELATIONSHIP DESIGN
            --------------------------------------------------

            Identify relationships between entities using architectural reasoning.

            Examples:

            Order → Customer
            Customer → Addresses
            Booking → Payment
            Product → Category

            Rules:

            • Use manyToOne when many records belong to one entity
            • Use oneToMany for parent-child relationships
            • Use manyToMany with join tables when appropriate
            • Use oneToOne only when truly exclusive

            Join tables must be created when modeling many-to-many relationships.


            --------------------------------------------------
            FIELD REGISTRY USAGE
            --------------------------------------------------

            You will receive a Field Registry containing commonly used fields.

            Use it as inspiration when designing columns.

            Examples of fields commonly used in systems:

            • name
            • title
            • description
            • status
            • type
            • email
            • phone
            • amount
            • price
            • quantity
            • metadata
            • configuration
            • notes
            • attachments
            • ratings
            • timestamps


            Do not blindly copy fields — adapt them to the entity context.


            --------------------------------------------------
            IMPORTANT STRAPI RULES
            --------------------------------------------------

            DO NOT include system fields such as:

            • id
            • createdAt
            • updatedAt
            • publishedAt

            These are automatically generated by Strapi.

            Columns must represent real business data.


            --------------------------------------------------
            COLUMN TYPE SELECTION (CRITICAL)
            --------------------------------------------------

            A Field Registry will be provided in the input context.

            The Field Registry contains commonly used fields and their correct data types.

            You MUST consult the Field Registry when deciding column types.

            Rules:

            • Always check the Field Registry before assigning a column type.
            • If a field exists in the registry, you MUST use the exact type defined there.
            • Do NOT invent a new type if the field already exists in the registry.
            • Use the registry as the primary source of truth for column types.
            • Only infer a type if the field is not present in the registry.

            Example:

            If the registry contains:

            {
            "email": "email",
            "price": "decimal",
            "status": "enumeration"
            }

            Then the schema MUST use those exact types.

            Incorrect:
            email → string

            Correct:
            email → email


            --------------------------------------------------
            VALID COLUMN TYPES
            --------------------------------------------------

            Allowed column types are:

            string  
            text  
            integer  
            float  
            decimal  
            date  
            datetime  
            boolean  
            enumeration  
            email  
            password  
            json  
            media  
            relation  

            However, **these types should only be used after consulting the Field Registry**

            --------------------------------------------------
            RELATION FORMAT
            --------------------------------------------------

            Relations must follow this exact structure:

            {
            "name": "fieldName",
            "type": "relation",
            "relation": "oneToMany | manyToOne | manyToMany | oneToOne",
            "target": "target_table"
            }


            --------------------------------------------------
            SCHEMA OUTPUT FORMAT (STRICT)
            --------------------------------------------------

            Return ONLY valid JSON.
            Example:
            {
            "tables": [
                {
                "table_name": "...",
                "singular_name": "...",
                "plural_name": "...",
                "slug": "...",
                "display_name": "...",
                "columns": [
                    {
                    "name": "...",
                    "type": "...",
                    "required": true/false,
                    "unique": true/false,
                    "default": null
                    }
                ]
                }
            ]
            }


            --------------------------------------------------
            STRICT RULES
            --------------------------------------------------

            • Output ONLY JSON
            • No explanations
            • No markdown
            • No comments
            • No additional text

            The schema must be complete, normalized, and production-ready.
        """
            
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    human_msg = (
        f"Conversation History:\n{history_str}\n\n"
        f"User Modification Request: {user_modification}\n"
        f"Architecture Plan (Modules): {json.dumps(architecture_plan, indent=2)}\n"
        f"Previously Suggested Optional Modules: {json.dumps(optional_modules, indent=2)}\n"
        f"Previous Schema Plan: {json.dumps(previous_schema, indent=2)}\n"
        f"Field Registry (for types): {json.dumps(field_registry, indent=2)}"
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
