import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def schema_optimizer_agent(state: AgentState) -> AgentState:
    """
    SchemaOptimizerAgent: Refines and optimizes the database schema design.
    Acts as a Senior Architect reviewing the design before visualization.
    """
    print("\n----- ENTERING SchemaOptimizerAgent (Technical Audit) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])
    requirement_output = state.get("requirements", {}) # From RequirementAgent
    planner_output = state.get("architecture_plan", {}) # system_domain, modules, entities
    base_schema = state.get("schema_plan", {}) # Tables, columns, relations from SchemaDesignerAgent
    
    existing_collections = state.get("existing_collections", [])
    existing_schema_map = state.get("existing_schema_map", state.get("existing_schema", {}))
    
    system_prompt = """
    You are a world-class Senior Database Architect and Schema Auditor. Ihr job is to review, refine, and optimize a database design before it is presented to a client.

    You will receive:
    1. The core User Request and Conversation History.
    2. Requirement Analysis (domain constraints).
    3. Architecture Plan (modules and logical entities).
    4. Base Schema Design (tables, columns, and relations).
    5. Existing Collections (tables already in the database).
    6. Existing Schema Map (details of existing tables).

    --------------------------------------------------
    YOUR OBJECTIVE
    --------------------------------------------------
    Your goal is to transform a "base" schema into a "production-grade" optimized schema. 
    You must ensure the design is scalable, consistent with the architecture modules, enriched with realistic domain attributes, AND strictly integrated with existing collections.

    --------------------------------------------------
    CRITICAL SCHEMA AWARENESS RULES (MUST FOLLOW)
    --------------------------------------------------
    1. NEVER RECREATE EXISTING TABLES (SEMANTIC INTELLIGENCE)
       - Prevent NOT ONLY exact name duplication, BUT ALSO same PURPOSE / same SCHEMA duplication.
       - If a table already exists in `existing_collections`, DO NOT include it again in the optimized schema.
       - If a proposed table shares a similar PURPOSE or has 60-70% schema overlap with an existing table (e.g., 'attendance' vs 'employee_attendance', or 'salary' vs 'employee_salary'):
         ❌ DO NOT create the new table.
         ✅ MERGE redundant proposed tables, and ensure ONE source of truth by relying on the existing system table.
         ✅ If the proposed table is fundamentally the same concept, REMOVE it from `optimized_schema`.

    2. RELATION INTELLIGENCE
       - When optimizing relations connecting to existing entities, refer to `existing_schema_map`.
       - NEVER guess target tables. Ensure relations correctly target existing tables using exact names/slugs from `existing_schema_map`.

    3. STRICT RELATION DEPENDENCY ENFORCEMENT (NON-NEGOTIABLE)
       🚨 You MUST NEVER output a schema with broken or missing relations.

       MANDATORY VALIDATION PROCESS (perform internally before output):

       STEP 1: Build a COMPLETE list of all available tables:
         - All tables in 'existing_collections' (already in database)
         - + All tables in the proposed 'optimized_schema'

       STEP 2: For EACH relation column in the optimized schema, identify the 'target' table.

       STEP 3: CHECK — does that target exist in the complete list from STEP 1?
         ✔ YES → relation is valid, continue.
         ❌ NO → MISSING DEPENDENCY DETECTED.

       STEP 4: For EVERY missing dependency:
         ✅ AUTO-CREATE a minimal but valid table for the missing entity.
         ✅ Include realistic fields: name, email, status (as appropriate to the domain).
         ✅ Add it to 'optimized_schema'.

       STEP 5: REPEAT Steps 2-4 until ZERO missing relation targets remain.

       ⚠️ ANTI-ASSUMPTION RULE:
       DO NOT assume tables like 'user', 'customer', 'account' exist.
       They MUST be verified against 'existing_collections'. If NOT present → CREATE THEM.

       FINAL VALIDATION CHECKLIST (mandatory before output):
       ✔ All relation targets exist (in existing_collections OR optimized_schema)
       ✔ No broken references
       ✔ No dangling relations
       ✔ Schema is logically complete

    4. OPTIMIZATION SCOPE
       - Detect duplicate-purpose tables and remove them. Ensure there is only ONE source of truth per concept.
       - Improve column structure, naming consistency, relations between NEW tables, and performance-related enhancements.
       - 🚨 RESERVED KEYWORD: "status" is NOT ALLOWED as a field name. 
         - Instead of "status", you MUST use "{table_name}_status" (e.g., "order_status", "employee_status").
       - 🚨 NEVER use 'component' or 'dynamiczone' types. These are not supported by the dynamic bridge. If the designer suggested them, change them to 'json' or separate tables.
       - NEVER modify existing system tables or rename existing collections.

    --------------------------------------------------
    CRITICAL OPTIMIZATION RULES
    --------------------------------------------------

    1. TABLE OPTIMIZATION:
       - ALIGNMENT: Ensure every entity mentioned in the 'Architecture Plan' has a corresponding table in the schema (unless it already exists in the database).
       - MISSING TABLES: If the domain logically requires a table and it does not exist, ADD it.
       - REDUNDANCY: Remove or merge tables that appear redundant or don't align with the planner's modules.
       - RESPECT LIMITS: If the user explicitly requested a specific number of tables or specific entities, YOU MUST NOT exceed that limit or remove those entities.

    2. COLUMN OPTIMIZATION:
       - ENRICHMENT: Add common domain fields that a designer might miss (e.g., 'status', 'description', 'priority', 'slug', 'is_active').
       - TYPES: Improve data types where necessary (e.g., using 'enumeration' for fixed states instead of 'string').
       - RESERVED FIELDS: DO NOT add or modify 'id', 'created_at', or 'updated_at'. Strapi handles these automatically.
       - IRRELEVANT FIELDS: Remove columns that don't add value to the specific entity.

    3. CONSTRAINT OPTIMIZATION:
       - REQUIRED/UNIQUE: Ensure primary identifiers (like 'email', 'code', 'slug') are marked as 'unique' and 'required'.
       - VALIDATION: Add validation rules like 'minLength', 'min', 'max' where appropriate (e.g., min length for passwords, range for age/price).

    4. RELATIONSHIP OPTIMIZATION:
       - LOGICAL CONNECTIVITY: Add missing relations that are standard in the domain (e.g., linking Students to Departments).
       - SIMPLIFICATION: Avoid unnecessary join tables if a direct relation satisfies the domain requirements.
       - 🚨 CIRCULAR DEPENDENCY PREVENTION & STRICT ENFORCEMENT (HARD CONSTRAINT):
         - MANDATORY FINAL VALIDATION STEP: Before returning the schema, you MUST scan ALL tables and their relations to detect mutual references.
         - NO TWO TABLES CAN REFERENCE EACH OTHER: If Table A references Table B, Table B **MUST NOT** reference Table A.
         - RELATION PRUNING RULES:
           - If `Table A -> manyToOne -> Table B` and `Table B -> oneToMany -> Table A` exist:
             ✅ KEEP only the `manyToOne` in Table A.
             ❌ REMOVE the `oneToMany` from Table B.
           - If `Table A -> oneToOne -> Table B` and `Table B -> oneToOne -> Table A` exist:
             ✅ KEEP exactly ONE of these based on business logic.
             ❌ REMOVE the other.
           - Every relationship MUST exist only once and in only ONE direction (CHILD to PARENT).
         - EXPLICIT CORRECTION (Example):
           - WRONG: Department has `faculty` (oneToMany) AND Faculty has `department` (manyToOne).
           - RIGHT: ONLY Faculty has `department` (manyToOne). Department MUST have NO relation field to Faculty.

    5. SUGGESTIONS:
       - Provide forward-looking recommendations for future scalability or advanced features.
       - These should NOT be applied to the 'optimized_schema' automatically. List them in the 'suggestions' array.

    --------------------------------------------------
    OUTPUT STRUCTURE (STRICT JSON)
    --------------------------------------------------
    You must return a valid JSON object with the following keys:

    - system_domain: (string) The domain from the planner.
    - modules: (list) The planner modules (can be adjusted if you added/removed tables).
    - base_schema: (object) The original schema you received.
    - optimized_schema: (object) The refined schema after all your improvements.
    - optimization_notes: (list of strings) Brief, technical explanation of what you changed and why.
    - suggestions: (list of strings) Polite, consultant-level recommendations for the future.

    Ensure the 'optimized_schema' follows the same structure as the base schema (a 'tables' list).
    """

    context_message = f"""
    USER REQUEST: {user_input}
    
    CONVERSATION HISTORY:
    {json.dumps(history, indent=2)}
    
    REQUIREMENTS:
    {json.dumps(requirement_output, indent=2)}
    
    ARCHITECTURE PLAN:
    {json.dumps(planner_output, indent=2)}
    
    BASE SCHEMA DESIGN:
    {json.dumps(base_schema, indent=2)}

    EXISTING COLLECTIONS (DO NOT DUPLICATE THESE):
    {json.dumps(existing_collections, indent=2)}
    
    EXISTING SCHEMA MAP (USE FOR RELATIONS):
    {json.dumps(existing_schema_map, indent=2)}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context_message)
    ]

    response = await llm.ainvoke(messages)  

    print("Optimizer response:",response.content)
    
    try:
        # Strip markdown code blocks if present
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        # Update state with optimized results
        state["schema_plan"] = result.get("optimized_schema", base_schema)
        state["optimization_notes"] = result.get("optimization_notes", [])
        state["suggestions"] = result.get("suggestions", [])
        
        # We also store the full metadata for the visualizer
        state["optimized_full_plan"] = result 
        
        print(f"Optimizer Notes: {len(state['optimization_notes'])} changes applied.")
        print(f"Suggestions: {len(state['suggestions'])} recommendations generated.")
        
    except Exception as e:
        print(f"Error parsing Optimizer output: {str(e)}")
        # Fallback to base schema
        state["optimization_notes"] = ["Optimizer failed to parse; using base designer output."]
        state["suggestions"] = []
    
    return state
