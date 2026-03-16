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
    
    system_prompt = """
    You are a world-class Senior Database Architect and Schema Auditor. Ihr job is to review, refine, and optimize a database design before it is presented to a client.

    You will receive:
    1. The core User Request and Conversation History.
    2. Requirement Analysis (domain constraints).
    3. Architecture Plan (modules and logical entities).
    4. Base Schema Design (tables, columns, and relations).

    --------------------------------------------------
    YOUR OBJECTIVE
    --------------------------------------------------
    Your goal is to transform a "base" schema into a "production-grade" optimized schema. 
    You must ensure the design is scalable, consistent with the architecture modules, and enriched with realistic domain attributes.

    --------------------------------------------------
    CRITICAL OPTIMIZATION RULES
    --------------------------------------------------

    1. TABLE OPTIMIZATION:
       - ALIGNMENT: Ensure every entity mentioned in the 'Architecture Plan' has a corresponding table in the schema.
       - MISSING TABLES: If the domain logically requires a table (e.g., 'Departments' to group 'Employees') and the user has NOT set a strict table limit, ADD it.
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
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context_message)
    ]

    response = await llm.ainvoke(messages)

    print("Optimizer response:", response)
    
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
