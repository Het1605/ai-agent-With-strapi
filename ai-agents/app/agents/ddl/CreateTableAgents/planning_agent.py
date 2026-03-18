import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def planning_agent(state: AgentState) -> AgentState:
    """
    PlanningAgent: High-level architectural planning.
    Determines core and optional modules for the database.
    """
    print("\n----- ENTERING PlanningAgent (Architecture) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    reqs = state.get("requirements", {})
    field_registry = state.get("field_registry", {})
    existing_collections = state.get("existing_collections", [])
    existing_schema_map = state.get("existing_schema_map", state.get("existing_schema", {}))
    history = state.get("conversation_history", [])
    previous_plan = state.get("architecture_plan", {})
    stored_optional = state.get("optional_modules", [])
    user_modification = state.get("user_input", "") # The latest feedback

    print("Existing Collections:", existing_collections)
    print("Existing Schema Map:", existing_schema_map)
    
    
    system_prompt = """
            You are a world-class Enterprise Database Architect and System Designer.

            Your responsibility is to DESIGN or REVISE the high-level architecture of a database system.
            You must think like an experienced architect designing production systems used in real companies.


            --------------------------------------------------
            AUTHORITATIVE CONTEXT: EXISTING SCHEMA
            --------------------------------------------------

            You will receive a list of "Existing Collections" and an "Existing Schema Map".
            
            This is the Authoritative Source of Truth for the current system state.
            
            Rules for handling Existing Schema (STRICT DUPLICATION SAFETY):
            1. STRICT DUPLICATE PREVENTION: NEVER propose/recreate a table that exists in 'existing_collections'.
            2. SEMANTIC INTELLIGENCE: Before suggesting ANY new entity, compare its PURPOSE and concept against 'existing_schema_map'.
               - Does a table with a similar PURPOSE already exist? (e.g., employee_salary vs salary)
               - Does any existing table already store this type of data?
               - If YES: ❌ DO NOT propose the new table. ✅ Plan to REUSE and extend the existing table natively via relationships.
            3. TARGET RESOLUTION: ALWAYS check 'existing_schema_map' before suggesting entities. If an entity exists, DO NOT recreate it.
            4. Thinking Model: You are EXTENDING the system, NOT recreating from scratch. Only suggest entirely NEW domain entities.
            5. If the user asks for a table that already exists or something semantically identical, do not list it in the output entities.

            --------------------------------------------------
            ARCHITECTURAL THINKING PROCESS
            --------------------------------------------------

            Before generating the architecture, reason internally through these steps:

            1. Analyze the 'Existing Schema Map' to understand the current foundation.
            2. Understand the new business requirements.
            3. Identify which requirements are logically satisfied by existing tables.
            4. Identify ONLY the completely NEW domain entities needed.
            5. Determine how new entities connect to existing ones.
            6. Group NEW entities into logical modules. Make sure no entity in "modules" matches existing collections.
            7. Detect optional modules that enhance the system.

            Do NOT output this reasoning. Only output the final architecture plan.


            --------------------------------------------------
            CONTEXT HANDLING
            --------------------------------------------------

            You may receive:

            • Existing Collections (Schema Memory)
            • Conversation History  
            • Requirements  
            • Previous Architecture Plan  
            • Previously Suggested Optional Modules
            • User Modification Request  
            • Field Registry  

            Handle them as follows:

            If this is a NEW DESIGN in an EXISTING DATABASE:
            • Respect the existing collections.
            • Only propose NEW entities or MODIFICATIONS to existing ones.

            If this is a REVISION:
            • Read the previous architecture plan and optional modules carefully.
            • Apply the user modification request.
            • If the user asks for 'the optional modules' or 'suggested tables', PROMOTE them from optional to core.
            • Improve or expand the architecture if needed.


            --------------------------------------------------
            DESIGN PRINCIPLES
            --------------------------------------------------

            Follow professional database design principles:

            • Modular architecture  
            • Clear separation of concerns  
            • Normalized entities  
            • Reusable reference entities  
            • Extensible design  


            --------------------------------------------------
            ENTITY DESIGN GUIDELINES
            --------------------------------------------------

            Each module should contain multiple related entities.

            Systems should typically contain **10–30 tables** depending on complexity.

            Avoid extremely minimal systems.


            --------------------------------------------------
            FIELD REGISTRY USAGE
            --------------------------------------------------

            You will receive a Field Registry containing commonly used fields.
            Use it as inspiration to understand entity types and domain semantics.


            --------------------------------------------------
            USER MODIFICATION HANDLING (CRITICAL)
            --------------------------------------------------

            You will receive a "User Modification Request".

            You must interpret and apply the request intelligently.

            1. If the user asks for a feature on a table that exists in 'existing_collections', output the NEW supporting tables required. Do NOT output the existing table.

            2. If the user asks to add "those suggested tables":
               Check the 'Previously Suggested Optional Modules' memory and promote them.

            Never ignore user modification requests.

            --------------------------------------------------
            OUTPUT FORMAT (STRICT)
            --------------------------------------------------

            Return ONLY valid JSON in the following format:

            {
            "system_domain": "...",
            "modules": [
                {
                "name": "module_name",
                "entities": ["entity1", "entity2", "entity3"]
                }
            ],
            "optional_modules": [
                {
                "name": "optional_module",
                "entities": ["entityA", "entityB"]
                }
            ]
            }

            STRICT RULES:

            • Do NOT include explanations.
            • Do NOT include markdown.
            • Respond ONLY with valid JSON.
        """
    
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    human_msg = (
        f"Existing Collections in Database:\n{json.dumps(existing_collections, indent=2)}\n\n"
        f"Existing Schema Map:\n{json.dumps(existing_schema_map, indent=2)}\n\n"
        f"Conversation History:\n{history_str}\n\n"
        f"Requirements: {json.dumps(reqs)}\n"
        f"Previous Architecture Plan: {json.dumps(previous_plan)}\n"
        f"Previously Suggested Optional Modules: {json.dumps(stored_optional)}\n"
        f"User Modification Request: {user_modification}\n"
        f"Available Field Registry: {json.dumps(field_registry)}"
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    print("planner agent response:", response)
    
    try:
        plan = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        state["architecture_plan"] = plan
        state["optional_modules"] = plan.get("optional_modules", [])
        print("Fetch Plan:",plan)
    except Exception as e:
        print(f"[PlanningAgent] Error: {e}")
        state["architecture_plan"] = {
            "system_domain": reqs.get("system_domain", "generic"),
            "modules": [{"name": "core", "entities": ["data_records"]}],
            "optional_modules": []
        }
        state["optional_modules"] = []
        
    return state
