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
    history = state.get("conversation_history", [])
    previous_plan = state.get("architecture_plan", {})
    stored_optional = state.get("optional_modules", [])
    user_modification = state.get("user_input", "") # The latest feedback
    
    system_prompt = """
            You are a world-class Enterprise Database Architect and System Designer.

            Your responsibility is to DESIGN or REVISE the high-level architecture of a database system.
            You must think like an experienced architect designing production systems used in real companies.


            --------------------------------------------------
            AUTHORITATIVE CONTEXT: EXISTING SCHEMA
            --------------------------------------------------

            You will receive a list of "Existing Collections" currently in the database.
            
            This is the Authoritative Source of Truth for the current system state.
            
            Rules for handling Existing Schema:
            1. NEVER recreate a table that already exists in 'existing_collections'.
            2. If the user asks for a table that already exists, treat it as a "modification" or "expansion" requirement, NOT a new creation.
            3. Always look for logical relationships between new entities you design and the entities already present in the database.
            4. If you decide to add a relationship to an existing table, list that existing table as part of your architecture if it helps clarify the design, but mark it clearly if your format allows (or just ensure it is referenced).


            --------------------------------------------------
            ARCHITECTURAL THINKING PROCESS
            --------------------------------------------------

            Before generating the architecture, reason internally through these steps:

            1. Analyze the 'Existing Collections' to understand the current foundation.
            2. Understand the new business requirements.
            3. Identify which requirements are already satisfied by existing tables.
            4. Identify new domain entities needed.
            5. Determine how new entities connect to existing ones.
            6. Group entities into logical modules (preserving existing ones).
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

            1. If the user asks for a table that exists in 'existing_collections':
               Intelligently respond that the table exists and propose adding the new required functionality to it.

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
    
    try:
        plan = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
        state["architecture_plan"] = plan
        state["optional_modules"] = plan.get("optional_modules", [])
        print(f"[PlanningAgent] Modules suggested: {len(plan.get('modules', []))} core, {len(plan.get('optional_modules', []))} optional.")
    except Exception as e:
        print(f"[PlanningAgent] Error: {e}")
        state["architecture_plan"] = {
            "system_domain": reqs.get("system_domain", "generic"),
            "modules": [{"name": "core", "entities": ["data_records"]}],
            "optional_modules": []
        }
        state["optional_modules"] = []
        
    return state
