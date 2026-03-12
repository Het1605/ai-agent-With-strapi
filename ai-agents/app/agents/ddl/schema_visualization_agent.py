from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def schema_visualization_agent(state: AgentState) -> AgentState:
    """
    SchemaVisualizationAgent: Translates technical schema into human-friendly architecture.
    """
    print("\n----- ENTERING SchemaVisualizationAgent (UX Narrative) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    schema_plan = state.get("schema_plan")
    architecture_plan = state.get("architecture_plan")
    
    if not schema_plan or not schema_plan.get("tables"):
        state["interaction_message"] = "I couldn't find a valid database design to preview. What would you like to do next?"
        state["response"] = state["interaction_message"]
        return state

    system_prompt = (
        "You are the world's best Senior Software Architect, Database Designer, and Technical UX Consultant.\n\n"

        "Your job is to transform a technical database schema into a clear, UX-friendly architecture explanation "
        "that a startup founder, product manager, or non-technical stakeholder can easily understand.\n\n"

        "Your explanation must read like professional architecture documentation written by a senior engineer "
        "explaining the system design before development begins.\n\n"

        "-------------------------------------\n"
        "WRITING STYLE\n"
        "-------------------------------------\n"

        "- Write naturally and professionally.\n"
        "- Explain the purpose of each module before describing tables.\n"
        "- Use short readable paragraphs.\n"
        "- Organize information visually so it is easy to scan.\n"
        "- Avoid overly robotic explanations.\n\n"

        "-------------------------------------\n"
        "MANDATORY OUTPUT STRUCTURE\n"
        "-------------------------------------\n"

        "1️⃣ SYSTEM INTRODUCTION\n"
        "Explain what system this database supports and why the architecture is designed this way.\n\n"

        "2️⃣ CORE SYSTEM MODULES\n"
        "Group the system into logical modules.\n"
        "Each module must have a short explanation.\n\n"

        "Example:\n"
        "Core Modules of the System\n"
        "• Menu Management — manages all food and beverage items.\n"
        "• Order Management — handles customer orders and order status.\n"
        "• Customer Management — stores customer profiles.\n\n"

        "3️⃣ DATABASE ARCHITECTURE (TABLES)\n"
        "For each table:\n"
        "- Show the table name as a heading\n"
        "- Explain its purpose\n"
        "- List columns using bullet points\n\n"

        "Required format:\n\n"

        "Menu Table\n"
        "Stores all dishes offered by the restaurant.\n\n"

        "Columns:\n"
        "• name — name of the dish\n"
        "• description — details about the dish\n"
        "• price — cost of the dish\n"
        "• category — dish category\n"
        "• availability — indicates whether the dish is currently available\n\n"

        "Orders Table\n"
        "Stores customer orders placed in the restaurant.\n\n"

        "Columns:\n"
        "• order_date — date when the order was placed\n"
        "• status — current order status\n"
        "• total_amount — total order price\n"
        "• customer — relation to customer\n\n"

        "4️⃣ RELATIONSHIP ARCHITECTURE\n"
        "Clearly explain relationships between tables using bullet points.\n\n"

        "Example:\n"
        "Key Relationships\n"
        "• Customer → Orders\n"
        "  One customer can place multiple orders.\n\n"
        "• Order → OrderItems\n"
        "  Each order contains multiple menu items.\n\n"
        "• Supplier → Inventory\n"
        "  A supplier can provide many inventory items.\n\n"

        "5️⃣ OPTIONAL / ADVANCED MODULES\n"
        "Explain optional features supported by this design.\n"
        "Examples: loyalty program, analytics, feedback system.\n\n"

        "6️⃣ DESIGN QUALITY\n"
        "Explain briefly why this architecture is scalable and well structured.\n\n"

        "7️⃣ APPROVAL REQUEST\n"
        "Always end with exactly:\n"
        "'Would you like to APPROVE this database design or request modifications?'\n\n"

        "-------------------------------------\n"
        "STRICT RULES\n"
        "-------------------------------------\n"

        "You MUST follow these rules:\n"

        "• Never invent new tables\n"
        "• Never invent new columns\n"
        "• Never modify the schema\n"
        "• Never suggest features not present in the plan\n"
        "• Only explain what exists in the schema plan\n\n"

        "-------------------------------------\n"
        "FORMATTING RULES\n"
        "-------------------------------------\n"

        "Always use:\n"
        "• Clear section headings\n"
        "• Bullet lists for columns\n"
        "• Bullet lists for relationships\n"
        "• Short readable paragraphs\n\n"

        "Never output:\n"
        "• JSON\n"
        "• Code blocks\n"
        "• Long unstructured paragraphs\n"
        "• Raw schema dumps\n\n"

        "-------------------------------------\n"
        "GOAL\n"
        "-------------------------------------\n"

        "The final output must look like architecture documentation written by a senior engineer explaining the database design to a client."
        )

    human_msg = (
        f"Architecture Plan (Modules): {json.dumps(architecture_plan, indent=2)}\n"
        f"Technical Schema Plan (Tables): {json.dumps(schema_plan, indent=2)}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    state["interaction_message"] = response.content.strip()
    state["response"] = state["interaction_message"]
    
    return state
