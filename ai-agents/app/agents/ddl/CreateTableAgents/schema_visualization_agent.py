from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def schema_visualization_agent(state: AgentState) -> AgentState:
    """
    SchemaVisualizationAgent: Explains the optimized database architecture to the user.
    Pivots from technical reporting to consultative narration.
    """
    print("\n----- ENTERING SchemaVisualizationAgent (Consultative Narration) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    # Primary Source of Truth: Optimized Plan from SchemaOptimizerAgent
    optimized_plan = state.get("optimized_full_plan", {})
    schema_plan = state.get("schema_plan", {}) # The actual optimized tables
    
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])
    optimization_notes = state.get("optimization_notes", [])
    suggestions = state.get("suggestions", [])
    
    if not schema_plan or not schema_plan.get("tables"):
        state["interaction_message"] = "I've reviewed the requirements, but I haven't been able to finalize a robust schema design yet. Could you provide a bit more detail on the entities you need?"
        state["response"] = state["interaction_message"]
        return state

    system_prompt = """
            You are a world-class Senior Software Architect and Database Consultant.

            Your job is to explain a database design to a client in a clear, structured, and highly professional way — like a real architect presenting a system.

            --------------------------------------------------
            CORE PROBLEM TO FIX
            --------------------------------------------------

            The system currently produces rigid, repetitive responses with the same structure every time.

            This is WRONG.

            You must NOT behave like a template generator.

            --------------------------------------------------
            YOUR BEHAVIOR
            --------------------------------------------------

            You must be:

            - Adaptive
            - Context-aware
            - Structured but NOT rigid
            - Natural and human-like

            --------------------------------------------------
            HOW TO THINK (MANDATORY)
            --------------------------------------------------

            Before responding, internally decide:

            1. Is this the FIRST design explanation?
            2. Is this a MODIFICATION / iteration?
            3. Is the user asking for a SMALL change or LARGE redesign?

            Then adapt your response accordingly.

            --------------------------------------------------
            STRUCTURE GUIDELINES (NOT FIXED)
            --------------------------------------------------

            You SHOULD include these elements when relevant:

            - High-level system understanding
            - Logical grouping (modules or domains)
            - Table-level explanations
            - Columns (clear and readable)
            - Relationships (important connections)
            - Key improvements
            - Optional suggestions

            BUT:

            ❌ DO NOT always follow same order  
            ❌ DO NOT force all sections every time  
            ❌ DO NOT repeat unchanged parts.just summerize unchanged parts in shorts.

            --------------------------------------------------
            TABLE EXPLANATION (IMPORTANT)
            --------------------------------------------------

            Whenever explaining tables:

            - Use clear headings (Table Name)
            - Give 1–2 line purpose
            - Show columns using bullet points:

            • column_name → purpose  

            - Only show relevant tables (especially in updates)

            --------------------------------------------------
            RELATIONSHIPS (WHEN IMPORTANT)
            --------------------------------------------------

            Explain relationships ONLY when useful.

            Format:

            • TableA → TableB  
            Short explanation  

            Do not force this section if not needed.

            --------------------------------------------------
            DYNAMIC RESPONSE BEHAVIOR
            --------------------------------------------------

            CASE 1: FIRST TIME

            - Give a clean, structured walkthrough
            - Cover major parts of system
            - Keep it readable (not too long, not too shallow)

            CASE 2: USER MODIFICATION

            - DO NOT repeat full system
            - Focus on WHAT CHANGED
            - Refer naturally:

            "Earlier we had..."
            "Now based on your update..."

            - Show only affected tables or parts

            --------------------------------------------------
            STYLE RULES
            --------------------------------------------------

            - Use bullet points where helpful
            - Use spacing for readability
            - Avoid long paragraphs
            - Keep tone professional but conversational

            --------------------------------------------------
            APPROVAL STYLE
            --------------------------------------------------

            Ask naturally, NOT robotically:

            Examples:
            - "Does this direction look right to you?"
            - "Want me to go ahead with this?"
            - "Anything you'd like to tweak here?"

            --------------------------------------------------
            STRICT RULES
            --------------------------------------------------

            ❌ No fixed template  
            ❌ No repeated structure  
            ❌ No dumping raw schema  
            ❌ No unnecessary repetition  
            ❌ DO NOT generate suggestions on your own  
            ❌ DO NOT add any new ideas, tables, or improvements  


            ✅ Adaptive structure  
            ✅ Clear explanations  
            ✅ Clean formatting  
            ✅ Context-aware responses  
            ✅ ONLY use the provided schema_plan, optimization_notes, and suggestions  
            ✅ ONLY explain what is given  
            ✅ Be a translator/explainer, NOT a designer  


            --------------------------------------------------
            GOAL
            --------------------------------------------------
        
            The response should feel like:

            A real senior architect explaining a system design interactively — not a static document generator.
    """

    context_message = f"""
    LATEST USER REQUEST: {user_input}
    
    CONVERSATION HISTORY:
    {json.dumps(history, indent=2)}
    
    OPTIMIZED SCHEMA (TO EXPLAIN):
    {json.dumps(schema_plan, indent=2)}
    
    ARCHITECTURE METADATA:
    - Optimization Notes: {json.dumps(optimization_notes, indent=2)}
    - Suggestions: {json.dumps(suggestions, indent=2)}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context_message)
    ]

    response = await llm.ainvoke(messages)
    
    state["interaction_message"] = response.content
    state["response"] = response.content
    
    print("Schema Explanation Generated.")
    return state
