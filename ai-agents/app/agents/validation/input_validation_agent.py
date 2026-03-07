from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def input_validation_agent(state: AgentState) -> AgentState:
    """
    InputValidationAgent: Validates the user input using LLM reasoning.
    """
    print("----- ENTERING InputValidationAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_input = state.get("user_input", "")
    
    system_prompt = (
        "You are an Input Validation Agent. Your job is to check if a user's request is valid, "
        "meaningless, or empty. Respond ONLY with a JSON object: "
        '{"is_valid": boolean, "reasoning": "technical explanation", "suggested_error_message": "user-friendly explanation"}\n\n'
        "CRITICAL: Be lenient with greetings and casual conversation. Words like 'hellow', 'hi', or short phrases "
        "are VALID. Only mark as is_valid=false if the input is truly garbage, empty, or harmful."
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User input: '{user_input}'")
    ])

    
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
        
        state["analysis"] = f"Validation Result: {result.get('reasoning')}"
        state["validation_results"]["input_validation"] = {
            "is_valid": result.get("is_valid", False),
            "suggested_message": result.get("suggested_error_message")
        }
        
        print(f"Validation decision: {'Valid' if result.get('is_valid') else 'Invalid'}")
            
    except Exception as e:
        print(f"Error in validation agent: {e}")
        state["validation_results"]["input_validation"] = {"is_valid": True} 
        state["analysis"] = "Validation system error, defaulting to valid."
        
    return state
