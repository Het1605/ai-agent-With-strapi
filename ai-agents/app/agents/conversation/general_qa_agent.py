from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def general_qa_agent(state: AgentState) -> AgentState:
    """
    GeneralQAAgent: Answering internal analysis for general questions.
    """
    print("----- ENTERING GeneralQAAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_input = state.get("user_input", "")
    
    system_prompt = (
        "Research and analyze the user's general knowledge question. "
        "Provide a comprehensive internal analysis of the facts and context. "
        "Do NOT speak to the user. Speak ONLY to the system with internal analysis."
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])
    
    state["analysis"] = response.content
    state["intent"] = "general_knowledge_retrieval"
    
    return state
