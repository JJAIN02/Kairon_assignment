
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.llms import OpenAI
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from typing import Dict, List
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI


# Set API keys (Ensure these are set in your environment variables) 
os.environ["TAVILY_API_KEY"] = "travily key"
os.environ["OPENAI_API_KEY"] = "openai key"
#keys are not exposed due to security issue


# Define system state
@dataclass
class AgentState:
    query: str
    search_results: List[str] = field(default_factory=list)
    final_answer: str = ""

# Define search agent using Tavily API
def web_search(query: str) -> List[str]:
    search_tool = TavilySearchResults()
    results = search_tool.run(query)
    if isinstance(results, list):  # Ensure it’s a list
        return [result.get("snippet", "") if isinstance(result, dict) else str(result) for result in results]
    return []

# Define answer generation agent using OpenAI
def generate_answer(query: str, search_results: List[str]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini")
    context = "\n".join(map(str, search_results))  # Convert all items to strings
    response = llm.invoke([
        SystemMessage(content="Use the following research data to answer the question."),
        HumanMessage(content=f"{query}\n\n{context}")
    ])
    return response.content


# Define research agent
def research_agent(state: Dict) -> Dict:
    query = state["query"]
    search_results = web_search(query)
    return {"query": query, "search_results": search_results, "final_answer": ""}

# Define answer drafting agent
def answer_agent(state: Dict) -> Dict:
    query = state["query"]
    search_results = state["search_results"]
    final_answer = generate_answer(query, search_results)
    return {"query": query, "search_results": search_results, "final_answer": final_answer}

# Create agent graph
graph = StateGraph(dict)  # Use dict instead of custom object
graph.add_node("research", research_agent)
graph.add_node("answer", answer_agent)
graph.set_entry_point("research")
graph.add_edge("research", "answer")
graph.add_edge("answer", END)
executor = graph.compile()

def run_agentic_system(query: str):
    initial_state = {"query": query, "search_results": [], "final_answer": ""}
    result = executor.invoke(initial_state)
    return result["final_answer"]

# Example usage
if __name__ == "__main__":
    query = "Impact of AI on modern software development"
    answer = run_agentic_system(query)
    print("Final Answer:", answer)