from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.utils.llm_factory import LLMFactory
from src.tools.research_tools import search_arxiv, search_web
from src.utils.memory_manager import MemoryManager
from typing import Dict, Any

class ResearchAgent:
    def __init__(self):
        self.llm = LLMFactory.get_groq_llm()
        self.memory = MemoryManager("research_memory")
        self.tools = [search_arxiv, search_web]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research specialist. Your task is to:
1. Search for relevant academic papers and current information
2. Extract key insights and findings
3. Store important findings in memory
4. Return structured research results

Use available tools to gather information. Focus on credible sources."""),
            ("human", "{task}")
        ])
    
    def execute(self, task: str) -> Dict[str, Any]:
        """Execute research task"""
        # Get existing context
        context = self.memory.get_context(task)
        
        # Search ArXiv
        arxiv_results = search_arxiv.invoke({"query": task, "max_results": 3})
        
        # Search web
        web_results = search_web.invoke({"query": task})
        
        # Analyze findings
        analysis_prompt = f"""
Task: {task}

Existing Context:
{context}

ArXiv Results:
{arxiv_results}

Web Results:
{web_results}

Provide a structured analysis of findings with key insights.
"""
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        
        # Store findings in memory
        metadata = {
            "agent": "research",
            "task": task,
            "sources": len(arxiv_results) + len(web_results.get("results", []))
        }
        
        self.memory.store_research(response.content, metadata)
        
        return {
            "agent": "research",
            "task": task,
            "findings": response.content,
            "sources": {
                "arxiv": arxiv_results,
                "web": web_results
            }
        }