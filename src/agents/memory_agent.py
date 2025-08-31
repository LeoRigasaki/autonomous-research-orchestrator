from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.utils.llm_factory import LLMFactory
from src.utils.memory_manager import MemoryManager
from typing import Dict, Any, List
import json

class MemoryAgent:
    def __init__(self):
        self.llm = LLMFactory.get_groq_llm()
        self.memory = MemoryManager("context_memory")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a memory and context specialist. Your task is to:
1. Manage conversation context
2. Retrieve relevant historical information
3. Maintain session continuity
4. Optimize context for other agents

Ensure agents have the right context for their tasks."""),
            ("human", "{task}")
        ])
    
    def execute(self, task: str, session_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute memory management task"""
        # Get context specific to the current query
        context = self.memory.get_context(task, max_docs=5)
        
        # If no specific context, indicate this clearly
        if not context or context == "No relevant context found.":
            context_analysis = f"""
Task: {task}

No specific context found for this query. This appears to be a new research topic.
The system will proceed with fresh research and analysis.

Recommendation: Gather comprehensive information from research sources 
and build new knowledge base for future queries on this topic.
"""
        else:
            context_analysis = f"""
Task: {task}

Found relevant context from previous research:
{context}

This information can be used to inform current analysis and avoid 
duplicating previous research efforts.
"""
        
        # Store the task for future context
        metadata = {
            "agent": "memory",
            "task": task,
            "timestamp": str(session_data.get("timestamp", "unknown")) if session_data else "unknown"
        }
        
        self.memory.store_research(f"Query: {task}", metadata)
        
        return {
            "agent": "memory",
            "task": task,
            "context_analysis": context_analysis,
            "has_context": bool(context and context != "No relevant context found.")
        }
    
    def get_agent_context(self, agent_type: str, query: str) -> str:
        """Get context for specific agent type"""
        return self.memory.get_context(query, max_docs=3)