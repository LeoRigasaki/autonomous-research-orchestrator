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
        # Get context from current memory only
        context = self.memory.get_context(task, max_docs=5)
        
        context_prompt = f"""
Memory Management Task: {task}

Session Data:
{json.dumps(session_data, indent=2) if session_data else 'No session data'}

Available Context:
{context}

Provide context analysis and recommendations for task execution.
"""
        
        response = self.llm.invoke([HumanMessage(content=context_prompt)])
        
        # Store context management results
        metadata = {
            "agent": "memory",
            "task": task,
            "context_available": bool(context.strip())
        }
        
        self.memory.store_research(response.content, metadata)
        
        return {
            "agent": "memory",
            "task": task,
            "context_analysis": response.content,
            "has_context": bool(context.strip())
        }
    
    def get_agent_context(self, agent_type: str, query: str) -> str:
        """Get context for specific agent type"""
        return self.memory.get_context(query, max_docs=3)