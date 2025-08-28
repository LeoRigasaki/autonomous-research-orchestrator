from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.utils.llm_factory import LLMFactory
from src.utils.memory_manager import MemoryManager
from typing import Dict, Any, List
import json

class SummaryAgent:
    def __init__(self):
        self.llm = LLMFactory.get_gemini_llm()
        self.memory = MemoryManager("summary_memory")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a summary specialist. Your task is to:
1. Create clear, concise summaries
2. Structure information logically
3. Highlight key points and insights
4. Generate executive summaries

Create professional reports suitable for stakeholders."""),
            ("human", "{task}")
        ])
    
    def execute(self, task: str, content: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute summary task"""
        # Get relevant context
        context = self.memory.get_context(task)
        
        summary_prompt = f"""
Summary Task: {task}

Content to Summarize:
{json.dumps(content, indent=2) if content else 'No content provided'}

Relevant Context:
{context}

Create a comprehensive summary with:
1. Executive Summary
2. Key Findings
3. Main Insights
4. Action Items (if applicable)
5. Conclusion

Format as a professional report.
"""
        
        response = self.llm.invoke([HumanMessage(content=summary_prompt)])
        
        # Store summary in memory
        metadata = {
            "agent": "summary",
            "task": task,
            "content_length": len(str(content)) if content else 0
        }
        
        self.memory.store_research(response.content, metadata)
        
        return {
            "agent": "summary",
            "task": task,
            "summary": response.content,
            "source_content": content
        }
    
    def create_final_report(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create final comprehensive report"""
        # Extract safe data without circular references
        safe_content = {
            "research_count": len([r for r in all_results if r.get("agent") == "research"]),
            "analysis_count": len([r for r in all_results if r.get("agent") == "analysis"]),
            "memory_count": len([r for r in all_results if r.get("agent") == "memory"]),
            "total_agents": len(all_results),
            "findings_summary": []
        }
        
        # Extract key findings safely
        for result in all_results:
            agent_type = result.get("agent", "unknown")
            if agent_type == "research":
                safe_content["findings_summary"].append(f"Research: {result.get('findings', 'No findings')[:200]}...")
            elif agent_type == "analysis":
                safe_content["findings_summary"].append(f"Analysis: {result.get('analysis', 'No analysis')[:200]}...")
        
        return self.execute("Create final comprehensive report", safe_content)