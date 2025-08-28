from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.utils.llm_factory import LLMFactory
from src.utils.memory_manager import MemoryManager
import pandas as pd
import json
from typing import Dict, Any, List

class AnalysisAgent:
    def __init__(self):
        self.llm = LLMFactory.get_gemini_llm()
        self.memory = MemoryManager("analysis_memory")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an analysis specialist. Your task is to:
1. Process and analyze research data
2. Identify patterns and trends
3. Generate insights and conclusions
4. Create structured analysis reports

Focus on data-driven insights and clear conclusions."""),
            ("human", "{task}")
        ])
    
    def execute(self, task: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute analysis task"""
        # Get relevant context
        context = self.memory.get_context(task)
        
        # Prepare analysis data
        analysis_data = data or {}
        
        analysis_prompt = f"""
Analysis Task: {task}

Available Data:
{json.dumps(analysis_data, indent=2)}

Relevant Context:
{context}

Provide detailed analysis with:
1. Key findings
2. Data patterns
3. Trends identified
4. Conclusions
5. Recommendations
"""
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        
        # Store analysis in memory
        metadata = {
            "agent": "analysis",
            "task": task,
            "data_sources": len(analysis_data) if analysis_data else 0
        }
        
        self.memory.store_research(response.content, metadata)
        
        return {
            "agent": "analysis",
            "task": task,
            "analysis": response.content,
            "processed_data": analysis_data
        }
    
    def analyze_research_findings(self, research_results: List[Dict]) -> Dict[str, Any]:
        """Analyze research findings from multiple sources"""
        # Extract safe data without circular references
        safe_data = {
            "total_sources": len(research_results),
            "findings_preview": []
        }
        
        # Extract findings safely
        for result in research_results:
            findings = result.get("findings", "")
            if findings:
                safe_data["findings_preview"].append(findings[:300] + "..." if len(findings) > 300 else findings)
        
        return self.execute("Analyze research findings", safe_data)