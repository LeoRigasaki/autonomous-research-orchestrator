from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.utils.llm_factory import LLMFactory
from typing import Dict, Any, List

class SupervisorAgent:
    def __init__(self):
        print("DEBUG: Initializing supervisor agent...")
        self.llm = LLMFactory.get_groq_llm()
        print("DEBUG: LLM initialized")
        
        # Initialize specialist agents with lazy loading
        self._research_agent = None
        self._analysis_agent = None
        self._summary_agent = None
        self._memory_agent = None
        print("DEBUG: Supervisor initialization complete")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the supervisor coordinating a team of specialist agents:
- Research Agent: Finds academic papers and current information
- Analysis Agent: Processes data and identifies patterns
- Summary Agent: Creates reports and summaries  
- Memory Agent: Manages context and historical data

Plan the workflow and coordinate agents to complete the user's request."""),
            ("human", "{request}")
        ])
    
    def plan_workflow(self, request: str) -> List[str]:
        """Plan agent execution workflow based on request type"""
        # Analyze request type
        request_lower = request.lower()
        
        # Technical/protocol queries
        if any(term in request_lower for term in ["mcp", "protocol", "server", "api", "technical"]):
            return ["memory", "research", "analysis", "summary"]
        
        # Research-heavy queries
        elif any(term in request_lower for term in ["trends", "research", "study", "analysis", "survey"]):
            return ["memory", "research", "analysis", "summary"]
        
        # Quick factual queries
        elif any(term in request_lower for term in ["what is", "define", "explain", "how"]):
            return ["memory", "research", "summary"]
        
        # Complex analysis queries
        elif any(term in request_lower for term in ["analyze", "compare", "evaluate", "assess"]):
            return ["memory", "research", "analysis", "summary"]
        
        # Default comprehensive workflow
        else:
            return ["memory", "research", "analysis", "summary"]
    
    @property
    def research_agent(self):
        if self._research_agent is None:
            from src.agents.research_agent import ResearchAgent
            self._research_agent = ResearchAgent()
        return self._research_agent
    
    @property
    def analysis_agent(self):
        if self._analysis_agent is None:
            from src.agents.analysis_agent import AnalysisAgent
            self._analysis_agent = AnalysisAgent()
        return self._analysis_agent
    
    @property
    def summary_agent(self):
        if self._summary_agent is None:
            from src.agents.summary_agent import SummaryAgent
            self._summary_agent = SummaryAgent()
        return self._summary_agent
    
    @property
    def memory_agent(self):
        if self._memory_agent is None:
            from src.agents.memory_agent import MemoryAgent
            self._memory_agent = MemoryAgent()
        return self._memory_agent
    
    def execute_workflow(self, request: str) -> Dict[str, Any]:
        """Execute complete workflow"""
        print(f"DEBUG: Planning workflow for: {request}")
        workflow = self.plan_workflow(request)
        print(f"DEBUG: Planned workflow: {workflow}")
        
        results = []
        
        # Execute agents in planned sequence
        for agent_name in workflow:
            print(f"DEBUG: Executing {agent_name} agent...")
            
            if agent_name == "memory":
                result = self.memory_agent.execute(request)
            elif agent_name == "research":
                result = self.research_agent.execute(request)
            elif agent_name == "analysis":
                # Pass safe previous results to analysis
                safe_results = [{"agent": r.get("agent"), "summary": str(r.get("findings", r.get("context_analysis", "")))[:200]} for r in results]
                result = self.analysis_agent.execute(request, {"previous_results": safe_results})
            elif agent_name == "summary":
                # Create final report
                result = self.summary_agent.create_final_report(results)
            
            print(f"DEBUG: {agent_name} agent completed")
            results.append(result)
        
        print("DEBUG: All agents completed")
        return {
            "request": request,
            "workflow": workflow,
            "results": results,
            "status": "completed"
        }