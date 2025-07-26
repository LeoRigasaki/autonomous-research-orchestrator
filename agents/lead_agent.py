from crewai import Agent
from crewai.tools import tool
import os
from groq import Groq
from google.generativeai import configure, GenerativeModel


class LeadAgent:
    """Lead Agent for coordinating research activities."""
    
    def __init__(self):
        self.groq_client = None
        self.gemini_model = None
        self._setup_llms()
    
    def _setup_llms(self):
        """Initialize LLM clients with fallback support."""
        try:
            # Setup Groq as primary
            if os.getenv('GROQ_API_KEY'):
                self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                
            # Setup Gemini as fallback
            if os.getenv('GEMINI_API_KEY'):
                configure(api_key=os.getenv('GEMINI_API_KEY'))
                self.gemini_model = GenerativeModel('gemini-1.5-flash')
                
        except Exception as e:
            print(f"Warning: LLM setup error: {e}")
    
    def create_agent(self, tools: list = None) -> Agent:
        """
        Create the Lead Agent with specified tools.
        
        Args:
            tools: List of tools available to the agent
            
        Returns:
            Configured CrewAI Agent
        """
        return Agent(
            role="Research Coordinator",
            goal="Plan and coordinate academic research activities to find the most relevant papers",
            backstory="""You are an experienced academic research coordinator with expertise in 
            information retrieval and research methodology. You have worked with researchers across 
            multiple domains including AI, machine learning, computer science, and related fields.
            
            Your expertise includes:
            - Breaking down complex research queries into focused search strategies
            - Understanding academic paper structures and content
            - Identifying key terms, synonyms, and related concepts
            - Coordinating with specialized agents to gather comprehensive information
            
            You excel at understanding the intent behind research queries and can translate 
            broad questions into specific, actionable search strategies.""",
            verbose=True,
            allow_delegation=True,
            tools=tools or [],
            max_iter=3,
            memory=True,
            llm="groq/llama-3.3-70b-versatile"  # Primary LLM
        )
    
    def analyze_query(self, query: str) -> dict:
        """
        Analyze a research query to extract key components.
        
        Args:
            query: User's research question
            
        Returns:
            Dictionary with query analysis
        """
        analysis_prompt = f"""
        Analyze this research query and break it down into components:
        
        Query: "{query}"
        
        Please provide:
        1. Main research topic
        2. Key terms and concepts
        3. Related synonyms and alternative terms
        4. Suggested arXiv categories to search
        5. Potential search strategies
        6. Difficulty level (basic/intermediate/advanced)
        
        Format as JSON.
        """
        
        try:
            # Try Groq first
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": analysis_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1000
                )
                return {"analysis": response.choices[0].message.content, "source": "groq"}
            
            # Fallback to Gemini
            elif self.gemini_model:
                response = self.gemini_model.generate_content(analysis_prompt)
                return {"analysis": response.text, "source": "gemini"}
            
            else:
                return {"analysis": "LLM not available", "source": "none"}
                
        except Exception as e:
            return {"analysis": f"Error analyzing query: {str(e)}", "source": "error"}
    
    def create_search_strategy(self, query: str) -> dict:
        """
        Create a comprehensive search strategy for the given query.
        
        Args:
            query: Research query
            
        Returns:
            Dictionary with search strategy
        """
        strategy_prompt = f"""
        Create a search strategy for this research query: "{query}"
        
        Provide:
        1. Primary search terms (most important keywords)
        2. Secondary search terms (related concepts)
        3. arXiv categories to focus on (e.g., cs.AI, cs.LG, cs.CL)
        4. Boolean search combinations
        5. Filters to apply (date range, author types, etc.)
        6. Expected number of relevant papers
        
        Make the strategy specific and actionable for arXiv searches.
        """
        
        try:
            if self.groq_client:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": strategy_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=800
                )
                return {"strategy": response.choices[0].message.content, "source": "groq"}
            
            elif self.gemini_model:
                response = self.gemini_model.generate_content(strategy_prompt)
                return {"strategy": response.text, "source": "gemini"}
            
            else:
                return {"strategy": "LLM not available", "source": "none"}
                
        except Exception as e:
            return {"strategy": f"Error creating strategy: {str(e)}", "source": "error"}
    
    def coordinate_research(self, query: str) -> dict:
        """
        Main coordination function that plans the research approach.
        
        Args:
            query: User's research query
            
        Returns:
            Comprehensive research plan
        """
        # Analyze the query
        query_analysis = self.analyze_query(query)
        
        # Create search strategy
        search_strategy = self.create_search_strategy(query)
        
        # Generate coordination plan
        plan = {
            "original_query": query,
            "query_analysis": query_analysis,
            "search_strategy": search_strategy,
            "next_steps": [
                "Execute primary search with main terms",
                "Execute secondary searches with alternative terms",
                "Filter and rank results by relevance",
                "Analyze top papers for key insights",
                "Generate comprehensive summary"
            ],
            "estimated_papers": "10-20 relevant papers expected",
            "estimated_time": "5-10 minutes for complete analysis"
        }
        
        return plan


def create_lead_agent_tool():
    """Create a tool wrapper for the Lead Agent functionality."""
    
    lead_agent = LeadAgent()
    
    @tool("research_coordination")
    def research_coordination_tool(query: str) -> str:
        """
        Coordinate research planning for the given query.
        
        Args:
            query: Research question or topic
            
        Returns:
            Formatted research plan
        """
        plan = lead_agent.coordinate_research(query)
        
        result = f"""
# Research Coordination Plan

**Original Query:** {plan['original_query']}

## Query Analysis
{plan['query_analysis']['analysis']}

## Search Strategy  
{plan['search_strategy']['strategy']}

## Next Steps
{chr(10).join([f"- {step}" for step in plan['next_steps']])}

**Estimated Results:** {plan['estimated_papers']}
**Estimated Time:** {plan['estimated_time']}
        """
        
        return result.strip()
    
    return research_coordination_tool