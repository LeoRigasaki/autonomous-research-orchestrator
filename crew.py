from crewai import Crew, Task, Process
from crewai.tools import tool
from agents.lead_agent import LeadAgent, create_lead_agent_tool
from agents.search_agent import SearchAgent, create_search_agent_tool
from agents.analysis_agent import AnalysisAgent, create_analysis_agent_tool
from agents.summary_agent import SummaryAgent, create_summary_agent_tool, create_literature_review_tool
from tools.arxiv_api import search_arxiv_papers
from tools.chroma_memory import ChromaMemory, create_memory_search_tool, create_memory_store_tool
import os
from dotenv import load_dotenv


class ResearchOrchestrator:
    """Main orchestrator for the autonomous research system."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize agent classes
        self.lead_agent_class = LeadAgent()
        self.search_agent_class = SearchAgent()
        self.analysis_agent_class = AnalysisAgent()
        self.summary_agent_class = SummaryAgent()
        
        # Initialize memory system
        self.memory = ChromaMemory()
        
        # Initialize tools
        self.tools = self._setup_tools()
        
        # Create agents with tools
        self.lead_agent = self.lead_agent_class.create_agent(tools=[self.tools['research_coordination']])
        self.search_agent = self.search_agent_class.create_agent(tools=[
            self.tools['arxiv_search'], 
            self.tools['comprehensive_search'],
            self.tools['memory_store']
        ])
        self.analysis_agent = self.analysis_agent_class.create_agent(tools=[
            self.tools['paper_analysis'],
            self.tools['memory_search']
        ])
        self.summary_agent = self.summary_agent_class.create_agent(tools=[
            self.tools['report_generation'],
            self.tools['literature_review']
        ])
        
        # Create crew
        self.crew = self._create_crew()
    
    def _setup_tools(self) -> dict:
        """Setup all tools for the agents."""
        
        # Basic arXiv search tool
        @tool("arxiv_search")
        def arxiv_search_tool(query: str, max_results: int = 10) -> str:
            """Search arXiv database for academic papers using keywords and queries"""
            return search_arxiv_papers(query, max_results)
        
        return {
            'arxiv_search': arxiv_search_tool,
            'research_coordination': create_lead_agent_tool(),
            'comprehensive_search': create_search_agent_tool(),
            'paper_analysis': create_analysis_agent_tool(),
            'report_generation': create_summary_agent_tool(),
            'literature_review': create_literature_review_tool(),
            'memory_search': create_memory_search_tool(),
            'memory_store': create_memory_store_tool()
        }
    
    def _create_crew(self) -> Crew:
        """Create and configure the research crew."""
        
        return Crew(
            agents=[
                self.lead_agent, 
                self.search_agent, 
                self.analysis_agent, 
                self.summary_agent
            ],
            tasks=[],  # Tasks will be created dynamically
            process=Process.sequential,
            verbose=True,
            memory=True,
            planning=True
        )
    
    def create_comprehensive_research_tasks(self, query: str) -> list:
        """
        Create a comprehensive research workflow with all agents.
        
        Args:
            query: User's research query
            
        Returns:
            List of configured tasks
        """
        
        # Task 1: Research Planning
        planning_task = Task(
            description=f"""
            Analyze the research query and create a comprehensive search strategy.
            
            Query: "{query}"
            
            Your responsibilities:
            1. Break down the query into key components and themes
            2. Identify relevant search terms, synonyms, and related concepts
            3. Suggest arXiv categories and domains to focus on
            4. Create a structured, multi-phase search strategy
            5. Estimate expected results, timeline, and success criteria
            
            Provide a detailed plan that guides the entire research workflow.
            """,
            agent=self.lead_agent,
            expected_output="Comprehensive research plan with search strategy, key terms, categories, and execution roadmap"
        )
        
        # Task 2: Paper Discovery and Collection
        search_task = Task(
            description=f"""
            Execute comprehensive paper search and collection based on the research plan.
            
            Query: "{query}"
            
            Your responsibilities:
            1. Use the research plan to guide your search strategy
            2. Search arXiv using multiple relevant terms and approaches
            3. Find and collect the most relevant academic papers (aim for 15-25 papers)
            4. Rank papers by relevance and quality
            5. Store valuable papers in memory for future reference
            6. Filter out low-quality or irrelevant results
            
            Focus on finding recent, high-impact papers that directly address the research query.
            Prioritize papers from reputable authors and venues.
            """,
            agent=self.search_agent,
            expected_output="Curated collection of 15-25 relevant papers with titles, authors, abstracts, and relevance rankings",
            context=[planning_task]
        )
        
        # Task 3: Deep Analysis and Insights
        analysis_task = Task(
            description=f"""
            Perform deep analysis of the collected papers to extract insights and patterns.
            
            Query: "{query}"
            
            Your responsibilities:
            1. Analyze the collected papers for key themes and methodologies
            2. Identify novel contributions and breakthrough insights
            3. Extract methodological trends and technical approaches
            4. Identify research gaps and limitations in current work
            5. Compare and contrast different research approaches
            6. Assess the quality and impact potential of the research
            
            Provide comprehensive analysis that goes beyond surface-level summaries to reveal
            deeper insights, connections, and implications.
            """,
            agent=self.analysis_agent,
            expected_output="Detailed analysis covering themes, methodologies, contributions, gaps, and quality assessment",
            context=[search_task]
        )
        
        # Task 4: Synthesis and Report Generation
        synthesis_task = Task(
            description=f"""
            Synthesize all research findings into a comprehensive, well-structured report.
            
            Query: "{query}"
            
            Your responsibilities:
            1. Combine insights from planning, search, and analysis phases
            2. Create a professional research report with clear structure
            3. Include executive summary, key findings, and recommendations
            4. Highlight practical applications and future directions
            5. Provide actionable insights and next steps
            6. Format the report for both technical and non-technical audiences
            
            Generate a publication-quality report that demonstrates thorough research
            and provides genuine value to readers interested in this topic.
            """,
            agent=self.summary_agent,
            expected_output="Comprehensive research report with executive summary, detailed findings, analysis, and actionable recommendations",
            context=[planning_task, search_task, analysis_task]
        )
        
        return [planning_task, search_task, analysis_task, synthesis_task]
    
    def execute_comprehensive_research(self, query: str) -> dict:
        """
        Execute the complete research workflow with all agents.
        
        Args:
            query: User's research question
            
        Returns:
            Dictionary with comprehensive research results
        """
        try:
            # Create tasks for this specific query
            tasks = self.create_comprehensive_research_tasks(query)
            
            # Update crew with new tasks
            self.crew.tasks = tasks
            
            # Execute the research workflow
            result = self.crew.kickoff()
            
            # Store research session in memory
            papers_data = str(result)  # Simplified - in production, extract structured data
            session_id = self.memory.store_research_session(
                query=query,
                papers=[],  # Would extract from results
                analysis=str(result)
            )
            
            return {
                'status': 'success',
                'query': query,
                'result': result,
                'session_id': session_id,
                'tasks_completed': len(tasks),
                'research_type': 'comprehensive'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': str(e),
                'result': None
            }
    
    def execute_quick_research(self, query: str, max_papers: int = 10) -> dict:
        """
        Execute a quick research workflow focusing on speed.
        
        Args:
            query: Research query
            max_papers: Maximum papers to analyze
            
        Returns:
            Quick research results
        """
        try:
            # Quick search
            search_results = self.search_agent_class.execute_comprehensive_search(query)
            
            # Quick analysis
            analysis_results = self.analysis_agent_class.analyze_papers_content(search_results, "general")
            
            # Quick summary
            summary_results = self.summary_agent_class.generate_research_report(
                f"Query: {query}\n\nResults: {search_results}\n\nAnalysis: {analysis_results}",
                "brief"
            )
            
            return {
                'status': 'success',
                'query': query,
                'search_results': search_results,
                'analysis': analysis_results,
                'summary': summary_results,
                'type': 'quick_research'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': str(e),
                'result': None
            }
    
    def create_literature_review(self, topic: str, focus_area: str = "") -> dict:
        """
        Create a literature review on a specific topic.
        
        Args:
            topic: Research topic
            focus_area: Specific focus area
            
        Returns:
            Literature review results
        """
        try:
            # Search for papers
            search_results = self.search_agent_class.execute_comprehensive_search(topic)
            
            # Create literature review
            review = self.summary_agent_class.create_literature_review(search_results, focus_area)
            
            return {
                'status': 'success',
                'topic': topic,
                'focus_area': focus_area,
                'review': review,
                'type': 'literature_review'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'topic': topic,
                'error': str(e),
                'result': None
            }
    
    def get_memory_stats(self) -> dict:
        """Get statistics about stored research."""
        try:
            papers_stats = self.memory.get_collection_stats("research_papers")
            sessions_stats = self.memory.get_collection_stats("research_sessions")
            collections = self.memory.list_collections()
            
            return {
                'papers_stored': papers_stats.get('document_count', 0),
                'research_sessions': sessions_stats.get('document_count', 0),
                'collections': collections,
                'memory_status': 'active'
            }
        except Exception as e:
            return {
                'memory_status': 'error',
                'error': str(e)
            }
    
    def create_research_tasks(self, query: str) -> list:
        """
        Create tasks for the research workflow.
        
        Args:
            query: User's research query
            
        Returns:
            List of configured tasks
        """
        
        # Task 1: Research Planning
        planning_task = Task(
            description=f"""
            Analyze the research query and create a comprehensive search strategy.
            
            Query: "{query}"
            
            Your responsibilities:
            1. Break down the query into key components
            2. Identify relevant search terms and synonyms
            3. Suggest arXiv categories to focus on
            4. Create a structured search strategy
            5. Estimate expected results and timeline
            
            Provide a detailed plan that the Search Agent can execute effectively.
            """,
            agent=self.lead_agent,
            expected_output="Detailed research plan with search strategy, key terms, and execution approach"
        )
        
        # Task 2: Paper Search Execution
        search_task = Task(
            description=f"""
            Execute comprehensive paper search based on the research plan.
            
            Query: "{query}"
            
            Your responsibilities:
            1. Use the research plan to guide your search strategy
            2. Search arXiv using multiple relevant terms and approaches
            3. Find and retrieve the most relevant academic papers
            4. Rank papers by relevance to the original query
            5. Filter out low-quality or irrelevant results
            
            Focus on finding recent, high-quality papers that directly address the research query.
            Aim for 10-15 of the most relevant papers.
            """,
            agent=self.search_agent,
            expected_output="Comprehensive list of relevant papers with titles, authors, abstracts, and relevance rankings",
            context=[planning_task]
        )
        
        return [planning_task, search_task]
    
    def execute_research(self, query: str) -> dict:
        """
        Execute the complete research workflow.
        
        Args:
            query: User's research question
            
        Returns:
            Dictionary with research results
        """
        try:
            # Create tasks for this specific query
            tasks = self.create_research_tasks(query)
            
            # Update crew with new tasks
            self.crew.tasks = tasks
            
            # Execute the research workflow
            result = self.crew.kickoff()
            
            return {
                'status': 'success',
                'query': query,
                'result': result,
                'tasks_completed': len(tasks)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': str(e),
                'result': None
            }
    
    def quick_search(self, query: str, max_results: int = 5) -> dict:
        """
        Execute a quick search without full crew coordination.
        
        Args:
            query: Research query
            max_results: Maximum papers to return
            
        Returns:
            Quick search results
        """
        try:
            # Use search agent directly for quick results
            search_results = self.search_agent_class.execute_comprehensive_search(query)
            
            return {
                'status': 'success',
                'query': query,
                'result': search_results,
                'type': 'quick_search'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': str(e),
                'result': None
            }


def create_research_crew() -> ResearchOrchestrator:
    """
    Factory function to create a configured research crew.
    
    Returns:
        Configured ResearchOrchestrator instance
    """
    return ResearchOrchestrator()


def test_research_workflow(query: str = "machine learning for drug discovery"):
    """
    Test function to validate the research workflow.
    
    Args:
        query: Test research query
    """
    print(f"Testing research workflow with query: '{query}'")
    print("="*60)
    
    try:
        # Create research orchestrator
        orchestrator = create_research_crew()
        print("✅ Research crew created successfully")
        
        # Test quick search first
        print("\n🔍 Testing quick search...")
        quick_result = orchestrator.quick_search(query, max_results=3)
        
        if quick_result['status'] == 'success':
            print("✅ Quick search completed")
            print(f"Result preview: {quick_result['result'][:200]}...")
        else:
            print(f"❌ Quick search failed: {quick_result['error']}")
        
        # Test full research workflow
        print("\n🚀 Testing full research workflow...")
        full_result = orchestrator.execute_research(query)
        
        if full_result['status'] == 'success':
            print("✅ Full research workflow completed")
            print(f"Tasks completed: {full_result['tasks_completed']}")
            print(f"Result preview: {str(full_result['result'])[:300]}...")
        else:
            print(f"❌ Full workflow failed: {full_result['error']}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run test when script is executed directly
    test_research_workflow()