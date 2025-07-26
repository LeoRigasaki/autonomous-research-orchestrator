from crewai import Agent
from crewai.tools import tool
from tools.arxiv_api import ArxivAPI, search_arxiv_papers
import os
import re
from typing import List, Dict


class SearchAgent:
    """Search Agent specialized in finding and retrieving academic papers."""
    
    def __init__(self):
        self.arxiv_client = ArxivAPI(delay_seconds=3.0, max_results=10)
    
    def create_agent(self, tools: list = None) -> Agent:
        """
        Create the Search Agent with specified tools.
        
        Args:
            tools: List of tools available to the agent
            
        Returns:
            Configured CrewAI Agent
        """
        if tools is None:
            tools = [self.create_arxiv_search_tool()]
        
        return Agent(
            role="Academic Paper Search Specialist",
            goal="Find the most relevant academic papers from arXiv based on research queries",
            backstory="""You are a specialized research librarian and information retrieval expert 
            with deep knowledge of academic databases, particularly arXiv. You have spent years 
            helping researchers find the most relevant papers for their work.
            
            Your expertise includes:
            - Understanding academic search strategies and Boolean logic
            - Knowledge of arXiv categories and paper classification systems  
            - Ability to construct effective search queries from natural language
            - Experience with paper ranking and relevance assessment
            - Understanding of academic paper structures and metadata
            
            You excel at translating research needs into precise search queries and can quickly 
            identify the most valuable papers from large result sets. You understand the importance 
            of recent work, seminal papers, and different types of contributions (theoretical, 
            empirical, survey, etc.).""",
            verbose=True,
            allow_delegation=False,
            tools=tools,
            max_iter=3,
            memory=True,
            llm="groq/llama-3.3-70b-versatile"
        )
    
    def create_arxiv_search_tool(self):
        """Create arXiv search tool for the agent."""
        
        @tool("arxiv_search")
        def arxiv_search_tool(query: str, max_results: int = 10) -> str:
            """
            Search arXiv for academic papers.
            
            Args:
                query: Search query string
                max_results: Maximum number of papers to return
                
            Returns:
                Formatted paper results
            """
            return search_arxiv_papers(query, max_results)
        
        return arxiv_search_tool
    
    def extract_search_terms(self, research_plan: str) -> List[str]:
        """
        Extract search terms from a research coordination plan.
        
        Args:
            research_plan: Formatted research plan from Lead Agent
            
        Returns:
            List of search terms to try
        """
        search_terms = []
        
        # Extract terms from common sections
        patterns = [
            r"Primary search terms?[:\-]\s*(.+?)(?:\n|$)",
            r"Key terms?[:\-]\s*(.+?)(?:\n|$)", 
            r"Main keywords?[:\-]\s*(.+?)(?:\n|$)",
            r"Search queries?[:\-]\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, research_plan, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Clean and split terms
                terms = [term.strip().strip('"-,') for term in match.split(',')]
                search_terms.extend([term for term in terms if term])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return unique_terms[:5]  # Limit to top 5 terms
    
    def refine_query(self, original_query: str) -> List[str]:
        """
        Generate multiple refined search queries from the original.
        
        Args:
            original_query: Original research query
            
        Returns:
            List of refined search queries
        """
        refined_queries = []
        
        # Clean the original query
        cleaned = re.sub(r'[^\w\s]', ' ', original_query.lower())
        words = [w for w in cleaned.split() if len(w) > 2]
        
        if len(words) >= 2:
            # Combinations of key terms
            refined_queries.append(' '.join(words[:3]))  # First 3 words
            refined_queries.append(' '.join(words[-3:]))  # Last 3 words
            
            # Add common academic variations
            if 'machine learning' in original_query.lower():
                refined_queries.append('machine learning')
                refined_queries.append('ML')
            
            if 'artificial intelligence' in original_query.lower():
                refined_queries.append('artificial intelligence')
                refined_queries.append('AI')
            
            if 'deep learning' in original_query.lower():
                refined_queries.append('deep learning')
                refined_queries.append('neural networks')
        
        # Add the original query as fallback
        refined_queries.append(original_query)
        
        # Remove duplicates
        return list(dict.fromkeys(refined_queries))[:3]
    
    def search_multiple_terms(self, terms: List[str], max_results_per_term: int = 5) -> List[Dict]:
        """
        Search for multiple terms and combine results.
        
        Args:
            terms: List of search terms
            max_results_per_term: Results per term
            
        Returns:
            Combined list of unique papers
        """
        all_papers = []
        seen_ids = set()
        
        for term in terms:
            papers = self.arxiv_client.search_papers(term, max_results_per_term)
            
            for paper in papers:
                if paper['id'] not in seen_ids:
                    seen_ids.add(paper['id'])
                    all_papers.append(paper)
        
        return all_papers
    
    def rank_papers_by_relevance(self, papers: List[Dict], query: str) -> List[Dict]:
        """
        Simple relevance ranking based on query terms in title/abstract.
        
        Args:
            papers: List of paper dictionaries
            query: Original search query
            
        Returns:
            Papers sorted by relevance score
        """
        query_terms = set(query.lower().split())
        
        for paper in papers:
            score = 0
            title_text = paper['title'].lower()
            abstract_text = paper['summary'].lower()
            
            # Score based on term matches
            for term in query_terms:
                if term in title_text:
                    score += 3  # Title matches are more important
                if term in abstract_text:
                    score += 1
            
            # Bonus for recent papers (last 2 years)
            try:
                from datetime import datetime
                pub_date = datetime.fromisoformat(paper['published'].replace('Z', '+00:00'))
                age_days = (datetime.now(pub_date.tzinfo) - pub_date).days
                if age_days < 730:  # 2 years
                    score += 2
            except:
                pass
            
            paper['relevance_score'] = score
        
        return sorted(papers, key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    def execute_comprehensive_search(self, query: str, research_plan: str = "") -> str:
        """
        Execute a comprehensive search strategy.
        
        Args:
            query: Original research query
            research_plan: Research plan from Lead Agent
            
        Returns:
            Formatted search results
        """
        all_papers = []
        
        # Extract terms from research plan if available
        if research_plan:
            planned_terms = self.extract_search_terms(research_plan)
            if planned_terms:
                all_papers.extend(self.search_multiple_terms(planned_terms, 3))
        
        # Fallback to query refinement
        if not all_papers:
            refined_queries = self.refine_query(query)
            all_papers.extend(self.search_multiple_terms(refined_queries, 4))
        
        if not all_papers:
            return f"No papers found for query: {query}"
        
        # Rank papers by relevance
        ranked_papers = self.rank_papers_by_relevance(all_papers, query)
        
        # Format results
        result = f"# Search Results for: '{query}'\n\n"
        result += f"Found {len(ranked_papers)} relevant papers\n\n"
        
        for i, paper in enumerate(ranked_papers[:10], 1):  # Top 10 papers
            result += f"## {i}. {paper['title']}\n"
            result += f"**Authors:** {', '.join(paper['authors'][:3])}"
            if len(paper['authors']) > 3:
                result += f" et al. ({len(paper['authors'])} total)"
            result += f"\n**Published:** {paper['published'][:10]}\n"
            result += f"**Categories:** {', '.join(paper['categories'])}\n"
            result += f"**arXiv ID:** {paper['id']}\n"
            result += f"**Relevance Score:** {paper.get('relevance_score', 0)}\n\n"
            
            # Truncated abstract
            abstract = paper['summary'][:250]
            if len(paper['summary']) > 250:
                abstract += "..."
            result += f"**Abstract:** {abstract}\n\n"
            result += f"**Links:** [PDF]({paper['pdf_url']}) | [arXiv]({paper['arxiv_url']})\n\n"
            result += "---\n\n"
        
        return result


def create_search_agent_tool():
    """Create a tool wrapper for Search Agent functionality."""
    
    search_agent = SearchAgent()
    
    @tool("comprehensive_paper_search")
    def comprehensive_search_tool(query: str, research_plan: str = "") -> str:
        """
        Execute comprehensive academic paper search.
        
        Args:
            query: Research query
            research_plan: Optional research plan context
            
        Returns:
            Formatted search results
        """
        return search_agent.execute_comprehensive_search(query, research_plan)
    
    return comprehensive_search_tool