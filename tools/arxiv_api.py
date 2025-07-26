import arxiv
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class ArxivAPI:
    """arXiv API client with rate limiting and error handling."""
    
    def __init__(self, delay_seconds: float = 3.0, max_results: int = 10):
        """
        Initialize arXiv API client.
        
        Args:
            delay_seconds: Delay between API calls (recommended 3+ seconds)
            max_results: Maximum papers to fetch per query
        """
        self.delay_seconds = delay_seconds
        self.max_results = max_results
        self.client = arxiv.Client()
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.delay_seconds:
            sleep_time = self.delay_seconds - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search_papers(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search for papers on arXiv.
        
        Args:
            query: Search query string
            max_results: Override default max results
            
        Returns:
            List of paper dictionaries with metadata
        """
        if max_results is None:
            max_results = self.max_results
        
        try:
            self._rate_limit()
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers = []
            for result in self.client.results(search):
                paper = {
                    'id': result.entry_id.split('/')[-1],
                    'title': result.title.strip(),
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary.strip(),
                    'published': result.published.isoformat(),
                    'updated': result.updated.isoformat() if result.updated else None,
                    'categories': result.categories,
                    'pdf_url': result.pdf_url,
                    'arxiv_url': result.entry_id,
                    'primary_category': result.primary_category
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error searching arXiv: {str(e)}")
            return []
    
    def search_by_category(self, category: str, days_back: int = 7, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search for recent papers in a specific category.
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'cs.LG')
            days_back: How many days back to search
            max_results: Override default max results
            
        Returns:
            List of paper dictionaries
        """
        if max_results is None:
            max_results = self.max_results
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format query with date range
        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}* TO {end_date.strftime('%Y%m%d')}*]"
        full_query = f"cat:{category} AND {date_query}"
        
        return self.search_papers(full_query, max_results)
    
    def search_by_author(self, author_name: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search for papers by a specific author.
        
        Args:
            author_name: Author's name
            max_results: Override default max results
            
        Returns:
            List of paper dictionaries
        """
        if max_results is None:
            max_results = self.max_results
        
        query = f"au:{author_name}"
        return self.search_papers(query, max_results)
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """
        Get a specific paper by its arXiv ID.
        
        Args:
            paper_id: arXiv paper ID (e.g., '2301.08727')
            
        Returns:
            Paper dictionary or None if not found
        """
        try:
            self._rate_limit()
            
            search = arxiv.Search(id_list=[paper_id])
            
            for result in self.client.results(search):
                return {
                    'id': result.entry_id.split('/')[-1],
                    'title': result.title.strip(),
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary.strip(),
                    'published': result.published.isoformat(),
                    'updated': result.updated.isoformat() if result.updated else None,
                    'categories': result.categories,
                    'pdf_url': result.pdf_url,
                    'arxiv_url': result.entry_id,
                    'primary_category': result.primary_category
                }
            
            return None
            
        except Exception as e:
            print(f"Error fetching paper {paper_id}: {str(e)}")
            return None
    
    def format_paper_summary(self, paper: Dict) -> str:
        """
        Format paper information for display.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Formatted string summary
        """
        authors_str = ", ".join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            authors_str += f" et al. ({len(paper['authors'])} authors)"
        
        return f"""
**{paper['title']}**
Authors: {authors_str}
Published: {paper['published'][:10]}
Categories: {', '.join(paper['categories'])}
arXiv ID: {paper['id']}

Abstract: {paper['summary'][:300]}{'...' if len(paper['summary']) > 300 else ''}

PDF: {paper['pdf_url']}
arXiv: {paper['arxiv_url']}
        """.strip()


# Tool function for CrewAI integration
def search_arxiv_papers(query: str, max_results: int = 10) -> str:
    """
    Tool function for CrewAI agents to search arXiv papers.
    
    Args:
        query: Research query
        max_results: Maximum number of papers to return
        
    Returns:
        Formatted string with paper results
    """
    arxiv_client = ArxivAPI(max_results=max_results)
    papers = arxiv_client.search_papers(query)
    
    if not papers:
        return f"No papers found for query: {query}"
    
    result = f"Found {len(papers)} papers for query: '{query}'\n\n"
    
    for i, paper in enumerate(papers, 1):
        result += f"## Paper {i}\n"
        result += arxiv_client.format_paper_summary(paper)
        result += "\n\n---\n\n"
    
    return result