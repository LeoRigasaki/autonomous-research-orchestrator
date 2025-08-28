import arxiv
import requests
import feedparser
from langchain_core.tools import tool
from typing import List, Dict

@tool
def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """Search ArXiv for research papers"""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for result in client.results(search):
        results.append({
            "title": result.title,
            "authors": [str(author) for author in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "published": str(result.published),
            "categories": result.categories
        })
    return results

@tool  
def search_web(query: str) -> Dict:
    """Search web for current information"""
    try:
        # Using a simple news API for demonstration
        url = f"https://feeds.feedburner.com/oreilly/radar"
        feed = feedparser.parse(url)
        
        results = []
        for entry in feed.entries[:5]:
            if query.lower() in entry.title.lower() or query.lower() in entry.summary.lower():
                results.append({
                    "title": entry.title,
                    "summary": entry.summary,
                    "link": entry.link,
                    "published": entry.published
                })
        
        return {"results": results, "query": query}
    except Exception as e:
        return {"error": str(e), "query": query}

@tool
def get_paper_content(pdf_url: str) -> str:
    """Get paper content from PDF URL"""
    try:
        response = requests.get(pdf_url, timeout=30)
        if response.status_code == 200:
            return f"PDF content retrieved from {pdf_url}"
        else:
            return f"Failed to retrieve PDF: {response.status_code}"
    except Exception as e:
        return f"Error retrieving PDF: {str(e)}"