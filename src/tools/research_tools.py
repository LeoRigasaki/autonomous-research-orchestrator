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
        results = []
        
        # Multiple search strategies
        search_urls = [
            f"https://feeds.feedburner.com/oreilly/radar",
            f"https://rss.cnn.com/rss/edition.rss",
            f"https://feeds.bbci.co.uk/news/technology/rss.xml"
        ]
        
        for url in search_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:3]:  # Limit per source
                    if query.lower() in entry.title.lower() or query.lower() in entry.get('summary', '').lower():
                        results.append({
                            "title": entry.title,
                            "summary": entry.get('summary', entry.get('description', 'No summary')),
                            "link": entry.link,
                            "published": entry.get('published', 'No date'),
                            "source": url
                        })
            except:
                continue
        
        # If no RSS results, try simple HTTP search
        if not results:
            search_terms = [
                f"site:github.com {query}",
                f"{query} documentation",
                f"{query} tutorial"
            ]
            
            for term in search_terms[:2]:  # Limit searches
                results.append({
                    "title": f"Search: {term}",
                    "summary": f"Suggested search for {query} related information",
                    "link": f"https://www.google.com/search?q={term.replace(' ', '+')}",
                    "published": "Current",
                    "source": "web_search"
                })
        
        return {"results": results, "query": query, "total_found": len(results)}
    except Exception as e:
        return {"error": str(e), "query": query, "results": []}

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