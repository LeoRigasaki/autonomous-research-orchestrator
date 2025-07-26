import chromadb
from chromadb.config import Settings
import os
import json
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime


class ChromaMemory:
    """ChromaDB-based memory system for storing and retrieving paper information."""
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """
        Initialize ChromaDB memory system.
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collections = {}
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with persistence."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize persistent client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            # Fallback to in-memory client
            self.client = chromadb.Client()
    
    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """
        Get or create a collection with the specified name.
        
        Args:
            name: Collection name
            
        Returns:
            ChromaDB collection
        """
        if name not in self.collections:
            try:
                self.collections[name] = self.client.get_or_create_collection(name=name)
            except Exception as e:
                print(f"Error creating collection {name}: {e}")
                return None
        
        return self.collections[name]
    
    def store_papers(self, papers: List[Dict], collection_name: str = "research_papers") -> bool:
        """
        Store papers in ChromaDB with embeddings.
        
        Args:
            papers: List of paper dictionaries
            collection_name: Name of the collection to store papers
            
        Returns:
            Success status
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return False
            
            documents = []
            metadatas = []
            ids = []
            
            for paper in papers:
                # Create document text for embedding
                doc_text = f"{paper.get('title', '')}\n\n{paper.get('summary', '')}"
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    'title': paper.get('title', ''),
                    'authors': ', '.join(paper.get('authors', [])),
                    'published': paper.get('published', ''),
                    'categories': ', '.join(paper.get('categories', [])),
                    'arxiv_id': paper.get('id', ''),
                    'pdf_url': paper.get('pdf_url', ''),
                    'arxiv_url': paper.get('arxiv_url', ''),
                    'primary_category': paper.get('primary_category', ''),
                    'stored_at': datetime.now().isoformat()
                }
                metadatas.append(metadata)
                
                # Create unique ID
                paper_id = paper.get('id', '')
                if not paper_id:
                    # Generate ID from title hash
                    paper_id = hashlib.md5(paper.get('title', '').encode()).hexdigest()
                ids.append(paper_id)
            
            # Add to collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            print(f"Error storing papers: {e}")
            return False
    
    def search_papers(self, query: str, collection_name: str = "research_papers", 
                     n_results: int = 10) -> List[Dict]:
        """
        Search for similar papers using semantic search.
        
        Args:
            query: Search query
            collection_name: Collection to search in
            n_results: Number of results to return
            
        Returns:
            List of similar papers
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return []
            
            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            papers = []
            if results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    paper = {
                        'id': results['ids'][0][i],
                        'title': metadata.get('title', ''),
                        'authors': metadata.get('authors', '').split(', ') if metadata.get('authors') else [],
                        'summary': doc.split('\n\n', 1)[1] if '\n\n' in doc else doc,
                        'published': metadata.get('published', ''),
                        'categories': metadata.get('categories', '').split(', ') if metadata.get('categories') else [],
                        'arxiv_id': metadata.get('arxiv_id', ''),
                        'pdf_url': metadata.get('pdf_url', ''),
                        'arxiv_url': metadata.get('arxiv_url', ''),
                        'primary_category': metadata.get('primary_category', ''),
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'stored_at': metadata.get('stored_at', '')
                    }
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error searching papers: {e}")
            return []
    
    def get_paper_by_id(self, paper_id: str, collection_name: str = "research_papers") -> Optional[Dict]:
        """
        Retrieve a specific paper by its ID.
        
        Args:
            paper_id: Paper ID
            collection_name: Collection to search in
            
        Returns:
            Paper dictionary or None
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return None
            
            results = collection.get(ids=[paper_id])
            
            if results['documents'] and len(results['documents']) > 0:
                doc = results['documents'][0]
                metadata = results['metadatas'][0] if results['metadatas'] else {}
                
                return {
                    'id': results['ids'][0],
                    'title': metadata.get('title', ''),
                    'authors': metadata.get('authors', '').split(', ') if metadata.get('authors') else [],
                    'summary': doc.split('\n\n', 1)[1] if '\n\n' in doc else doc,
                    'published': metadata.get('published', ''),
                    'categories': metadata.get('categories', '').split(', ') if metadata.get('categories') else [],
                    'arxiv_id': metadata.get('arxiv_id', ''),
                    'pdf_url': metadata.get('pdf_url', ''),
                    'arxiv_url': metadata.get('arxiv_url', ''),
                    'primary_category': metadata.get('primary_category', ''),
                    'stored_at': metadata.get('stored_at', '')
                }
            
            return None
            
        except Exception as e:
            print(f"Error getting paper {paper_id}: {e}")
            return None
    
    def store_research_session(self, query: str, papers: List[Dict], 
                             analysis: str = "", collection_name: str = "research_sessions") -> str:
        """
        Store a complete research session.
        
        Args:
            query: Original research query
            papers: List of papers found
            analysis: Analysis results
            collection_name: Collection for research sessions
            
        Returns:
            Session ID
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return ""
            
            # Create session document
            session_doc = f"Research Query: {query}\n\nAnalysis: {analysis}"
            
            # Create session metadata
            session_id = hashlib.md5(f"{query}{datetime.now().isoformat()}".encode()).hexdigest()
            
            metadata = {
                'query': query,
                'paper_count': len(papers),
                'paper_ids': ', '.join([p.get('id', '') for p in papers]),
                'session_date': datetime.now().isoformat(),
                'has_analysis': bool(analysis)
            }
            
            # Store session
            collection.add(
                documents=[session_doc],
                metadatas=[metadata],
                ids=[session_id]
            )
            
            return session_id
            
        except Exception as e:
            print(f"Error storing research session: {e}")
            return ""
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Statistics dictionary
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return {}
            
            count = collection.count()
            
            return {
                'collection_name': collection_name,
                'document_count': count,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}
    
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def reset_collection(self, collection_name: str) -> bool:
        """
        Reset (delete and recreate) a collection.
        
        Args:
            collection_name: Collection to reset
            
        Returns:
            Success status
        """
        try:
            # Delete collection if it exists
            try:
                self.client.delete_collection(collection_name)
            except:
                pass  # Collection might not exist
            
            # Remove from cache
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            # Recreate collection
            self.get_or_create_collection(collection_name)
            
            return True
            
        except Exception as e:
            print(f"Error resetting collection {collection_name}: {e}")
            return False


def create_memory_search_tool():
    """Create memory search tool for agents."""
    memory = ChromaMemory()
    
    from crewai.tools import tool
    
    @tool("memory_search")
    def memory_search_tool(query: str, collection: str = "research_papers") -> str:
        """
        Search stored papers in memory using semantic similarity.
        
        Args:
            query: Search query for finding similar papers
            collection: Collection to search in (default: research_papers)
            
        Returns:
            Formatted results from memory search
        """
        papers = memory.search_papers(query, collection, n_results=5)
        
        if not papers:
            return f"No papers found in memory for query: {query}"
        
        result = f"Found {len(papers)} similar papers in memory:\n\n"
        
        for i, paper in enumerate(papers, 1):
            result += f"## {i}. {paper['title']}\n"
            result += f"**Authors:** {', '.join(paper['authors'])}\n"
            result += f"**Published:** {paper['published'][:10]}\n"
            result += f"**Similarity:** {paper['similarity_score']:.3f}\n"
            result += f"**Categories:** {', '.join(paper['categories'])}\n"
            result += f"**arXiv ID:** {paper['arxiv_id']}\n\n"
            
            # Truncated summary
            summary = paper['summary'][:200]
            if len(paper['summary']) > 200:
                summary += "..."
            result += f"**Abstract:** {summary}\n\n---\n\n"
        
        return result
    
    return memory_search_tool


def create_memory_store_tool():
    """Create memory storage tool for agents."""
    memory = ChromaMemory()
    
    from crewai.tools import tool
    
    @tool("memory_store")
    def memory_store_tool(papers_json: str, collection: str = "research_papers") -> str:
        """
        Store papers in memory for future retrieval.
        
        Args:
            papers_json: JSON string containing list of papers
            collection: Collection to store in (default: research_papers)
            
        Returns:
            Storage status message
        """
        try:
            papers = json.loads(papers_json)
            if not isinstance(papers, list):
                return "Error: papers_json must be a JSON list of paper objects"
            
            success = memory.store_papers(papers, collection)
            
            if success:
                return f"Successfully stored {len(papers)} papers in memory collection '{collection}'"
            else:
                return "Failed to store papers in memory"
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for papers_json"
        except Exception as e:
            return f"Error storing papers: {str(e)}"
    
    return memory_store_tool