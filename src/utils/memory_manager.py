from langchain_chroma import Chroma
from langchain_core.documents import Document
from .llm_factory import LLMFactory
from .config import Config
import os
from typing import List, Dict

class MemoryManager:
    def __init__(self, collection_name: str = "research_memory"):
        self.embeddings = LLMFactory.get_embeddings()
        self.collection_name = collection_name
        
        # Ensure directory exists
        os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
        
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_DB_PATH
        )
    
    def store_research(self, content: str, metadata: Dict) -> str:
        """Store research findings in vector database"""
        doc = Document(page_content=content, metadata=metadata)
        doc_id = self.vectorstore.add_documents([doc])[0]
        return doc_id
    
    def retrieve_similar(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve similar documents from memory"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_context(self, query: str, max_docs: int = 5) -> str:
        """Get formatted context for query"""
        docs = self.retrieve_similar(query, k=max_docs)
        if not docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"Context {i}: {doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def clear_memory(self):
        """Clear all stored memory"""
        self.vectorstore.delete_collection()