"""
Base Interfaces for AuRAG Retrieval Systems
============================================

Defines common interfaces for retrieval and generation systems that can be
plugged into evaluation harnesses.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from pathlib import Path


class RetrieverSystem(ABC):
    """
    Base interface for retrieval systems.
    
    All retrieval baselines (SPHR, flat chunking, semantic chunking, etc.)
    must implement this interface to be compatible with the evaluation harness.
    """
    
    @abstractmethod
    def setup(
        self,
        corpus: Dict[str, Dict[str, str]],
        embedding_model: str,
        db_path: Path,
        **kwargs
    ) -> None:
        """
        Initialize the retrieval system with a corpus.
        
        Args:
            corpus: Dictionary mapping document_id -> {'caption': str, 'text': str}
            embedding_model: Name of embedding model (e.g., 'BAAI/bge-small-en-v1.5')
            db_path: Path to vector database storage
            **kwargs: System-specific configuration
        """
        pass
    
    @abstractmethod
    def retrieve_articles(
        self,
        query: str,
        top_k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant document IDs for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of document IDs, ordered by relevance (most relevant first)
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return system metadata for logging/reporting.
        
        Returns:
            Dictionary with system name, configuration, and statistics
        """
        pass
    
    def cleanup(self) -> None:
        """
        Optional cleanup method (e.g., close database connections).
        Default implementation does nothing.
        """
        pass


class RAGSystem(ABC):
    """
    Base interface for end-to-end RAG systems (retrieval + generation).
    
    Used for Template 2 evaluation (citation constraint + generation quality).
    """
    
    @abstractmethod
    def setup(
        self,
        corpus: Dict[str, Dict[str, str]],
        embedding_model: str,
        llm_model_path: str,
        db_path: Path,
        **kwargs
    ) -> None:
        """
        Initialize the RAG system.
        
        Args:
            corpus: Document corpus
            embedding_model: Embedding model name
            llm_model_path: Path to LLM weights
            db_path: Vector database path
            **kwargs: System-specific configuration
        """
        pass
    
    @abstractmethod
    def query(
        self,
        question: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Process a query end-to-end: retrieve + generate answer with citations.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            {
                'answer': str,
                'reasoning': str,
                'citations': List[str],  # Document IDs cited in answer
                'retrieved': List[str],  # Document IDs retrieved
                'retrieval_time': float,  # Seconds
                'generation_time': float,  # Seconds
            }
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return system metadata.
        
        Returns:
            Dictionary with system name, configuration, and statistics
        """
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup."""
        pass
