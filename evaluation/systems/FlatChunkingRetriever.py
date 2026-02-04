"""
Flat Chunking Retriever (Standard RAG Baseline)
================================================

Implements traditional flat chunking as a baseline for comparison with SPHR.
No hierarchical structure, no cross-reference enrichment.
"""

from typing import List, Dict, Any
from pathlib import Path
import tiktoken

from base import RetrieverSystem

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FlatChunkingRetriever(RetrieverSystem):
    """
    Standard flat chunking baseline.
    
    Uses LangChain's RecursiveCharacterTextSplitter to create fixed-size
    chunks with overlap. No parent-child hierarchy.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        encoding_name: str = 'cl100k_base',
    ):
        """
        Initialize flat chunking configuration.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            encoding_name: Tokenizer encoding for length calculation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        
        self.article_mapping = None  # Maps chunk_id -> article_id
        self.vectorstore = None
        self.num_chunks = 0
    
    def setup(
        self,
        corpus: Dict[str, Dict[str, str]],
        embedding_model: str,
        db_path: Path,
        **kwargs
    ) -> None:
        """Initialize flat chunking retrieval system."""
        print(f"\n🔧 Setting up Flat Chunking Retriever...")
        print(f"   Chunk size: {self.chunk_size} tokens")
        print(f"   Chunk overlap: {self.chunk_overlap} tokens")
        
        # Initialize tokenizer for length calculation
        try:
            encoding = tiktoken.get_encoding(self.encoding_name)
        except Exception:
            encoding = tiktoken.encoding_for_model("gpt-4")
        
        def _length_function(text: str) -> int:
            """Token-based length function."""
            return len(encoding.encode(text))
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=_length_function,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        
        # Chunk all documents
        documents = []
        self.article_mapping = {}
        
        for doc_id, doc_data in sorted(corpus.items()):
            caption = (doc_data.get('caption') or '').strip()
            text = (doc_data.get('text') or '').strip()
            
            doc_text = f"{caption}\n{text}" if caption else text
            
            # Split into chunks
            chunks = text_splitter.split_text(doc_text)
            
            # Create LangChain documents
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        'chunk_id': chunk_id,
                        'article_id': doc_id,
                        'chunk_index': chunk_idx,
                    }
                )
                documents.append(doc)
                self.article_mapping[chunk_id] = doc_id
        
        self.num_chunks = len(documents)
        print(f"✓ Created {self.num_chunks} flat chunks from {len(corpus)} documents")
        
        # Build vector database
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Remove existing DB and rebuild
        if db_path.exists():
            import shutil
            shutil.rmtree(db_path)
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(db_path)
        )
        
        print(f"✓ Vector DB ready with {len(documents)} chunks")
        print(f"✓ Flat Chunking Retriever ready")
    
    def retrieve_articles(
        self,
        query: str,
        top_k: int = 5
    ) -> List[str]:
        """Retrieve articles using flat chunking."""
        if self.vectorstore is None:
            raise RuntimeError("System not initialized. Call setup() first.")
        
        # Retrieve top chunks (retrieve more chunks to ensure we get top_k unique articles)
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k * 5}  # Retrieve more to ensure coverage
        )
        
        retrieved_docs = retriever.invoke(query)
        
        # Extract unique article IDs in order
        retrieved_articles = []
        seen = set()
        
        for doc in retrieved_docs:
            article_id = doc.metadata.get('article_id')
            if article_id and article_id not in seen:
                retrieved_articles.append(article_id)
                seen.add(article_id)
                if len(retrieved_articles) >= top_k:
                    break
        
        return retrieved_articles
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return system metadata."""
        return {
            'system_name': 'FlatChunking',
            'description': 'Standard flat chunking baseline (no hierarchy)',
            'config': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'encoding_name': self.encoding_name,
            },
            'statistics': {
                'num_chunks': self.num_chunks,
            }
        }
