"""
AuRAG System Wrapper for Template 2 Evaluation
==============================================

Wraps the AuRAGPipeline (SPHR + RDG) for end-to-end RAG evaluation.
This system uses grammar-constrained generation to enforce hard citation constraints.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import shutil

# Add src/ to path to import AuRAG modules
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from SPHR import HierarchicalChunker, HierarchicalRetriever
from RDG import get_rdg_pipeline

from .base import RAGSystem


class AuRAGSystem(RAGSystem):
    """
    AuRAG system (SPHR + RDG) for Template 2 evaluation.
    
    Architecture:
    - Layer 1: SPHR hierarchical retrieval with parent-child structure
    - Layer 2: RDG grammar-constrained generation with hard citation constraints
    
    This implementation differs from src/AuRAG.py in that it:
    1. Accepts a pre-built corpus (no document loading)
    2. Returns timing information for evaluation metrics
    3. Extracts article IDs from citations for hallucination measurement
    """
    
    def __init__(self, system_name: str = "aurag"):
        self.system_name = system_name
        self.corpus = None
        self.vectordb = None
        self.retriever = None
        self.rdg = None
        self.embedding_model = None
        
        # Track statistics
        self.retrieval_times = []
        self.generation_times = []
        
    def setup(
        self,
        corpus: Dict[str, Dict[str, str]],
        embedding_model: str,
        llm_model_path: str,
        db_path: Path,
        **kwargs
    ) -> None:
        """
        Initialize AuRAG system with corpus.
        
        Args:
            corpus: Dict mapping article_id -> {'caption': article_number, 'text': full_text}
            embedding_model: Embedding model name (e.g., 'intfloat/multilingual-e5-base')
            llm_model_path: Path to GGUF model file for RDG
            db_path: Path for ChromaDB persistence
            **kwargs: Additional config (child_chunk_size, top_k, n_ctx, etc.)
        """
        print(f"\n{'='*70}")
        print(f"Setting up {self.system_name.upper()} System (SPHR + RDG)")
        print(f"{'='*70}")
        
        self.corpus = corpus
        self.embedding_model = embedding_model
        self.llm_model_path = llm_model_path
        
        # Extract configuration
        child_chunk_size = kwargs.get('child_chunk_size', 300)
        child_chunk_overlap = kwargs.get('child_chunk_overlap', 90)
        self.top_k = kwargs.get('top_k', 5)
        self.context_budget = kwargs.get('context_budget', 12000)
        rebuild_index = kwargs.get('rebuild_index', False)
        n_ctx = kwargs.get('n_ctx', 16384)
        n_gpu_layers = kwargs.get('n_gpu_layers', -1)
        
        print(f"\n📋 Configuration:")
        print(f"  - Child chunk size: {child_chunk_size}")
        print(f"  - Child chunk overlap: {child_chunk_overlap}")
        print(f"  - Top-K retrieval: {self.top_k}")
        print(f"  - Context budget: {self.context_budget} tokens")
        print(f"  - Model context: {n_ctx} tokens")
        print(f"  - GPU layers: {n_gpu_layers}")
        print(f"  - Rebuild index: {rebuild_index}")
        
        # Step 1: Hierarchical chunking (SPHR Layer 1)
        print(f"\n🔹 Step 1: Hierarchical Chunking (SPHR)")
        chunker = HierarchicalChunker(
            child_size=child_chunk_size,
            child_overlap=child_chunk_overlap
        )
        
        all_parent_chunks = []
        all_child_chunks = []
        parent_to_children = {}
        
        for article_id, article_data in corpus.items():
            article_text = article_data['text']
            article_num = article_data['caption']
            
            # Generate hierarchical chunks using chunk_contract method
            parent_chunks, child_chunks = chunker.chunk_contract(
                article_text,
                contract_id=article_num
            )
            
            all_parent_chunks.extend(parent_chunks)
            all_child_chunks.extend(child_chunks)
            
            # Build parent-child mapping
            for parent in parent_chunks:
                parent_id = parent.chunk_id
                parent_to_children[parent_id] = [
                    child.chunk_id for child in child_chunks
                    if child.parent_id == parent_id
                ]
        
        print(f"  ✓ Created {len(all_parent_chunks)} parent chunks")
        print(f"  ✓ Created {len(all_child_chunks)} child chunks")
        
        # Step 2: Build vector database
        print(f"\n🔹 Step 2: Building Vector Database")
        splits = [
            Document(
                page_content=child.text,
                metadata={
                    'chunk_id': child.chunk_id,
                    'chunk_type': 'child',
                    'parent_id': child.parent_id,
                    'section_num': child.section_num,
                    'section_title': child.section_title,
                    'contract_id': child.contract_id,  # This is the article number
                    'token_count': child.token_count
                }
            )
            for child in all_child_chunks
        ]
        
        embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'mps'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # IMPORTANT: Avoid duplicating vectors across repeated runs.
        # - If a persisted DB exists and is non-empty, reuse it.
        # - Otherwise, build from scratch via Chroma.from_documents.
        if rebuild_index and db_path.exists():
            shutil.rmtree(db_path)

        existing_count = 0
        if db_path.exists():
            try:
                candidate = Chroma(
                    persist_directory=str(db_path),
                    embedding_function=embedding,
                )
                existing_count = candidate._collection.count()
                if existing_count > 0:
                    self.vectordb = candidate
            except Exception:
                existing_count = 0

        if existing_count > 0:
            print(f"  ✓ Reusing existing index ({existing_count} vectors)")
        else:
            self.vectordb = Chroma.from_documents(
                documents=splits,
                embedding=embedding,
                persist_directory=str(db_path),
            )
            print(f"  ✓ Indexed {self.vectordb._collection.count()} vectors")
        
        # Step 3: Initialize hierarchical retriever
        print(f"\n🔹 Step 3: Initializing Hierarchical Retriever")
        self.retriever = HierarchicalRetriever(
            self.vectordb,
            all_parent_chunks,
            parent_to_children,
            top_k=self.top_k,
            context_budget=self.context_budget
        )
        print(f"  ✓ Retriever ready")
        
        # Step 4: Initialize RDG pipeline (lazy loading for efficiency)
        print(f"\n🔹 Step 4: RDG Pipeline Ready (will load on first query)")
        print(f"  - Model: {llm_model_path}")
        self.rdg_config = {
            'model_path': llm_model_path,
            'n_ctx': n_ctx,
            'n_gpu_layers': n_gpu_layers
        }
        
        print(f"\n{'='*70}")
        print(f"✅ {self.system_name.upper()} System Ready")
        print(f"{'='*70}\n")
    
    def _get_rdg(self):
        """Lazy load RDG pipeline on first query"""
        if self.rdg is None:
            print(f"\n🔧 Loading RDG model (first query)...")
            self.rdg = get_rdg_pipeline(
                model_path=self.rdg_config['model_path'],
                n_ctx=self.rdg_config['n_ctx'],
                n_gpu_layers=self.rdg_config['n_gpu_layers']
            )
            print(f"  ✓ Model loaded")
        return self.rdg
    
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Process end-to-end RAG query: retrieve + generate with citations.
        
        Args:
            question: User question
            top_k: Override retrieval top-k (optional)
            
        Returns:
            {
                'answer': str,
                'reasoning': str,  # Structured CoT JSON string
                'citations': List[str],  # Article IDs cited in answer
                'retrieved': List[str],  # Article IDs retrieved
                'retrieval_time': float,
                'generation_time': float
            }
        """
        if top_k is None:
            top_k = self.top_k
        
        # Step 1: Retrieve context
        t0 = time.time()
        docs = self.retriever.invoke(question)
        retrieval_time = time.time() - t0
        self.retrieval_times.append(retrieval_time)
        
        # Extract article IDs from retrieved documents
        # IMPORTANT: These are the "retrieved" articles used in hallucination check
        # Each doc.metadata['contract_id'] contains the article number (e.g., "123", "456")
        retrieved_article_ids = sorted(set(
            doc.metadata.get('contract_id', 'unknown')
            for doc in docs
        ))
        
        if not docs:
            return {
                'answer': "No relevant context found.",
                'reasoning': "",
                'citations': [],
                'retrieved': [],
                'retrieval_time': retrieval_time,
                'generation_time': 0.0
            }
        
        # Step 2: Generate with RDG
        t1 = time.time()
        try:
            rdg = self._get_rdg()
            structured_output = rdg.generate(
                question=question,
                documents=docs,
                max_tokens=3072,
                answer_mode="yn",
            )
            generation_time = time.time() - t1
            self.generation_times.append(generation_time)
            
            # Extract citations from structured output
            # IMPORTANT: These are the "cited" articles used in hallucination check
            # Citations in reasoning steps are section IDs like "571_sec_571"
            # We need to extract just the article number (before "_sec_") for comparison
            # HALLUCINATION = cited article NOT in retrieved_article_ids set
            cited_article_ids = []
            if structured_output.reasoning:  # Changed from structured_reasoning to reasoning
                for step in structured_output.reasoning.steps:
                    if step.citations:
                        for citation in step.citations:
                            # Extract article ID from section ID (e.g., "571_sec_571" → "571")
                            if '_sec_' in citation:
                                article_id = citation.split('_sec_')[0]
                                cited_article_ids.append(article_id)
                            else:
                                # Fallback: use citation as-is
                                cited_article_ids.append(citation)
            
            # Deduplicate and sort
            cited_article_ids = sorted(set(cited_article_ids))
            
            return {
                'answer': structured_output.answer,
                'reasoning': structured_output.to_dict(),  # Changed from to_json() to to_dict()
                'citations': cited_article_ids,
                'retrieved': retrieved_article_ids,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time
            }
            
        except Exception as e:
            generation_time = time.time() - t1
            self.generation_times.append(generation_time)
            print(f"⚠️  RDG generation failed: {e}")
            
            return {
                'answer': f"Error: {str(e)}",
                'reasoning': "",
                'citations': [],
                'retrieved': retrieved_article_ids,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time
            }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return system metadata for reporting"""
        return {
            'system_name': self.system_name,
            'architecture': 'SPHR + RDG (Grammar-Constrained)',
            'embedding_model': self.embedding_model,
            'llm_model': self.rdg_config.get('model_path', 'unknown'),
            'top_k': self.top_k,
            'context_budget': self.context_budget,
            'num_queries_processed': len(self.retrieval_times),
            'avg_retrieval_time': sum(self.retrieval_times) / len(self.retrieval_times) if self.retrieval_times else 0,
            'avg_generation_time': sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0
        }
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.rdg:
            del self.rdg
            self.rdg = None
        if self.vectordb:
            self.vectordb = None
