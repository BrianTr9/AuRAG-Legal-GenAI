"""
SPHR Retriever System
=====================

Implements Structure-Preserving Hierarchical Retrieval as a pluggable system
for evaluation harness.
"""

import sys
from typing import List, Dict, Any
from pathlib import Path

# Add src to path to import SPHR module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from base import RetrieverSystem
from SPHR import HierarchicalChunker, HierarchicalRetriever, Chunk

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class SPHRRetriever(RetrieverSystem):
    """
    SPHR-based retrieval system.
    
    Uses hierarchical chunking (parent-child) with cross-reference enrichment.
    """
    
    def __init__(
        self,
        child_size: int = 300,
        child_overlap: int = 90,
        parent_max_tokens: int = 3000,
        encoding_name: str = 'cl100k_base',
        child_top_k: int = 50,
        context_budget: int = 100000,
        enable_cross_ref_enrichment: bool = True,
    ):
        """
        Initialize SPHR configuration.
        
        Args:
            child_size: Tokens per child chunk
            child_overlap: Token overlap between children
            parent_max_tokens: Maximum tokens per parent chunk
            encoding_name: Tokenizer encoding
            child_top_k: Number of child chunks to retrieve
            context_budget: Maximum tokens for context expansion
            enable_cross_ref_enrichment: Whether to enrich with cross-references
        """
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.parent_max_tokens = parent_max_tokens
        self.encoding_name = encoding_name
        self.child_top_k = child_top_k
        self.context_budget = context_budget
        self.enable_cross_ref_enrichment = enable_cross_ref_enrichment
        
        self.chunker = None
        self.retriever = None
        self.article_mapping = None
        self.parents = None
        self.children = None
        self.num_enriched_edges = 0
    
    def setup(
        self,
        corpus: Dict[str, Dict[str, str]],
        embedding_model: str,
        db_path: Path,
        **kwargs
    ) -> None:
        """Initialize SPHR retrieval system."""
        print(f"\n🔧 Setting up SPHR Retriever...")
        print(f"   Child size: {self.child_size} tokens")
        print(f"   Child overlap: {self.child_overlap} tokens")
        print(f"   Parent max: {self.parent_max_tokens} tokens")
        print(f"   Cross-ref enrichment: {self.enable_cross_ref_enrichment}")
        
        # Initialize chunker
        self.chunker = HierarchicalChunker(
            child_size=self.child_size,
            child_overlap=self.child_overlap,
            encoding_name=self.encoding_name,
            parent_max_tokens=self.parent_max_tokens,
        )
        
        # Chunk corpus
        all_parents = []
        all_children = []
        self.article_mapping = {}
        
        for doc_id, doc_data in sorted(corpus.items()):
            caption = (doc_data.get('caption') or '').strip()
            text = (doc_data.get('text') or '').strip()
            
            doc_text = f"{caption}\n{text}" if caption else text
            contract_id = f"civil_article_{doc_id}"
            
            try:
                parents, children = self.chunker.chunk_contract(doc_text, contract_id)
                all_parents.extend(parents)
                all_children.extend(children)
                
                for parent in parents:
                    self.article_mapping[parent.chunk_id] = doc_id
            except Exception as e:
                print(f"  ⚠️  Error chunking document {doc_id}: {e}")
                continue
        
        self.parents = all_parents
        self.children = all_children
        
        print(f"✓ Created {len(self.parents)} parent chunks")
        print(f"✓ Created {len(self.children)} child chunks")
        
        # Build vector database
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        documents = []
        for child in self.children:
            doc = Document(
                page_content=child.text,
                metadata={
                    'chunk_id': child.chunk_id,
                    'parent_id': child.parent_id,
                    'section_num': child.section_num,
                    'contract_id': child.contract_id,
                    'char_start': child.char_start,
                    'char_end': child.char_end,
                }
            )
            documents.append(doc)
        
        # Remove existing DB and rebuild
        if db_path.exists():
            import shutil
            shutil.rmtree(db_path)
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(db_path)
        )
        
        print(f"✓ Vector DB ready with {len(documents)} child chunks")
        
        # Build parent-child mapping
        parent_to_children = HierarchicalChunker.build_parent_child_mapping(
            self.parents,
            self.children
        )
        
        initial_edges = sum(len(v) for v in parent_to_children.values())
        
        # Apply cross-reference enrichment
        if self.enable_cross_ref_enrichment:
            # Treat all as same document for cross-references
            for p in self.parents:
                p.contract_id = 'civil_code'
            for c in self.children:
                c.contract_id = 'civil_code'
            
            parent_to_children = self.chunker.enrich_parent_to_children_with_cross_refs(
                parent_to_children,
                self.parents,
                self.children,
            )
            
            self.num_enriched_edges = sum(len(v) for v in parent_to_children.values())
            print(f"✓ Cross-ref enrichment: {initial_edges} → {self.num_enriched_edges} edges")
        else:
            self.num_enriched_edges = initial_edges
        
        # Initialize hierarchical retriever
        self.retriever = HierarchicalRetriever(
            vectordb=vectorstore,
            parent_chunks=self.parents,
            parent_to_children=parent_to_children,
            top_k=self.child_top_k,
            context_budget=self.context_budget,
        )
        
        print(f"✓ SPHR Retriever ready")
    
    def retrieve_articles(
        self,
        query: str,
        top_k: int = 5
    ) -> List[str]:
        """Retrieve articles using hierarchical retrieval."""
        if self.retriever is None:
            raise RuntimeError("System not initialized. Call setup() first.")
        
        # Get parent documents from hierarchical retriever
        parent_docs = self.retriever.get_relevant_documents(query)
        
        # Extract article IDs from parent chunks
        retrieved_articles = []
        seen = set()
        
        for doc in parent_docs:
            parent_id = doc.metadata.get('chunk_id')
            if parent_id in self.article_mapping:
                article_id = self.article_mapping[parent_id]
                if article_id not in seen:
                    retrieved_articles.append(article_id)
                    seen.add(article_id)
                    if len(retrieved_articles) >= top_k:
                        break
        
        return retrieved_articles
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return system metadata."""
        return {
            'system_name': 'SPHR',
            'description': 'Structure-Preserving Hierarchical Retrieval',
            'config': {
                'child_size': self.child_size,
                'child_overlap': self.child_overlap,
                'parent_max_tokens': self.parent_max_tokens,
                'encoding_name': self.encoding_name,
                'child_top_k': self.child_top_k,
                'context_budget': self.context_budget,
                'cross_ref_enrichment': self.enable_cross_ref_enrichment,
            },
            'statistics': {
                'num_parents': len(self.parents) if self.parents else 0,
                'num_children': len(self.children) if self.children else 0,
                'num_enriched_edges': self.num_enriched_edges,
            }
        }
