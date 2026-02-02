"""
AuRAG: Auditable Retrieval-Augmented Generation
================================================

Local RAG system with two-layer architecture:
- Layer 1 (SPHR): Hierarchical chunking and parent-child retrieval
- Layer 2 (RDG): Grammar-constrained citation generation

Stack:
- LLM: Llama-3.1-8B-Instruct (GGUF) with GBNF constraints
- Embeddings: BAAI/bge-small-en-v1.5 (MPS-accelerated)
- Vector DB: ChromaDB (persistent, local)
"""

import os
import json
import shutil
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from SPHR import HierarchicalChunker, HierarchicalRetriever
from RDG import get_rdg_pipeline


@dataclass
class AuRAGConfig:
    """Configuration for AuRAG pipeline"""
    # Paths
    docs_folder: str = "data/raw"
    chroma_dir: str = "data/vector_db/chroma/"
    chunks_output: str = "data/processed/chunks.json"
    
    # Layer 1: Hierarchical Chunking
    child_chunk_size: int = 300
    child_chunk_overlap: int = 90
    top_k: int = 5
    context_budget: int = 12000
    
    # Layer 2: RDG Model
    n_gpu_layers: int = -1
    n_ctx: int = 16384
    max_tokens: int = 3072
    
    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "mps"


class AuRAGPipeline:
    """
    Complete AuRAG pipeline with two-layer defense architecture.
    
    Research-focused design: Rebuilds vector DB on each initialization
    to ensure reproducibility and fresh indexing for experiments.
    """
    
    def __init__(self, config: Optional[AuRAGConfig] = None):
        self.config = config or AuRAGConfig()
        
        # State
        self.all_parent_chunks = []
        self.all_child_chunks = []
        self.parent_to_children = {}
        self.vectordb = None
        self.retriever = None
        self.rdg = None
        
        # Initialize pipeline
        self._setup()
    
    def _setup(self):
        """Execute full pipeline setup (research mode: always rebuild)"""
        print("Initializing AuRAG Pipeline (Research Mode)...")
        
        # Load and chunk documents
        docs = self._load_documents()
        self._chunk_documents(docs)
        
        # Build vector database (always rebuild for research)
        self._build_vector_db()
        
        # Initialize retriever
        self._setup_retriever()
        
        print("\n✓ System ready!")
        print(f"✓ Layer 1 (SPHR Hierarchical Retrieval): ACTIVE")
        print(f"✓ Layer 2 (RDG Grammar-Constrained Decoding): ACTIVE")
    
    def _load_documents(self) -> List[Document]:
        """Load all documents from configured folder"""
        print("\nLoading documents...")
        docs = []
        
        # Load PDFs
        pdf_loader = DirectoryLoader(
            self.config.docs_folder, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader, 
            recursive=True
        )
        docs.extend(pdf_loader.load())
        
        # Load TXT files
        txt_loader = DirectoryLoader(
            self.config.docs_folder, 
            glob="**/*.txt", 
            loader_cls=TextLoader, 
            recursive=True
        )
        docs.extend(txt_loader.load())
        
        if not docs:
            raise FileNotFoundError(
                f"No PDF or TXT files found in {self.config.docs_folder}/"
            )
        
        print(f"✓ Loaded {len(docs)} document pages")
        return docs
    
    def _chunk_documents(self, docs: List[Document]):
        """Apply hierarchical chunking to documents"""
        print("\nApplying hierarchical chunking...")
        
        chunker = HierarchicalChunker(
            child_size=self.config.child_chunk_size,
            child_overlap=self.config.child_chunk_overlap
        )
        
        # Group by source file
        grouped_docs = {}
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in grouped_docs:
                grouped_docs[source] = []
            grouped_docs[source].append(doc)
        
        # Process each source
        sorted_sources = sorted(grouped_docs.keys())
        for idx, source in enumerate(sorted_sources):
            source_docs = grouped_docs[source]
            source_docs.sort(key=lambda d: d.metadata.get('page', 0))
            
            # Combine pages
            contract_text = "\n\n".join([d.page_content for d in source_docs])
            filename = source.split('/')[-1]
            contract_id = f"doc_{idx}_{filename}"
            
            # Chunk
            parents, children = chunker.chunk_contract(contract_text, contract_id)
            self.all_parent_chunks.extend(parents)
            self.all_child_chunks.extend(children)
        
        # Build mappings
        self.parent_to_children = HierarchicalChunker.build_parent_child_mapping(
            self.all_parent_chunks, 
            self.all_child_chunks
        )
        
        # Enrich with cross-references
        self.parent_to_children = chunker.enrich_parent_to_children_with_cross_refs(
            self.parent_to_children,
            self.all_parent_chunks,
            self.all_child_chunks
        )
        
        print(f"✓ Chunked {len(self.all_parent_chunks)} parents + {len(self.all_child_chunks)} children")
        
        # Save metadata
        self._save_chunk_metadata()
    
    def _save_chunk_metadata(self):
        """Save chunk metadata for auditing"""
        chunks_metadata = {
            'parents': [p.to_dict() for p in self.all_parent_chunks],
            'children': [c.to_dict() for c in self.all_child_chunks],
            'parent_to_children': self.parent_to_children
        }
        
        os.makedirs(os.path.dirname(self.config.chunks_output), exist_ok=True)
        with open(self.config.chunks_output, 'w') as f:
            json.dump(chunks_metadata, f, indent=2)
        print(f"✓ Saved chunk metadata to {self.config.chunks_output}")
    
    def _build_vector_db(self):
        """Build vector database (always rebuild for research)"""
        print("\nBuilding vector database (research mode: fresh rebuild)...")
        
        # Clean existing DB
        vector_db_path = "data/vector_db"
        if os.path.exists(vector_db_path):
            shutil.rmtree(vector_db_path)
        
        # Prepare child chunks for vectorization
        splits = [
            Document(
                page_content=child.text,
                metadata={
                    'chunk_id': child.chunk_id,
                    'chunk_type': 'child',
                    'parent_id': child.parent_id,
                    'section_num': child.section_num,
                    'section_title': child.section_title,
                    'contract_id': child.contract_id,
                    'token_count': child.token_count
                }
            )
            for child in self.all_child_chunks
        ]
        
        # Create embeddings
        embedding = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': self.config.embedding_device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Build Chroma DB
        self.vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=self.config.chroma_dir,
        )
        print(f"✓ Created {self.vectordb._collection.count()} vectors")
    
    def _setup_retriever(self):
        """Initialize hierarchical retriever"""
        print("\nInitializing hierarchical retriever...")
        self.retriever = HierarchicalRetriever(
            self.vectordb, 
            self.all_parent_chunks,
            self.parent_to_children,
            top_k=self.config.top_k,
            context_budget=self.config.context_budget
        )
    
    def _get_rdg(self):
        """Get or initialize RDG pipeline (lazy loading)"""
        if self.rdg is None:
            self.rdg = get_rdg_pipeline(
                n_ctx=self.config.n_ctx, 
                n_gpu_layers=self.config.n_gpu_layers
            )
        return self.rdg
    
    @staticmethod
    def _build_sources(docs: List[Document]) -> List[Dict]:
        """Extract unique source metadata from documents"""
        sources = []
        seen_chunks = set()
        for doc in docs:
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            if chunk_id not in seen_chunks:
                sources.append({
                    'chunk_id': chunk_id,
                    'section': doc.metadata.get('section_title', 'Unknown'),
                    'section_num': doc.metadata.get('section_num', 'N/A'),
                    'token_count': doc.metadata.get('token_count', 0),
                    'parent_id': doc.metadata.get('parent_id', 'N/A')
                })
                seen_chunks.add(chunk_id)
        return sources
    
    def query(self, question: str) -> Dict:
        """
        Query the pipeline with a question.
        
        Args:
            question: User query
            
        Returns:
            Dict with 'answer', 'sources', 'structured_output'
        """
        # Retrieve context
        docs = self.retriever.invoke(question)
        
        if not docs:
            return {
                'answer': "No relevant context found.",
                'sources': [],
                'structured_output': None
            }
        
        # Generate with RDG
        try:
            rdg = self._get_rdg()
            structured_output = rdg.generate(
                question=question,
                documents=docs,
                max_tokens=self.config.max_tokens
            )
            
            return {
                'answer': structured_output.answer,
                'sources': self._build_sources(docs),
                'structured_output': structured_output.to_dict()
            }
        
        except Exception as e:
            print(f"⚠️  RDG generation failed: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'sources': self._build_sources(docs),
                'structured_output': None
            }


def run_cli(pipeline: AuRAGPipeline):
    """Interactive CLI interface"""
    print(f"\nAuRAG - Auditable RAG | Type 'exit' to quit\n")
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if question.lower() == 'exit':
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n" + "="*60)
            print("🔍 Retrieving context...")
            result = pipeline.query(question)
            
            # Display results
            if result.get('structured_output'):
                print(f"\n📄 Answer (Grammar-Constrained Decoding):")
                print("="*60)
                print(json.dumps(result['structured_output'], indent=2))
                print("="*60)
            else:
                print(f"\n📖 Answer: {result['answer']}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    # Initialize and run
    pipeline = AuRAGPipeline()
    run_cli(pipeline)