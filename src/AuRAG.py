"""
AuRAG: Auditable Retrieval-Augmented Generation
================================================
Local RAG with Defense-in-Depth architecture:
- Layer 1: Hierarchical Chunking (SPHR) - Section-aware retrieval
- Layer 2: Grammar-Constrained Decoding (RDG) - Citation constraints
- Layer 3: Deterministic Verification (planned)

Dual-Mode Architecture:
- PRIMARY: GCD via llama-cpp-python (JSON structure + citation validation)
- FALLBACK: Post-hoc validation via Ollama (~99% accuracy)

Core Components:
- LLM: Llama-3.1-8B-Instruct (GGUF or Ollama)
- Embeddings: BAAI/bge-small-en-v1.5
- Vector DB: ChromaDB (local)
- Chunking: Hierarchical with semantic boundaries
"""

import os
import json
import shutil
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import Layer 1 & Layer 2 implementations
from SPHR import HierarchicalChunker, HierarchicalRetriever
from RDG import get_rdg_pipeline, get_rdg_mode

# ==========================================
# CONFIGURATION
# ==========================================
DOCS_FOLDER = "data/raw"
CHROMA_DIR = "data/vector_db/chroma/"
CHUNKS_OUTPUT = "data/processed/chunks.json"

# Layer 1: Hierarchical Chunking Configuration
CHILD_CHUNK_SIZE = 300      # Tokens per child chunk
CHILD_CHUNK_OVERLAP = 90    # Token overlap between children
TOP_K = 5                    # Top-k children to retrieve

# Layer 2: RDG Model Configuration
N_GPU_LAYERS = -1           # All layers on GPU (-1 = auto)
N_CTX = 16384               # Context window for GGUF model (increased to 16K)

# LLM Generation Configuration
MAX_TOKENS = 3072     # Max output tokens for structured JSON output

# Retrieval & Context Configuration
CONTEXT_BUDGET = 12000  # Max tokens for retrieved context documents

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def _load_documents(folder: str) -> List[Document]:
    """Load all documents (PDF + TXT) from folder recursively"""
    docs = []
    
    # Load PDF files
    pdf_loader = DirectoryLoader(
        folder, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True
    )
    docs.extend(pdf_loader.load())
    
    # Load TXT files
    txt_loader = DirectoryLoader(
        folder, glob="**/*.txt", loader_cls=TextLoader, recursive=True
    )
    docs.extend(txt_loader.load())
    
    if not docs:
        raise FileNotFoundError(f"No PDF or TXT files found in {folder}/")
    
    return docs

# ==========================================
# 1. INITIALIZE RDG PIPELINE (AUTO-DETECT)
# ==========================================
print("Initializing RDG Pipeline (auto-detecting mode)...")

def get_rdg():
    """Get or create RDG pipeline with auto-detection.
    
    Priority:
    1. GGUF model exists → True GCD (primary)
    2. Ollama available → Post-hoc validation (fallback)
    """
    return get_rdg_pipeline(n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS)

print(f"✓ RDG Pipeline ready (lazy initialization)")

# ==========================================
# 2. LOAD & CHUNK DOCUMENTS (LAYER 1)
# ==========================================
print("\nLoading & chunking documents...")
docs = _load_documents(DOCS_FOLDER)

print("\nApplying hierarchical chunking...")
chunker = HierarchicalChunker(
    child_size=CHILD_CHUNK_SIZE,
    child_overlap=CHILD_CHUNK_OVERLAP
)

all_parent_chunks = []
all_child_chunks = []

# Group documents by source to handle multi-page PDFs correctly
grouped_docs = {}
for doc in docs:
    source = doc.metadata.get('source', 'unknown')
    if source not in grouped_docs:
        grouped_docs[source] = []
    grouped_docs[source].append(doc)

# Process each source file as one unique contract
sorted_sources = sorted(grouped_docs.keys())

for idx, source in enumerate(sorted_sources):
    source_docs = grouped_docs[source]
    # Sort pages by page number if available (for PDFs)
    source_docs.sort(key=lambda d: d.metadata.get('page', 0))
    
    # Combine all pages into full contract text
    contract_text = "\n\n".join([d.page_content for d in source_docs])
    
    # Generate ID from filename
    filename = source.split('/')[-1]
    contract_id = f"doc_{idx}_{filename}"
    
    # Apply hierarchical chunking
    parents, children = chunker.chunk_contract(contract_text, contract_id)
    
    all_parent_chunks.extend(parents)
    all_child_chunks.extend(children)

# Build parent-to-children mapping (moved from loop to use static method)
parent_to_children = HierarchicalChunker.build_parent_child_mapping(
    all_parent_chunks, 
    all_child_chunks
)

# Enrich mapping with cross-references (lightweight graph for internal references)
parent_to_children = chunker.enrich_parent_to_children_with_cross_refs(
    parent_to_children,
    all_parent_chunks,
    all_child_chunks
)

print(f"✓ Chunked {len(all_parent_chunks)} parents + {len(all_child_chunks)} children")

# Save chunk metadata for auditing
chunks_metadata = {
    'parents': [p.to_dict() for p in all_parent_chunks],
    'children': [c.to_dict() for c in all_child_chunks],
    'parent_to_children': parent_to_children
}

os.makedirs(os.path.dirname(CHUNKS_OUTPUT), exist_ok=True)
with open(CHUNKS_OUTPUT, 'w') as f:
    json.dump(chunks_metadata, f, indent=2)
print(f"✓ Saved chunk metadata to {CHUNKS_OUTPUT}")

# Prepare documents for vectorization (use child chunks for retrieval)
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
    for child in all_child_chunks
]

print(f"\n✓ Prepared {len(splits)} child chunks for vectorization")

# ==========================================
# 3. CREATE EMBEDDINGS & VECTOR DB
# ==========================================
# Fix: Delete old vector_db to prevent duplicate embeddings
print("\nBuilding vector database...")
vector_db_path = "data/vector_db"
if os.path.exists(vector_db_path):
    shutil.rmtree(vector_db_path)

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True}
)

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=CHROMA_DIR,
)
print(f"✓ Created {vectordb._collection.count()} vectors")

# ==========================================
# 4. CREATE QA CHAIN WITH HIERARCHICAL RETRIEVAL
# ==========================================
print("Creating QA chain with Layer 1 (Hierarchical Retrieval)...")

# Initialize hierarchical retriever (from Layer 1)
hierarchical_retriever = HierarchicalRetriever(
    vectordb, 
    all_parent_chunks,
    parent_to_children,
    top_k=TOP_K,
    context_budget=CONTEXT_BUDGET
)

# Template not needed - RDG builds its own prompt with GBNF constraints

def _build_sources(docs) -> List[Dict]:
    """Extract unique source documents from retrieval results"""
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


def answer_with_sources(question: str):
    """Get answer with sources using Grammar-Constrained Decoding.
    
    AuRAG Phase 2 Flow:
    1. Retrieve context (Layer 1: SPHR)
    2. Extract valid citations from retrieved documents
    3. Generate with GBNF grammar constraints (Layer 2: RDG)
    4. Validate and return structured output with citations
    
    Args:
        question: User's query
    
    Returns:
        Dict with 'answer', 'sources', 'structured_output'
    """
    # Step 1: Retrieve context (Layer 1: SPHR Hierarchical Retrieval)
    docs = hierarchical_retriever.invoke(question)
    
    if not docs:
        return {
            'answer': "No relevant context found.",
            'sources': [],
            'structured_output': None
        }
    
    # Step 2: Generate with GCD (Layer 2: RDG)
    try:
        rdg = get_rdg()
        
        # GCD: Grammar-constrained generation + citation validation
        # JSON structure enforced at token level, citations validated post-generation
        structured_output = rdg.generate(
            question=question,
            documents=docs,
            max_tokens=MAX_TOKENS
        )
        
        return {
            'answer': structured_output.answer,
            'sources': _build_sources(docs),
            'structured_output': structured_output.to_dict()
        }
    
    except Exception as e:
        print(f"⚠️  RDG generation failed: {e}")
        return {
            'answer': f"Error: {str(e)}",
            'sources': _build_sources(docs),
            'structured_output': None
        }

print("\n✓ System ready!")
print(f"✓ Layer 1 (SPHR Hierarchical Retrieval): ACTIVE")
print(f"✓ Layer 2 (RDG): ACTIVE (mode will be detected on first query)")

# ==========================================
# 5. CLI INTERFACE (Single-Click Operation)
# ==========================================
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
        result = answer_with_sources(question)
        
        # Output structured JSON
        if result.get('structured_output'):
            mode = get_rdg_mode()
            mode_label = "Grammar-Constrained Decoding" if mode == 'gcd' else "Post-hoc Validation"
            print(f"\n📄 Answer ({mode_label}):")
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