"""
AuRAG: Auditable Retrieval-Augmented Generation
================================================
Local RAG with Defense-in-Depth architecture:
- Layer 1: Hierarchical Chunking (implemented)
- Layer 2: Grammar-Constrained Decoding (planned)
- Layer 3: Deterministic Verification (planned)

Core Components:
- LLM: Llama-3.1-8B-Instruct (via Ollama)
- Embeddings: BAAI/bge-small-en-v1.5
- Vector DB: ChromaDB (local)
- Chunking: Hierarchical with semantic boundaries

Setup: ollama pull llama3.1:8b-instruct
"""

import os
import json
import shutil
from typing import List, Dict, Tuple
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.callbacks import StreamingStdOutCallbackHandler, CallbackManager
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Import Layer 1 implementation
from SPHR import HierarchicalChunker, Chunk, HierarchicalRetriever

# ==========================================
# CONFIGURATION
# ==========================================
DOCS_FOLDER = "data/raw"
CHROMA_DIR = "data/vector_db/chroma/"
CHUNKS_OUTPUT = "data/processed/chunks.json"

# Ollama model configuration
OLLAMA_MODEL = "llama3.1"  # Uses auto-selected variant; full name: llama3.1:8b-instruct
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama server

# Layer 1: Hierarchical Chunking Configuration
CHILD_CHUNK_SIZE = 300      # Tokens per child chunk
CHILD_CHUNK_OVERLAP = 90    # Token overlap between children
TOP_K = 8                    # Top-k children to retrieve

# M3 Pro Optimization
USE_METAL_GPU = True  # Metal acceleration for M3
N_GPU_LAYERS = -1     # All layers on GPU (-1 = auto detect)
N_CTX = 131072        # 128K context window for Llama-3.1-8B

# LLM Generation Configuration
MAX_TOKENS = 256      # Max output tokens (legal Q&A: 200-250 words typical)
                      # Sweet spot: 256 = 30-40% faster than 512, sufficient for contract queries
                      # Adjust based on: longer_answers=512, concise_answers=128

# Retrieval & Context Configuration
CONTEXT_BUDGET = 100000  # Max tokens for retrieved context documents (for 128K models)
                         # Example allocation for 128K window:
                         #   - Context (retrieved docs): 100000 tokens
                         #   - Generation (answer): 256 tokens
                         #   - Prompt + Question + Buffer: ~30816 tokens

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
# 1. INITIALIZE OLLAMA LLM
# ==========================================
print(f"Connecting to Ollama ({OLLAMA_MODEL})...")
print(f"  Note: If model not found, run: ollama pull {OLLAMA_MODEL}")
llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_ctx=N_CTX,
    num_predict=MAX_TOKENS,
)
print(f"✓ Connected to {OLLAMA_MODEL}")

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

for idx, doc in enumerate(docs):
    contract_text = doc.page_content
    contract_id = f"doc_{idx}_{doc.metadata.get('source', 'unknown').split('/')[-1]}"
    
    # Apply hierarchical chunking
    parents, children = chunker.chunk_contract(contract_text, contract_id)
    
    all_parent_chunks.extend(parents)
    all_child_chunks.extend(children)

# Build parent-to-children mapping (moved from loop to use static method)
parent_to_children = HierarchicalChunker.build_parent_child_mapping(
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

template = (
    "Use the following context to answer the question.\n"
    "If you don't know, say you don't know.\n"
    "Keep the answer concise.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
prompt = PromptTemplate.from_template(template)

# Create QA chain
qa_chain_base = prompt | llm | StrOutputParser()

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
    """Get answer and track source documents for references"""
    docs = hierarchical_retriever.invoke(question)
    context_str = '\n---\n'.join([d.page_content for d in docs])
    # Format prompt manually since we override context
    formatted = template.format(context=context_str, question=question)
    answer = llm.invoke(formatted).strip()
    return {'answer': answer, 'sources': _build_sources(docs)}

print("\n✓ System ready!")

# ==========================================
# 6. CLI INTERFACE
# ==========================================
print("\nAuRAG - Auditable RAG | Type 'exit' to quit\n")

while True:
    try:
        question = input("\nYou: ").strip()
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🔍 Retrieving context...")
        result = answer_with_sources(question)
        
        print(f"\n📖 Answer: {result['answer']}\n")
        
        # Print references
        if result['sources']:
            print("📚 References:")
            for i, source in enumerate(result['sources'], 1):
                print(f"   [{i}] {source['section']} (Section {source['section_num']})")
                print(f"       Chunk ID: {source['chunk_id']}")
                print(f"       Tokens: {source['token_count']}")
            print()
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}\n")