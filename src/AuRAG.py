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
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Import Layer 1 & Layer 2 implementations
from SPHR import HierarchicalChunker, Chunk, HierarchicalRetriever
from RDG import RDGPipeline

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

# Layer 2: Retrieval-Dependent Grammar Configuration
USE_RDG_LAYER2 = True        # Enable/disable Layer 2 constraint-based generation
RDG_USE_OUTLINES = False     # Token-level constraints (requires: pip install outlines)
RDG_STRICT_MODE = True       # Strict=remove invalid citations, Lenient=keep best-effort

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

template = (
    "Use the following context to answer the question.\n"
    "If you don't know, say you don't know.\n"
    "Keep the answer concise.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

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

def _build_reference_format(docs) -> List[str]:
    """Build RDG-compatible reference format from retrieved documents
    
    Format: {contract_id}_Section_{section_num}_{section_title}
    Example: doc_0_AGENCY_AGREEMENT_Section_II_DUTIES_OF_AGENT
    """
    references = []
    seen = set()
    for doc in docs:
        contract_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
        sec_num = doc.metadata.get('section_num', 'N/A')
        sec_title = (doc.metadata.get('section_title') or 'Unknown').replace(' ', '_').strip()
        ref = f"{contract_id}_Section_{sec_num}_{sec_title}"
        
        if ref not in seen:
            references.append(ref)
            seen.add(ref)
    return references

def answer_with_sources(question: str, use_layer2: bool = None):
    """Get answer with sources - supports both Layer 1 only or Layer 1+2
    
    Args:
        question: User's query
        use_layer2: Override global USE_RDG_LAYER2 config. If None, uses global setting.
    
    Returns:
        Layer 1 only: {'answer': str, 'sources': List[Dict]}
        Layer 1+2: {'answer': str, 'sources': List[Dict], 'structured_output': Dict}
    """
    use_layer2 = use_layer2 if use_layer2 is not None else USE_RDG_LAYER2
    
    # Step 1: Retrieve context (Layer 1)
    docs = hierarchical_retriever.invoke(question)
    
    if not use_layer2:
        # Original Layer 1-only flow
        parts = []
        for d in docs:
            content = (d.page_content or '').strip()
            contract_id = d.metadata.get('contract_id') or d.metadata.get('parent_id', 'unknown')
            sec_num = d.metadata.get('section_num', 'N/A')
            sec_title = (d.metadata.get('section_title') or 'Unknown').replace('\n', ' ').strip()
            ref = f"{contract_id}_Section_{sec_num}_{sec_title}"
            parts.append(f"context: {content}\nreference: {ref}")

        context_str = '\n---\n'.join(parts)
        formatted = template.format(context=context_str, question=question)
        answer = llm.invoke(formatted).strip()
        return {'answer': answer, 'sources': _build_sources(docs)}
    
    else:
        # Layer 1+2 flow with RDG constraints
        try:
            rdg_pipeline = RDGPipeline()
            
            # Step 2: Prepare generation constraints (Layer 2)
            prep = rdg_pipeline.prepare_generation(
                documents=docs,
                question=question,
                context_text=""
            )
            
            # Step 3: Build constrained prompt
            parts = []
            for d in docs:
                content = (d.page_content or '').strip()
                contract_id = d.metadata.get('contract_id') or d.metadata.get('parent_id', 'unknown')
                sec_num = d.metadata.get('section_num', 'N/A')
                sec_title = (d.metadata.get('section_title') or 'Unknown').replace('\n', ' ').strip()
                ref = f"{contract_id}_Section_{sec_num}_{sec_title}"
                parts.append(f"context: {content}\nreference: {ref}")

            context_str = '\n---\n'.join(parts)
            
            # Add constraint instruction
            constraint_instruction = (
                f"\nIMPORTANT: Your answer MUST be in JSON format with these exact keys:\n"
                f"{{'citations': [...], 'reasoning': '...', 'answer': '...'}}\n"
                f"Valid citations: {prep['valid_references']}\n"
                f"Citations must be from this list ONLY. Use exact strings."
            )
            
            formatted = template.format(context=context_str, question=question)
            formatted += constraint_instruction
            
            # Step 4: Generate with constraints
            raw_output = llm.invoke(formatted).strip()
            
            # Step 5: Validate and fix output (Layer 2)
            structured_output, validation_errors = rdg_pipeline.validate_generation(
                raw_output=raw_output,
                valid_references=prep['valid_references']
            )
            
            # Step 6: Return structured result
            result = {
                'answer': structured_output.answer if structured_output else raw_output,
                'sources': _build_sources(docs),
                'structured_output': structured_output.to_dict() if structured_output else None,
                'validation_errors': validation_errors
            }
            
            if validation_errors:
                print(f"⚠️  Citation validation warnings: {validation_errors}")
            
            return result
        
        except Exception as e:
            # Fallback to Layer 1 only if Layer 2 fails
            print(f"⚠️  Layer 2 failed ({str(e)}), falling back to Layer 1 only")
            parts = []
            for d in docs:
                content = (d.page_content or '').strip()
                contract_id = d.metadata.get('contract_id') or d.metadata.get('parent_id', 'unknown')
                sec_num = d.metadata.get('section_num', 'N/A')
                sec_title = (d.metadata.get('section_title') or 'Unknown').replace('\n', ' ').strip()
                ref = f"{contract_id}_Section_{sec_num}_{sec_title}"
                parts.append(f"context: {content}\nreference: {ref}")

            context_str = '\n---\n'.join(parts)
            formatted = template.format(context=context_str, question=question)
            answer = llm.invoke(formatted).strip()
            return {'answer': answer, 'sources': _build_sources(docs)}

print("\n✓ System ready!")
print(f"✓ Layer 1 (Hierarchical Retrieval): ACTIVE")
print(f"✓ Layer 2 (RDG Constraints): {'ACTIVE' if USE_RDG_LAYER2 else 'DISABLED'}")

# ==========================================
# 6. CLI INTERFACE
# ==========================================
print(f"\nAuRAG - Auditable RAG | Commands: 'exit' (quit), 'mode <1|2>' (toggle layers)\n")

cli_use_layer2 = USE_RDG_LAYER2  # Local override for CLI

while True:
    try:
        question = input("\nYou: ").strip()
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        # CLI command: toggle between Layer 1 only and Layer 1+2
        if question.lower().startswith('mode '):
            mode = question.split()[1].strip() if len(question.split()) > 1 else None
            if mode == '1':
                cli_use_layer2 = False
                print("✓ Mode: Layer 1 only (standard retrieval)")
            elif mode == '2':
                cli_use_layer2 = True
                print("✓ Mode: Layer 1+2 (retrieval + RDG constraints)")
            else:
                print(f"Current mode: Layer {'1+2' if cli_use_layer2 else '1'}")
            continue
        
        if not question:
            continue
        
        print("\n🔍 Retrieving context...")
        result = answer_with_sources(question, use_layer2=cli_use_layer2)
        
        # Output depends on mode
        if cli_use_layer2 and result.get('structured_output'):
            # Layer 1+2 mode: output structured JSON (citations already included)
            print("\n📄 Structured Output:")
            print(json.dumps(result['structured_output'], indent=2))
        else:
            # Layer 1 only mode: plain text with references
            print(f"\n📖 Answer: {result['answer']}\n")
            
            # Print references (only in Layer 1 mode)
            if result['sources']:
                print("📚 Retrieved Sections:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   [{i}] {source['section']} (Section {source['section_num']})")
                print()
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}\n")