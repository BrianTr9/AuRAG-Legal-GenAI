"""
Local RAG Baseline
==================
Minimal local RAG implementation:
- LLM: Gemma-2-9B-It (llama.cpp)
- Embeddings: BAAI/bge-small-en-v1.5
- Vector DB: ChromaDB (local)
"""

import os
import glob
from huggingface_hub import hf_hub_download
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler, CallbackManager
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# ==========================================
# CONFIGURATION
# ==========================================
DOCS_FOLDER = "data/raw"
CHROMA_DIR = "data/vector_db/chroma/"
MODELS_DIR = "models"

LLM_REPO = "lmstudio-community/gemma-2-9b-it-GGUF"
LLM_FILENAME = "gemma-2-9b-it-Q4_K_M.gguf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

# M3 Pro Optimization
USE_METAL_GPU = True  # Metal acceleration for M3
N_GPU_LAYERS = 35     # Layers on GPU (adjust based on model size)
N_CTX = 4096          # Larger context for M3 Pro
N_BATCH = 1024        # Larger batch for faster processing

# ==========================================
# 1. LOAD MODEL
# ==========================================
print("Downloading model...")
llm_model_path = hf_hub_download(
    repo_id=LLM_REPO,
    filename=LLM_FILENAME,
    cache_dir=MODELS_DIR,
    resume_download=True,
)
print(f"Model ready: {llm_model_path}")

# ==========================================
# 2. LOAD DOCUMENTS
# ==========================================
print("\nLoading PDFs...")
pdf_files = glob.glob(f'{DOCS_FOLDER}/**/*.pdf', recursive=True)

if not pdf_files:
    raise FileNotFoundError(f"No PDFs found in {DOCS_FOLDER}/")

docs = []
for pdf_path in pdf_files:
    docs.extend(PyPDFLoader(pdf_path).load())

print(f"Loaded {len(docs)} pages from {len(pdf_files)} PDFs")

# ==========================================
# 3. SPLIT DOCUMENTS
# ==========================================
print("\nSplitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks")

# ==========================================
# 4. CREATE EMBEDDINGS & VECTOR DB
# ==========================================
print("\nCreating embeddings...")
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'mps'},  # Metal Performance Shaders for M3
    encode_kwargs={'normalize_embeddings': True}
)

print(f"Building vector database from {len(splits)} chunks...")
print("This may take a few minutes...")
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=CHROMA_DIR,
)
print(f"✓ Saved {vectordb._collection.count()} vectors to {CHROMA_DIR}")

# ==========================================
# 5. INITIALIZE LLM
# ==========================================
print("\nLoading LLM...")
llm = LlamaCpp(
    model_path=llm_model_path,
    temperature=0,
    max_tokens=512,
    n_ctx=N_CTX,
    n_batch=N_BATCH,
    n_gpu_layers=N_GPU_LAYERS if USE_METAL_GPU else 0,  # Metal GPU acceleration
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=False,
)

# ==========================================
# 6. CREATE QA CHAIN
# ==========================================
print("Creating QA chain...")
template = (
    "Use the following context to answer the question.\n"
    "If you don't know, say you don't know.\n"
    "Keep the answer concise.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
prompt = PromptTemplate.from_template(template)

retriever = vectordb.as_retriever(
    search_type='mmr',
    search_kwargs={'k': TOP_K}
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt},
)

print("System ready!\n")

# ==========================================
# 7. CLI
# ==========================================
print("="*50)
print("RAG - Type 'exit' to quit")
print("="*50)

while True:
    try:
        query = input("\nYou: ").strip()
        
        if not query or query.lower() in ["exit", "quit"]:
            break
        
        result = qa_chain.invoke({'query': query})
        print(f"\nAnswer: {result['result']}")
        
        # Show sources
        if result.get('source_documents'):
            sources = [(doc.metadata.get('source', 'Unknown'), doc.metadata.get('page', '?')) 
                       for doc in result['source_documents']]
            print(f"\n📄 {', '.join([f'{os.path.basename(s)}:p{p}' for s, p in sources])}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")