# Local RAG Baseline

Minimal local RAG system optimized for MacOS M3 Pro.

## Quick Start

```bash
# Setup
bash setup.sh

# Add PDFs
cp your_documents.pdf docs/

# Run
source .venv/bin/activate
python RAG.py
```

## How It Works

1. Downloads Gemma-2-9B-It model (~5.4GB, first run only)
2. Loads PDFs from `docs/` folder
3. Creates vector database in `docs/chroma/`
4. Starts Q&A CLI

## Usage

```
You: What is this about?
Answer: [AI answers based on your PDFs]

📄 document.pdf:p5, guide.pdf:p12

You: exit
```

## Requirements

- Python 3.10+
- 8GB+ RAM
- ~6GB disk space
- MacOS M3 Pro (Metal GPU optimized)

**Note**: This project is hardcoded for M3 Metal GPU (`device='mps'`). For other platforms:
- **Intel Mac/Linux/Windows**: Change `device='mps'` to `device='cpu'` in RAG.py
- **NVIDIA GPU**: Change to `device='cuda'`

## Tech Stack

- **LLM**: Gemma-2-9B-It Q4_K_M (llama.cpp + Metal)
- **Embeddings**: BAAI/bge-small-en-v1.5
- **Vector DB**: ChromaDB
- **Framework**: LangChain

## Configuration

Edit [`RAG.py`](RAG.py):

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
N_GPU_LAYERS = 35
N_CTX = 4096
N_BATCH = 1024
```

## Troubleshooting

**No PDFs found**
```bash
ls docs/*.pdf
```

**Out of memory**
```python
# In RAG.py:
N_GPU_LAYERS = 20
N_CTX = 2048
N_BATCH = 512
```

**Metal support issues**
```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
```

## License

MIT
