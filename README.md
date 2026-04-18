# AuRAG: Auditable Retrieval-Augmented Generation

**Enhancing Trustworthiness via Structure-Preserving Hierarchical Retrieval (SPHR) and Retrieval-Dependent Grammars (RDG)**

This repository implements **AuRAG**, a specialized RAG framework designed for **high-stakes legal domains** (specifically statutory law). It mathematically eliminates citation fabrication (syntactic hallucination) and significantly enhances retrieval specificity through structure-aware document processing.

![Comparison](https://img.shields.io/badge/Status-Research%20Preview-blue) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 **Vision & Novelty**

### **The Problem: Citation Hallucinations**
Current RAG systems struggle with 4 types of citation errors:
1.  **Type 1 (Fabrication):** Citing a source that was never retrieved (pure hallucination).
2.  **Type 2 (Misattribution):** Citing a valid retrieved source that is irrelevant to the claim.
3.  **Type 3 (Reasoning Error):** Citing a relevant source but deriving flawed logic.
4.  **Type 4 (Retrieval Failure):** Missing the necessary evidence entirely.

### **The AuRAG Solution**
AuRAG introduces a two-layer defense architecture to address these:

-   **Layer 1: SPHR (Structure-Preserving Hierarchical Retrieval)**
    -   *Mitigates Type 4 (Retrieval Failure).*
    -   Indexes small child chunks for precise search, but retrieves full parent sections based on legal document structure (Sections/Articles).
    -   **Novelty:** Represents Domain Adaptation Novelty, leveraging the specific "legal document structure" to maximize retrieval specificity while maintaining contextual integrity.
    -   Ensures the LLM reasoning window contains complete statutory context, not just fragmented keywords.

-   **Layer 2: RDG (Retrieval-Dependent Grammars)**
    -   *Eliminates Type 1 (Fabrication) & Mitigates Type 2/3.*
    -   Constructs a **dynamic GBNF grammar** on-the-fly based *only* on the retrieved context.
    -   **Novelty:** Represents Technical/Architectural Novelty, solving the problem of Syntactic Correctness (guaranteeing valid citations). Unlike static grammars (IDG, Geng et al., 2023), AuRAG's grammar adapts to the retrieval set dynamically.
    -   Enforces **Structured Chain-of-Thought (CoT)**: `Premise → Inference → Conclusion` to improve reasoning quality.


---

## 🛠 **Technical Implementation**

### **Layer 1: Retrieval Stack**
-   **Engine:** `LangChain` + `ChromaDB`.
-   **Embeddings:** `BAAI/bge-small-en-v1.5`.
-   **Cross-Reference Enrichment:** Resolves internal pointers (e.g., "see §123") before embedding, ensuring chunks are self-contained (Regex-based).

### **Layer 2: Generative Stack**
-   **Inference:** `llama-cpp-python`.
-   **Model:** `Meta-Llama-3.1-8B-Instruct` (GGUF Quantization).
-   **Grammar:** Custom GBNF generator interacting directly with sampler logits to enforce citation constraints.

---

## 🚀 **Quick Start**

### **Prerequisites**
-   Python 3.10+
-   `make`, `cmake` (for llama.cpp build)

### **One-Click Setup (Recommended)**

AuRAG includes a setup script that creates the environment, downloads the GGUF model, and prepares the data format.

```bash
git clone https://github.com/yourusername/AuRAG.git
cd AuRAG
chmod +x setup.sh
./setup.sh
```

### **Manual Installation**

If you prefer to set up manually:

1.  **Environment & Dependencies**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Download Models**
    -   **LLM:** Download `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (or similar) into `models/`.
    -   **Embeddings:** AuRAG uses `BAAI/bge-small-en-v1.5` (auto-downloaded by HuggingFace).

3.  **Add Data**
    -   Create `data/raw/` and place your legal documents (`.pdf` or `.txt`) there.
    -   Other directories (`data/processed/`, `data/vector_db/`) are auto-created when you run the program.

### **Running the System**
```bash
python src/AuRAG.py
```
*The system will automatically:*
1.  Parse and hierarchically chunk your documents (SPHR).
2.  Build/Rebuild the local vector database (Chroma).
3.  Launch the interactive CLI for querying.

---

## 📊 **Evaluation (COLIEE 2025)**

AuRAG is evaluated on the **COLIEE task 3 (Statute Law)** benchmark.

### **Metrics**
-   **Citation Hallucination Rate (Type 1):** 0.0% (Guaranteed by RDG).
-   **Citation Precision/Recall:** Measures Type 2/3 errors against Ground Truth.
-   **Retrieval Hit@K / Recall@K:** Measures Layer 1 effectiveness.

### **Run Evaluation**
```bash
# Minimal CLI (quick run)
python3 evaluation/evaluate_end2end.py --year R06 --system aurag

# Complete CLI (reproducible, model-agnostic token budget)
python3 evaluation/evaluate_end2end.py \
    --corpus benchmark/COLIEE/civil.xml \
    --queries benchmark/COLIEE/simple/simple_R06_jp.xml \
    --year R06 \
    --system aurag \
    --embedding multilingual \
    --llm-model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --n-ctx 16384 \
    --max-tokens auto \
    --top-k 5 \
    --retrieval-mode hybrid \
    --bm25-weight 0.5 \
    --rrf-k 60 \
    --rebuild-index \
    --seed 42

# Layer-2 generation only (ideal retrieval)
python3 evaluation/evaluate_generation.py --mode rdg --llm-model <model.gguf> --n-ctx <ctx> --max-tokens auto
```

---

## 📁 **Project Structure**
```
AuRAG/
├── src/
│   ├── AuRAG.py       # Main pipeline entry point
│   ├── SPHR.py        # Layer 1: Hierarchical Chunking & Retrieval
│   └── RDG.py         # Layer 2: Grammar-Constrained Generation
├── evaluation/
│   ├── evaluate_end2end.py   # End-to-end (retrieval + generation) benchmark
│   ├── evaluate_generation.py # Layer-2 generation eval with GT retrieval
│   └── systems/              # System wrappers for evaluation
├── data/
│   ├── raw/           # Input documents
│   └── vector_db/     # Persistent ChromaDB
└── models/            # GGUF models
```

---

## **License**
MIT License. See [LICENSE](LICENSE) for details.

## **Author**
**Trung Bao (Brian) Truong**  
*Honours Thesis Project*
