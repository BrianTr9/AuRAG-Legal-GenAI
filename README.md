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
    -   Leverages legal document structure (Sections/Articles) to index small child chunks for precise search, but retrieves full parent sections for context integrity.
    -   Ensures the LLM reasoning window contains complete statutory context, not just fragmented keywords.

-   **Layer 2: RDG (Retrieval-Dependent Grammars)**
    -   *Eliminates Type 1 (Fabrication) & Mitigates Type 2/3.*
    -   Constructs a **dynamic GBNF grammar** on-the-fly based *only* on the retrieved context.
    -   **Novelty:** Unlike static grammars IDG (Geng et al., 2023), AuRAG's grammar adapts to the retrieval set. If Article 30 and 31 are retrieved, the model is mathematically constrained to cite *only* {Article 30, Article 31} or nothing. It cannot physically generate a hallucinated "Article 99".
    -   Enforces **Structured Chain-of-Thought (CoT)**: `Premise → Inference → Conclusion` to improve reasoning quality.

---

## 🏗 **Architecture**

### **1. SPHR (Layer 1)**
-   **Hierarchical Indexing:** Splits documents into Parent Sections (full context) and Child Chunks (searchable units).
-   **Cross-Reference Enrichment:** Resolves "see §X.X" references during indexing so chunks stand alone.
-   **Stack:** `LangChain`, `ChromaDB`, `HuggingFaceEmbeddings` (MPS-accelerated).

### **2. RDG (Layer 2)**
-   **Constrained Decoding:** Uses `llama-cpp-python` with GBNF grammars.
-   **Dynamic Constraint:** The valid token set for citation fields is restricted to the IDs of retrieved documents at inference time.
-   **Stack:** `Llama-3.1-8B-Instruct (GGUF)`, `llama.cpp`.

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

## 📊 **Evaluation (COLIEE 2024)**

AuRAG is evaluated on the **COLIEE 2024 (Statute Law)** benchmark.

### **Metrics**
-   **Citation Hallucination Rate (Type 1):** 0.0% (Guaranteed by RDG).
-   **Citation Precision/Recall:** Measures Type 2/3 errors against Ground Truth.
-   **Retrieval Hit@K / Recall@K:** Measures Layer 1 effectiveness.

### **Run Evaluation**
```bash
python evaluation/evaluate_citation.py --year R06 --system aurag --top-k 5
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
│   ├── evaluate_citation.py  # COLIEE benchmark runner
│   └── systems/              # System wrappers for evaluation
├── data/
│   ├── raw/           # Input documents
│   └── vector_db/     # Persistent ChromaDB
└── models/            # GGUF models
```

---

## 📜 **License**
MIT License. See [LICENSE](LICENSE) for details.

## ✍️ **Author**
**Trung Bao (Brian) Truong**  
*Honours Thesis Project*
