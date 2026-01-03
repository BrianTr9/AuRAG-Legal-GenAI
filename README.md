# AuRAG: Auditable Retrieval-Augmented Generation

**Enhancing Trustworthiness via Hierarchical Retrieval and Retrieval-Dependent Grammars**

## Overview

AuRAG is a research project designed to address critical challenges in deploying RAG systems for high-stakes domains such as Law and Healthcare. Current systems often suffer from **citation hallucinations** (plausible but non-existent references) and rely on "black-box" verification that compromises data privacy.

This project proposes a **Defense-in-Depth architecture** that moves from probabilistic "soft constraints" (prompt engineering) to deterministic "hard control" over generation, ensuring structural integrity and enabling independent auditing.

## Core Methodology

The system is built upon three distinct layers designed to work in unison:

### Layer 1: Hierarchical Indexing (Retrieval Optimization)
*   **Mechanism**: Implements a "Small-to-Big" retrieval strategy.
*   **Process**: Documents are parsed into a tree structure (e.g., Statute → Provision → Atomic Proposition). Dense retrieval targets granular "Information Nuggets" (child nodes) for high precision, which are then mapped to their parent contexts for generation.
*   **Goal**: Maximize retrieval specificity while maintaining contextual integrity.

### Layer 2: Grammar-Constrained Decoding (Generative Hard-Constraint)
*   **Mechanism**: Utilizes **Retrieval-Dependent Grammars (RDG)**.
*   **Process**: The grammar is constructed on-the-fly based on retrieved documents. The LLM is restricted at the token-sampling level to only generate citations that exist in the retrieved context.
*   **Goal**: Mathematically eliminate out-of-context citations during the generation phase.

### Layer 3: Deterministic Verification (Automated Auditing)
*   **Mechanism**: Post-generation auditing.
*   **Process**: A deterministic script verifies that every generated citation ID matches a record in the retrieved list.
*   **Goal**: Act as a final fail-safe to ensure every response is backed by a verifiable source.

## System Architecture

1.  **Ingestion**: Segment raw text into Parent/Child nodes -> Index in Vector DB.
2.  **Query**: User query processing and dense vector search.
3.  **Retrieval**: Fetch top-k child nodes and resolve to parent contexts.
4.  **Generation**: Local LLM generation constrained by RDG to ensure valid citations.
5.  **Validation**: Automated audit to accept or reject the response.


## Project Structure

The repository is organized to support this modular architecture:

*   `src/`: Core implementation of the three layers.
*   `data/`: Storage for raw corpora, processed chunks, and vector indices.
*   `benchmark/`: External datasets for evaluation.
*   `models/`: Local model weights.
*   `tests/`: Validation suites for each layer.

---
*This project is based on the Honours Thesis Proposal by Trung Bao Truong (2025).*
