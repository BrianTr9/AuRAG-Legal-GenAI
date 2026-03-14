#!/usr/bin/env python3
"""
Template 1: Retrieval-Only Evaluation (Layer 1)
================================================

Evaluates retrieval quality only (no generation) on COLIEE dataset.
Metrics: Recall@K, Precision@K, F1@K, Hit@K, MRR, NDCG@K.

Systems:
- sphr: Structure-Preserving Hierarchical Retrieval (AuRAG Layer 1)
- flat: Flat chunking baseline (fixed-size, no hierarchy)

Usage:
    # SPHR (Layer 1)
    python3 evaluation/evaluate_retrieval.py \
        --corpus benchmark/COLIEE/civil.xml \
        --queries benchmark/COLIEE/simple/simple_R06_jp.xml \
        --system sphr \
        --top-k 5 \
        --rebuild-index \
        --retrieval-mode hybrid

    # Flat baseline
    python3 evaluation/evaluate_retrieval.py \
        --corpus benchmark/COLIEE/civil.xml \
        --queries benchmark/COLIEE/simple/simple_R06_jp.xml \
        --system flat \
        --top-k 5 \
        --rebuild-index \
        --retrieval-mode hybrid

    # Fast smoke test (10 queries)
    python3 evaluation/evaluate_retrieval.py \
        --system sphr --top-k 5 --sample-queries 10
"""

import argparse
import json
import random
import shutil
import math
import re
from pathlib import Path
from typing import Dict, List, Any
from statistics import mean, stdev
import xml.etree.ElementTree as ET

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

# Add src/ to path to import AuRAG modules
import sys
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from SPHR import HierarchicalChunker, HierarchicalRetriever

try:
    import tiktoken
except Exception:
    tiktoken = None


def load_coliee_corpus(xml_path: Path) -> Dict[str, Dict[str, str]]:
    """Load COLIEE corpus from civil.xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    corpus = {}
    for article in root.findall('Article'):
        article_num = article.get('num')
        caption_elem = article.find('caption')
        caption = caption_elem.text.strip() if caption_elem is not None and caption_elem.text else ""
        text_elem = article.find('text')
        article_text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""
        if article_num and article_text:
            corpus[article_num] = {
                'caption': article_num,
                'text': article_text
            }
    return corpus


def load_coliee_queries(xml_path: Path) -> List[Dict[str, Any]]:
    """Load COLIEE entailment queries from simple_R0X_jp.xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    queries = []
    for pair in root.findall('pair'):
        pair_id = pair.get('id')
        ground_truth = (pair.get('label') or 'N').strip()
        t2_elem = pair.find('t2')
        question = "".join(t2_elem.itertext()).strip() if t2_elem is not None else ""
        
        # Relevant articles - use itertext() to handle nested elements, allow hyphens like "424-4"
        relevant_articles: List[str] = []
        for art in pair.findall('article'):
            val = ("".join(art.itertext()) if art is not None else "").strip()
            if len(val) > 0:
                relevant_articles.append(val)
        
        queries.append({
            'query_id': pair_id,
            'question': question,
            'ground_truth_answer': ground_truth,
            'relevant_articles': relevant_articles
        })
    return queries


def chunk_flat(text: str, contract_id: str, chunk_size: int = 300, overlap: int = 90) -> List[Document]:
    """Flat chunking baseline aligned to token budget when possible."""
    if not text:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)

    chunk_idx = 0

    if tiktoken is not None:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        for i in range(0, len(tokens), step):
            token_slice = tokens[i:i + chunk_size]
            if not token_slice:
                continue
            chunk_text = encoding.decode(token_slice)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    'chunk_type': 'flat',
                    'contract_id': contract_id,
                    'chunk_id': f"{contract_id}_flat_{chunk_idx}",
                }
            ))
            chunk_idx += 1
        return chunks

    # Fallback: word-based chunking if tokenizer is unavailable
    words = text.split()
    if not words:
        return []
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            continue
        chunk_text = " ".join(chunk_words)
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                'chunk_type': 'flat',
                'contract_id': contract_id,
                'chunk_id': f"{contract_id}_flat_{chunk_idx}",
            }
        ))
        chunk_idx += 1
    return chunks


class FlatRetriever:
    """Flat retriever with dense/bm25/hybrid ranking on flat chunks."""

    def __init__(
        self,
        vectordb,
        flat_documents: List[Document],
        top_k: int = 5,
        retrieval_mode: str = 'dense',
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.vectordb = vectordb
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode.lower()
        self.bm25_weight = max(0.0, min(1.0, bm25_weight))
        self.dense_weight = 1.0 - self.bm25_weight
        self.rrf_k = max(1, int(rrf_k))

        if self.retrieval_mode not in {'dense', 'hybrid'}:
            raise ValueError(
                f"Invalid retrieval_mode='{retrieval_mode}'. Choose from: dense, hybrid"
            )

        self.chunk_id_to_doc: Dict[str, Document] = {}
        self.chunk_ids: List[str] = []
        for doc in flat_documents:
            chunk_id = doc.metadata.get('chunk_id')
            if not chunk_id:
                continue
            if chunk_id not in self.chunk_id_to_doc:
                self.chunk_id_to_doc[chunk_id] = doc
                self.chunk_ids.append(chunk_id)

        tokenized = [self._bm25_tokenize(self.chunk_id_to_doc[cid].page_content) for cid in self.chunk_ids]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def _bm25_tokenize(text: str) -> List[str]:
        if not text:
            return []

        s = str(text).lower()
        tokens: List[str] = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", s)

        cjk_spans = re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+", s)
        for span in cjk_spans:
            if len(span) == 1:
                tokens.append(span)
            else:
                tokens.extend(span[i:i+2] for i in range(len(span) - 1))

        return tokens

    def _dense_chunk_ids(self, query: str) -> List[str]:
        docs = self.vectordb.similarity_search(query, k=self.top_k)
        seen = set()
        ranked = []
        for doc in docs:
            chunk_id = doc.metadata.get('chunk_id')
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                ranked.append(chunk_id)
        return ranked[:self.top_k]

    def _bm25_chunk_ids(self, query: str) -> List[str]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._bm25_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        return [self.chunk_ids[idx] for idx in ranked_indices[:self.top_k]]

    def _rank_chunk_ids(self, query: str) -> List[str]:
        if self.retrieval_mode == 'dense':
            return self._dense_chunk_ids(query)

        dense_ids = self._dense_chunk_ids(query)
        bm25_ids = self._bm25_chunk_ids(query)

        if not bm25_ids:
            return dense_ids

        scores: Dict[str, float] = {}
        for rank, chunk_id in enumerate(dense_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + self.dense_weight / (self.rrf_k + rank)
        for rank, chunk_id in enumerate(bm25_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + self.bm25_weight / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:self.top_k]]

    def invoke(self, query: str) -> List[Document]:
        ranked_ids = self._rank_chunk_ids(query)
        return [self.chunk_id_to_doc[cid] for cid in ranked_ids if cid in self.chunk_id_to_doc]


def build_sphr_index(
    corpus: Dict[str, Dict[str, str]],
    embedding_model: str,
    child_chunk_size: int,
    child_chunk_overlap: int,
    top_k: int,
    context_budget: int,
    persist_dir: Path,
    device: str,
    normalize_embeddings: bool,
    rebuild_index: bool,
    retrieval_mode: str,
    bm25_weight: float,
    rrf_k: int,
):
    """Build SPHR index and retriever."""
    chunker = HierarchicalChunker(child_size=child_chunk_size, child_overlap=child_chunk_overlap)
    all_parent_chunks = []
    all_child_chunks = []
    parent_to_children = {}

    for article_id, article_data in corpus.items():
        article_text = article_data['text']
        article_num = article_data['caption']
        parent_chunks, child_chunks = chunker.chunk_contract(article_text, contract_id=article_num)
        all_parent_chunks.extend(parent_chunks)
        all_child_chunks.extend(child_chunks)
        for parent in parent_chunks:
            parent_id = parent.chunk_id
            parent_to_children[parent_id] = [
                child.chunk_id for child in child_chunks
                if child.parent_id == parent_id
            ]

    # Enrich parent-child graph with cross-references (SPHR intent)
    parent_to_children = chunker.enrich_parent_to_children_with_cross_refs(
        parent_to_children,
        all_parent_chunks,
        all_child_chunks
    )

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

    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': normalize_embeddings}
    )

    if rebuild_index and persist_dir.exists():
        shutil.rmtree(persist_dir)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=str(persist_dir)
    )

    retriever = HierarchicalRetriever(
        vectordb,
        all_parent_chunks,
        parent_to_children,
        top_k=top_k,
        context_budget=context_budget,
        retrieval_mode=retrieval_mode,
        bm25_weight=bm25_weight,
        rrf_k=rrf_k,
    )

    return retriever


def build_flat_index(
    corpus: Dict[str, Dict[str, str]],
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    top_k: int,
    retrieval_mode: str,
    bm25_weight: float,
    rrf_k: int,
    persist_dir: Path,
    device: str,
    normalize_embeddings: bool,
    rebuild_index: bool
):
    """Build flat baseline index."""
    documents = []
    for article_id, article_data in corpus.items():
        article_text = article_data['text']
        article_num = article_data['caption']
        documents.extend(chunk_flat(article_text, contract_id=article_num, chunk_size=chunk_size, overlap=overlap))

    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': normalize_embeddings}
    )

    if rebuild_index and persist_dir.exists():
        shutil.rmtree(persist_dir)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=str(persist_dir)
    )

    return FlatRetriever(
        vectordb=vectordb,
        flat_documents=documents,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        bm25_weight=bm25_weight,
        rrf_k=rrf_k,
    )


def calculate_retrieval_metrics(retrieved: List[str], relevant: List[str]) -> Dict[str, float]:
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    overlap = retrieved_set & relevant_set

    hit = 1.0 if overlap else 0.0
    precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(overlap) / len(relevant_set) if relevant_set else 0.0

    # F1@K
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    # MRR: rank of first relevant
    mrr = 0.0
    if relevant_set:
        for idx, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                mrr = 1.0 / (idx + 1)
                break

    # NDCG@K for binary relevance
    dcg = 0.0
    for idx, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(relevant_set), len(retrieved))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits)) if ideal_hits > 0 else 0.0
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        'hit@k': hit,
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1,
        'mrr': mrr,
        'ndcg@k': ndcg,
        'num_retrieved': len(retrieved_set),
        'num_gt': len(relevant_set),
        'num_overlap': len(overlap)
    }


def main():
    parser = argparse.ArgumentParser(description="Retrieval-Only Evaluation (Layer 1)")
    parser.add_argument('--corpus', type=str, default='benchmark/COLIEE/civil.xml')
    parser.add_argument('--queries', type=str, default='benchmark/COLIEE/simple/simple_R06_jp.xml')
    parser.add_argument('--system', type=str, default='sphr', choices=['sphr', 'flat'])
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--embedding', type=str, default='intfloat/multilingual-e5-small')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--no-normalize-embeddings', action='store_false', dest='normalize_embeddings')
    parser.add_argument('--child-chunk-size', type=int, default=300)
    parser.add_argument('--child-chunk-overlap', type=int, default=90)
    parser.add_argument('--retrieval-mode', type=str, default='dense', choices=['dense', 'hybrid'])
    parser.add_argument('--bm25-weight', type=float, default=0.5)
    parser.add_argument('--rrf-k', type=int, default=60)
    parser.add_argument('--rebuild-index', action='store_true', default=False)
    parser.add_argument('--sample-queries', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    if not hasattr(args, 'normalize_embeddings'):
        args.normalize_embeddings = True
    elif args.normalize_embeddings is None:
        args.normalize_embeddings = True

    random.seed(args.seed)

    corpus = load_coliee_corpus(Path(args.corpus))
    queries = load_coliee_queries(Path(args.queries))

    if args.sample_queries and args.sample_queries > 0:
        queries = random.sample(queries, min(args.sample_queries, len(queries)))

    output_dir = Path(__file__).parent

    # Retrieval-only evaluation: always disable context truncation to isolate ranking quality.
    context_budget = 10**9
    if args.system == 'sphr':
        retriever = build_sphr_index(
            corpus,
            args.embedding,
            args.child_chunk_size,
            args.child_chunk_overlap,
            args.top_k,
            context_budget,
            output_dir / 'vector_db_sphr',
            args.device,
            args.normalize_embeddings,
            args.rebuild_index,
            args.retrieval_mode,
            args.bm25_weight,
            args.rrf_k,
        )
    else:
        retriever = build_flat_index(
            corpus,
            args.embedding,
            args.child_chunk_size,
            args.child_chunk_overlap,
            args.top_k,
            args.retrieval_mode,
            args.bm25_weight,
            args.rrf_k,
            output_dir / 'vector_db_flat',
            args.device,
            args.normalize_embeddings,
            args.rebuild_index
        )

    per_query_results = []
    hit_scores, prec_scores, rec_scores, f1_scores, mrr_scores, ndcg_scores = [], [], [], [], [], []

    def collect_unique_articles(docs: List[Document]) -> List[str]:
        retrieved = []
        for d in docs:
            doc_id = d.metadata.get('contract_id', 'unknown')
            if doc_id not in retrieved:
                retrieved.append(doc_id)
        return retrieved

    for q in queries:
        question = q['question']
        relevant = q['relevant_articles']

        docs = retriever.invoke(question)
        retrieved = collect_unique_articles(docs)

        metrics = calculate_retrieval_metrics(retrieved, relevant)
        hit_scores.append(metrics['hit@k'])
        prec_scores.append(metrics['precision@k'])
        rec_scores.append(metrics['recall@k'])
        f1_scores.append(metrics['f1@k'])
        mrr_scores.append(metrics['mrr'])
        ndcg_scores.append(metrics['ndcg@k'])

        per_query_results.append({
            'query_id': q['query_id'],
            'ground_truth': relevant,
            'retrieved': retrieved,
            'metrics': metrics
        })

    aggregate = {
        'hit@k': {'mean': mean(hit_scores), 'std': stdev(hit_scores) if len(hit_scores) > 1 else 0.0},
        'precision@k': {'mean': mean(prec_scores), 'std': stdev(prec_scores) if len(prec_scores) > 1 else 0.0},
        'recall@k': {'mean': mean(rec_scores), 'std': stdev(rec_scores) if len(rec_scores) > 1 else 0.0},
        'f1@k': {'mean': mean(f1_scores), 'std': stdev(f1_scores) if len(f1_scores) > 1 else 0.0},
        'mrr': {'mean': mean(mrr_scores), 'std': stdev(mrr_scores) if len(mrr_scores) > 1 else 0.0},
        'ndcg@k': {'mean': mean(ndcg_scores), 'std': stdev(ndcg_scores) if len(ndcg_scores) > 1 else 0.0}
    }

    output = {
        'metadata': {
            'system': args.system,
            'embedding': args.embedding,
            'top_k': args.top_k,
            'num_queries': len(per_query_results),
            'child_chunk_size': args.child_chunk_size,
            'child_chunk_overlap': args.child_chunk_overlap,
            'context_budget': context_budget,
            'retrieval_mode': args.retrieval_mode,
            'bm25_weight': args.bm25_weight,
            'rrf_k': args.rrf_k,
            'device': args.device,
            'normalize_embeddings': args.normalize_embeddings,
            'rebuild_index': args.rebuild_index,
            'seed': args.seed
        },
        'aggregate_metrics': aggregate,
        'per_query_results': per_query_results
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"evaluation_results_layer1_retrieval_{args.system}_k{args.top_k}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Retrieval evaluation complete: {output_path}")


if __name__ == '__main__':
    main()
