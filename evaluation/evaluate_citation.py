"""
Template 2: End-to-End RAG Evaluation with Citation Hallucination Measurement
=============================================================================

Evaluates complete RAG systems (retrieval + generation) on:
1. Citation Hallucination Rate: How many cited articles were NOT retrieved?
2. Answer Correctness: Y/N accuracy vs ground truth
3. Latency: Total time per query (retrieval + generation)

This script loads COLIEE entailment pairs (Y/N questions + correct article IDs)
and measures whether RAG systems cite only retrieved articles (hallucination-free).
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET
from statistics import mean, stdev

from systems.base import RAGSystem
from systems.AuRAGSystem import AuRAGSystem


def normalize_yn_answer(text: str) -> str | None:
    """Normalize a free-form model answer to a COLIEE-style label: 'Y' or 'N'.

    Returns:
        'Y' if the text clearly indicates entailment/yes/true,
        'N' if the text clearly indicates non-entailment/no/false,
        None if ambiguous.

    Notes:
        - We prefer explicit leading labels like "Y" / "N".
        - Supports common English and Japanese variants.
        - This is intentionally conservative: if both signals appear, returns None.
    """
    if not text:
        return None

    s = " ".join(str(text).strip().split()).lower()
    if not s:
        return None

    # Strong signal: starts with a label
    # e.g., "Y", "N", "Y:", "Answer: Y"
    prefixes_y = ("y", "yes", "entails", "entailed", "entailment", "true")
    prefixes_n = ("n", "no", "not entail", "non-entail", "contradict", "false")
    for p in ("answer:", "label:", "prediction:"):
        if s.startswith(p):
            s = s[len(p):].lstrip()
            break
    if s.startswith("y") and (len(s) == 1 or s[1] in ": )].,;"):
        return "Y"
    if s.startswith("n") and (len(s) == 1 or s[1] in ": )].,;"):
        return "N"

    # Token-based heuristics
    y_markers = {
        "yes", "y", "entailed", "entails", "entailment", "true", "correct", "supported",
        "is entailed", "is supported",
        "はい", "該当", "成立", "認められる",
    }
    n_markers = {
        "no", "n", "not", "not entailed", "non-entailment", "contradiction", "false", "incorrect", "unsupported",
        "is not entailed", "is not supported",
        "いいえ", "非該当", "成立しない", "認められない",
    }

    has_y = any(m in s for m in y_markers)
    has_n = any(m in s for m in n_markers)

    if has_y and not has_n:
        return "Y"
    if has_n and not has_y:
        return "N"
    return None


def load_coliee_corpus(xml_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load COLIEE corpus from civil.xml.
    
    Args:
        xml_path: Path to civil.xml
        
    Returns:
        Dict mapping article_id -> {'caption': article_number, 'text': full_text}
    """
    print(f"\n📂 Loading corpus from {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    corpus = {}
    for article in root.findall('Article'):  # Capital 'A'
        article_num = article.get('num')  # Use 'num' attribute
        
        # Extract caption
        caption_elem = article.find('caption')
        caption = caption_elem.text.strip() if caption_elem is not None and caption_elem.text else ""
        
        # Extract text
        text_elem = article.find('text')
        article_text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""
        
        if article_num and article_text:
            corpus[article_num] = {
                'caption': article_num,  # Use article number as caption
                'text': article_text
            }
    
    print(f"  ✓ Loaded {len(corpus)} articles")
    return corpus


def load_coliee_queries(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Load COLIEE entailment queries from simple_R0X_jp.xml.
    
    Args:
        xml_path: Path to simple_R0X_jp.xml
        
    Returns:
        List of queries with format:
        [
            {
                'query_id': 'R06-01-A',
                'question': 'Japanese legal statement...',
                'ground_truth_answer': 'Y' or 'N',
                'relevant_articles': ['11', '7']  # Article numbers from <article> tags
            },
            ...
        ]
    """
    print(f"\n📂 Loading queries from {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    queries = []
    for pair in root.findall('pair'):
        pair_id = pair.get('id')

        # Ground-truth label is on the <pair> element
        ground_truth = (pair.get('label') or 'N').strip()

        # The statement itself is in <t2> (no <t1> in these COLIEE 'simple' files)
        t2_elem = pair.find('t2')
        question = "".join(t2_elem.itertext()).strip() if t2_elem is not None else ""

        # Relevant articles are explicitly listed as <article>11</article>
        relevant_articles: List[str] = []
        for art in pair.findall('article'):
            val = ("".join(art.itertext()) if art is not None else "").strip()
            # Fix: Allow hyphens for articles like "424-4", "98-2"
            if len(val) > 0:
                relevant_articles.append(val)
        
        queries.append({
            'query_id': pair_id,
            'question': question,
            'ground_truth_answer': ground_truth,
            'relevant_articles': relevant_articles
        })
    
    print(f"  ✓ Loaded {len(queries)} queries")
    return queries


def calculate_citation_metrics(
    citations: List[str],
    retrieved: List[str],
    relevant_articles: List[str]
) -> Dict[str, float]:
    """
    Calculate citation hallucination and retrieval metrics.
    
    CRITICAL: Hallucination is SYNTACTIC check (cited but NOT retrieved),
    not semantic check (cited but not relevant).
    
    Args:
        citations: Article IDs cited in the generated answer (from LLM output)
        retrieved: Article IDs retrieved by the system (from retrieval step)
        relevant_articles: Ground truth relevant article IDs (from COLIEE labels)
        
    Returns:
        {
            # --- Type 1 (Fabrication / Syntactic)
            'citation_hallucination_rate': float,
            'fabrication_rate': float,  # alias of citation_hallucination_rate

            # --- Type 2 (Misattribution / Non-GT among retrieved)
            'misattribution_rate': float,  # |(C ∩ R) \ GT| / |C|
            'misattribution_rate_conditional': float,  # |(C ∩ R) \ GT| / |C ∩ R|
            'non_gt_citation_rate': float,  # alias of misattribution_rate

            # --- Standard evidence metrics
            'citation_precision': float,
            'citation_recall': float,

            # --- Counts (for debugging and analysis)
            'num_citations': int,
            'num_retrieved': int,
            'num_valid_retrieved_citations': int,  # |C ∩ R|
            'num_hallucinated_citations': int,      # |C \ R|
            'num_misattributed_citations': int,     # |(C ∩ R) \ GT|
        }
    
    Example:
        retrieved = ["123", "456", "789"]
        cited = ["123", "456", "999"]
        relevant = ["123", "789"]
        
        hallucinated = {"999"}  # cited but NOT retrieved (syntactic violation)
        hallucination_rate = 1/3 = 0.333 (33.3%)
        
        correct_citations = {"123"}  # cited AND relevant
        citation_precision = 1/3 = 0.333 (33.3%)
        citation_recall = 1/2 = 0.5 (50%)
    """
    # Convert to sets for efficient set operations
    cited_set = set(citations)
    retrieved_set = set(retrieved)
    relevant_set = set(relevant_articles)
    
    # PRIMARY METRIC: Citation Hallucination Rate (TYPE 1: FABRICATION / SYNTACTIC CHECK)
    # Definition: Citations that appear in answer but were NOT in retrieved set
    # This is a HARD CONSTRAINT violation - system cited articles it never retrieved
    hallucinated_citations = cited_set - retrieved_set  # Set difference: cited \ retrieved
    hallucination_rate = len(hallucinated_citations) / len(cited_set) if len(cited_set) > 0 else 0.0
    
    # NEW METRIC: Misattribution Rate (TYPE 2: SEMANTIC HALLUCINATION)
    # Definition: Citations that are VALIDLY retrieved (syntactically correct) but NOT IN GT.
    # NOTE: In legal tasks, GT can be incomplete (alternative valid statutes may exist). For paper writing,
    # interpret this as "Non-GT citation rate" unless you validate GT completeness.
    #
    # We report two versions:
    # - misattribution_rate: overall, normalized by |C|
    # - misattribution_rate_conditional: conditioned on syntactically-valid citations |C ∩ R|
    validly_retrieved_citations = cited_set & retrieved_set
    misattributed_citations = validly_retrieved_citations - relevant_set
    # Metric: Misattribution Rate Conditional (TYPE 2: SEMANTIC ERROR)
    # Definition: Of the citations that were valid (in retrieved set), how many were NOT in GT?
    # This isolates semantic selection error from syntactic hallucination.
    misattribution_rate_conditional = (
        len(misattributed_citations) / len(validly_retrieved_citations)
        if len(validly_retrieved_citations) > 0 else 0.0
    )

    # SECONDARY METRIC: Citation Precision (SEMANTIC CHECK)
    # Definition: What percentage of citations are actually relevant (correct)?
    # This measures answer quality, not constraint violation
    correct_citations = cited_set & relevant_set  # Set intersection: cited ∩ relevant
    citation_precision = len(correct_citations) / len(cited_set) if len(cited_set) > 0 else 0.0
    
    # SECONDARY METRIC: Citation Recall (COVERAGE CHECK)
    # Definition: What percentage of relevant articles were cited?
    # This measures completeness of answer
    citation_recall = len(correct_citations) / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    return {
        'citation_hallucination_rate': hallucination_rate,  # Core thesis metric (Fabrication/Type 1)
        'fabrication_rate': hallucination_rate,             # Explicit alias for Type 1
        'misattribution_rate_conditional': misattribution_rate_conditional,  # Type 2 conditioned on |C ∩ R|
        
        'citation_precision': citation_precision,
        'citation_recall': citation_recall,
        'num_citations': len(cited_set),
        'num_valid_retrieved_citations': len(validly_retrieved_citations),
        'num_hallucinated_citations': len(hallucinated_citations),
        'num_misattributed_citations': len(misattributed_citations),
        'num_retrieved': len(retrieved_set)
    }


def calculate_retrieval_metrics(retrieved: List[str], relevant_articles: List[str]) -> Dict[str, float]:
    """Calculate retrieval quality against COLIEE GT citations.

    These metrics explain upstream failures (missed GT articles) that cap generation/citation recall.
    """
    retrieved_set = set(retrieved)
    relevant_set = set(relevant_articles)
    overlap = retrieved_set & relevant_set

    hit = 1.0 if overlap else 0.0
    precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(overlap) / len(relevant_set) if relevant_set else 0.0

    return {
        'retrieval_hit': hit,
        'retrieval_precision': precision,
        'retrieval_recall': recall,
        'num_retrieved': len(retrieved_set),
        'num_gt': len(relevant_set),
        'num_overlap': len(overlap),
    }


def evaluate_rag_system(
    system: RAGSystem,
    queries: List[Dict[str, Any]],
    system_name: str,
    sample_size: int = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate a RAG system on citation constraints and answer quality.
    
    Args:
        system: RAG system instance (implements RAGSystem interface)
        queries: List of query dictionaries
        system_name: Name for logging
        sample_size: Limit to first N queries (for testing)
        
    Returns:
        (aggregate_metrics, per_query_results)
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {system_name.upper()} System")
    print(f"{'='*70}")
    
    # Sample queries if requested
    if sample_size:
        queries = queries[:sample_size]
        print(f"\n📊 Testing on {len(queries)} sample queries")
    else:
        print(f"\n📊 Evaluating on {len(queries)} queries")
    
    per_query_results = []
    
    # Metrics to aggregate
    hallucination_rates = []
    misattribution_rates = []
    misattribution_rates_conditional = []
    citation_precisions = []
    citation_recalls = []
    answer_correctness = []
    answer_coverage = []
    retrieval_hits = []
    retrieval_precisions = []
    retrieval_recalls = []
    retrieval_times = []
    generation_times = []
    total_times = []
    
    for i, query_data in enumerate(queries, 1):
        query_id = query_data['query_id']
        question = query_data['question']
        ground_truth = query_data['ground_truth_answer']
        relevant_articles = query_data['relevant_articles']
        
        print(f"\n[{i}/{len(queries)}] Processing {query_id}...")
        print(f"  Question: {question[:100]}...")
        print(f"  GT Citations: {len(relevant_articles)} articles")
        
        # Query the system
        t_start = time.time()
        result = system.query(question)
        total_time = time.time() - t_start
        
        # Extract results
        answer = result.get('answer', '')
        citations = result.get('citations', [])
        retrieved = result.get('retrieved', [])
        retrieval_time = result.get('retrieval_time', 0.0)
        generation_time = result.get('generation_time', 0.0)

        predicted_label = normalize_yn_answer(answer)
        gt_label = (ground_truth or 'N').strip().upper()
        is_correct: int | None
        if predicted_label is None:
            is_correct = None
        else:
            is_correct = 1 if predicted_label == gt_label else 0
        
        print(f"  Retrieved: {len(retrieved)} articles")
        print(f"  Citations: {len(citations)} articles")
        
        # Calculate metrics
        metrics = calculate_citation_metrics(citations, retrieved, relevant_articles)
        retrieval_metrics = calculate_retrieval_metrics(retrieved, relevant_articles)
        
        print(f"  Hallucination (Type 1): {metrics['citation_hallucination_rate']*100:.1f}%")
        print(f"  Misattribution (Type 2): {metrics['misattribution_rate']*100:.1f}%")
        print(f"  Misattribution|Valid (Type 2): {metrics['misattribution_rate_conditional']*100:.1f}%")
        print(f"  Citation Precision: {metrics['citation_precision']*100:.1f}%")
        print(f"  Citation Recall: {metrics['citation_recall']*100:.1f}%")
        print(f"  Retrieval Hit@K: {retrieval_metrics['retrieval_hit']*100:.1f}%")
        print(f"  Retrieval Recall@K: {retrieval_metrics['retrieval_recall']*100:.1f}%")
        print(f"  Answer (GT→Pred): {gt_label} → {predicted_label if predicted_label is not None else 'UNK'}")
        # Debug-friendly inspection: show GT vs LLM citation IDs directly.
        gt_list = sorted(set(relevant_articles))
        llm_list = sorted(set(citations))
        print(f"  GT: {gt_list}")
        print(f"  LLM: {llm_list}")
        print(f"  Time: {total_time:.2f}s (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
        
        # Aggregate
        hallucination_rates.append(metrics['citation_hallucination_rate'])
        misattribution_rates.append(metrics['misattribution_rate'])
        misattribution_rates_conditional.append(metrics['misattribution_rate_conditional'])
        citation_precisions.append(metrics['citation_precision'])
        citation_recalls.append(metrics['citation_recall'])
        retrieval_hits.append(retrieval_metrics['retrieval_hit'])
        retrieval_precisions.append(retrieval_metrics['retrieval_precision'])
        retrieval_recalls.append(retrieval_metrics['retrieval_recall'])
        if is_correct is not None:
            answer_correctness.append(is_correct)
            answer_coverage.append(1)
        else:
            answer_coverage.append(0)
        retrieval_times.append(retrieval_time)
        generation_times.append(generation_time)
        total_times.append(total_time)
        
        # Store per-query result
        per_query_results.append({
            'query_id': query_id,
            'question': question,
            'ground_truth_answer': ground_truth,
            'system_answer': answer,
            'predicted_answer': predicted_label,
            'answer_correct': is_correct,
            # For easy inspection: make GT vs LLM citations explicit.
            # - GT citations: COLIEE-labeled relevant article numbers (<article> tags)
            # - LLM citations: auto-collected citations from RDG reasoning steps
            'gt_citations': relevant_articles,
            'llm_citations': citations,

            # Backward compatible keys (older output schema)
            'relevant_articles': relevant_articles,
            'retrieved_articles': retrieved,
            'cited_articles': citations,
            'metrics': metrics,
            'retrieval_metrics': retrieval_metrics,
            'timing': {
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'total_time': total_time
            }
        })
    
    # Compute aggregate statistics
    aggregate_metrics = {
        'citation_hallucination_rate': {  # Type 1: Fabrication
            'mean': mean(hallucination_rates),
            'std': stdev(hallucination_rates) if len(hallucination_rates) > 1 else 0.0,
            'min': min(hallucination_rates),
            'max': max(hallucination_rates)
        },
        'misattribution_rate': {  # Type 2: Semantic Hallucination
            'mean': mean(misattribution_rates) if misattribution_rates else 0.0,
            'std': stdev(misattribution_rates) if len(misattribution_rates) > 1 else 0.0,
            'min': min(misattribution_rates) if misattribution_rates else 0.0,
            'max': max(misattribution_rates) if misattribution_rates else 0.0
        },
        'misattribution_rate_conditional': {
            'mean': mean(misattribution_rates_conditional) if misattribution_rates_conditional else 0.0,
            'std': stdev(misattribution_rates_conditional) if len(misattribution_rates_conditional) > 1 else 0.0,
            'min': min(misattribution_rates_conditional) if misattribution_rates_conditional else 0.0,
            'max': max(misattribution_rates_conditional) if misattribution_rates_conditional else 0.0
        },
        'citation_precision': {
            'mean': mean(citation_precisions),
            'std': stdev(citation_precisions) if len(citation_precisions) > 1 else 0.0,
            'min': min(citation_precisions),
            'max': max(citation_precisions)
        },
        'citation_recall': {
            'mean': mean(citation_recalls),
            'std': stdev(citation_recalls) if len(citation_recalls) > 1 else 0.0,
            'min': min(citation_recalls),
            'max': max(citation_recalls)
        },
        'answer_accuracy': {
            'mean': mean(answer_correctness) if answer_correctness else 0.0,
            'std': stdev(answer_correctness) if len(answer_correctness) > 1 else 0.0,
            'num_scored': len(answer_correctness),
            'num_total': len(queries)
        },
        'answer_coverage': {
            # Fraction of queries where we could confidently parse a Y/N from the model output.
            'mean': mean(answer_coverage) if answer_coverage else 0.0,
            'num_scored': len(answer_correctness),
            'num_total': len(queries)
        },
        'retrieval_hit': {
            'mean': mean(retrieval_hits) if retrieval_hits else 0.0,
            'std': stdev(retrieval_hits) if len(retrieval_hits) > 1 else 0.0,
        },
        'retrieval_precision': {
            'mean': mean(retrieval_precisions) if retrieval_precisions else 0.0,
            'std': stdev(retrieval_precisions) if len(retrieval_precisions) > 1 else 0.0,
        },
        'retrieval_recall': {
            'mean': mean(retrieval_recalls) if retrieval_recalls else 0.0,
            'std': stdev(retrieval_recalls) if len(retrieval_recalls) > 1 else 0.0,
        },
        'retrieval_time': {
            'mean': mean(retrieval_times),
            'std': stdev(retrieval_times) if len(retrieval_times) > 1 else 0.0,
            'total': sum(retrieval_times)
        },
        'generation_time': {
            'mean': mean(generation_times),
            'std': stdev(generation_times) if len(generation_times) > 1 else 0.0,
            'total': sum(generation_times)
        },
        'total_time': {
            'mean': mean(total_times),
            'std': stdev(total_times) if len(total_times) > 1 else 0.0,
            'total': sum(total_times)
        }
    }
    
    print(f"\n{'='*70}")
    print(f"📊 AGGREGATE RESULTS")
    print(f"{'='*70}")
    print(f"Citation Hallucination Rate (Type 1): {aggregate_metrics['citation_hallucination_rate']['mean']*100:.2f}% ± {aggregate_metrics['citation_hallucination_rate']['std']*100:.2f}%")
    print(f"Misattribution Rate (Type 2):         {aggregate_metrics['misattribution_rate']['mean']*100:.2f}% ± {aggregate_metrics['misattribution_rate']['std']*100:.2f}%")
    print(f"Misattr.|Valid (Type 2):              {aggregate_metrics['misattribution_rate_conditional']['mean']*100:.2f}% ± {aggregate_metrics['misattribution_rate_conditional']['std']*100:.2f}%")
    print(f"Citation Precision:                   {aggregate_metrics['citation_precision']['mean']*100:.2f}% ± {aggregate_metrics['citation_precision']['std']*100:.2f}%")
    print(f"Citation Recall:                      {aggregate_metrics['citation_recall']['mean']*100:.2f}% ± {aggregate_metrics['citation_recall']['std']*100:.2f}%")

    print(f"Answer Accuracy: {aggregate_metrics['answer_accuracy']['mean']*100:.2f}% (scored {aggregate_metrics['answer_accuracy']['num_scored']}/{aggregate_metrics['answer_accuracy']['num_total']})")
    print(f"Retrieval Hit@K: {aggregate_metrics['retrieval_hit']['mean']*100:.2f}% ± {aggregate_metrics['retrieval_hit']['std']*100:.2f}%")
    print(f"Retrieval Recall@K: {aggregate_metrics['retrieval_recall']['mean']*100:.2f}% ± {aggregate_metrics['retrieval_recall']['std']*100:.2f}%")
    print(f"Average Total Time: {aggregate_metrics['total_time']['mean']:.2f}s ± {aggregate_metrics['total_time']['std']:.2f}s")
    print(f"{'='*70}\n")
    
    return aggregate_metrics, per_query_results


def save_results(
    output_path: Path,
    system_name: str,
    aggregate_metrics: Dict[str, Any],
    per_query_results: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> None:
    """Save evaluation results to JSON file"""
    results = {
        'metadata': {
            'system_name': system_name,
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(per_query_results),
            **metadata
        },
        'aggregate_metrics': aggregate_metrics,
        'per_query_results': per_query_results
    }
    
    # Delete old result file if it exists
    if output_path.exists():
        output_path.unlink()
        print(f"🗑️  Deleted old result: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Template 2: End-to-End RAG Evaluation")
    
    # Data paths
    parser.add_argument('--corpus', type=str, default='benchmark/COLIEE/civil.xml',
                       help='Path to corpus XML file')
    parser.add_argument('--queries', type=str, default='benchmark/COLIEE/simple/simple_R06_jp.xml',
                       help='Path to queries XML file')
    parser.add_argument('--year', type=str, default='R06',
                       help='COLIEE year identifier')
    
    # System configuration
    parser.add_argument('--system', type=str, default='aurag',
                       choices=['aurag'],
                       help='RAG system to evaluate')
    parser.add_argument('--embedding', type=str, default='multilingual',
                       help='Embedding model identifier')
    parser.add_argument('--llm-model', type=str,
                       default='models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
                       help='Path to LLM model (GGUF format) - Llama-3.1-8B for legal context')
    
    # Evaluation parameters
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of documents to retrieve')
    parser.add_argument('--sample-queries', type=int, default=None,
                       help='Limit to first N queries (for testing)')
    parser.add_argument('--rebuild-index', action='store_true',
                       help='Delete and rebuild the persisted vector DB before running (recommended for clean paper runs)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='Directory to save results')
    
    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    
    # Resolve paths
    workspace_root = Path(__file__).parent.parent
    corpus_path = workspace_root / args.corpus
    queries_path = workspace_root / args.queries
    output_dir = workspace_root / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    corpus = load_coliee_corpus(corpus_path)
    queries = load_coliee_queries(queries_path)
    
    # Embedding model mapping
    EMBEDDINGS_CONFIG = {
        'multilingual': 'intfloat/multilingual-e5-base',
        'jp-bge': 'BAAI/bge-base-ja-v1.5',
        'en-bge': 'BAAI/bge-small-en-v1.5'
    }
    embedding_model = EMBEDDINGS_CONFIG.get(args.embedding, args.embedding)
    
    # Setup system
    print(f"\n{'='*70}")
    print(f"TEMPLATE 2: END-TO-END RAG EVALUATION")
    print(f"{'='*70}")
    print(f"\nSystem: {args.system.upper()}")
    print(f"Corpus: {len(corpus)} articles")
    print(f"Queries: {len(queries)} questions")
    print(f"Embedding: {embedding_model}")
    print(f"LLM: {args.llm_model}")
    print(f"Top-K: {args.top_k}")
    if args.sample_queries:
        print(f"Sample Size: {args.sample_queries} queries")
    
    # Initialize system
    if args.system == 'aurag':
        system = AuRAGSystem(system_name='aurag')
        db_path = output_dir / f"vector_db_{args.system}_{args.year}_{args.embedding}"
        
        system.setup(
            corpus=corpus,
            embedding_model=embedding_model,
            llm_model_path=args.llm_model,
            db_path=db_path,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            child_chunk_size=300,
            child_chunk_overlap=90,
            context_budget=12000,
            n_ctx=16384,
            n_gpu_layers=-1
        )
    else:
        raise ValueError(f"Unknown system: {args.system}")
    
    # Run evaluation
    aggregate_metrics, per_query_results = evaluate_rag_system(
        system=system,
        queries=queries,
        system_name=args.system,
        sample_size=args.sample_queries
    )
    
    # Save results
    output_filename = f"evaluation_results_end2end_{args.system}_k{args.top_k}.json"
    if args.sample_queries:
        output_filename = output_filename.replace('.json', f'_sample{args.sample_queries}.json')
    
    output_path = output_dir / output_filename
    
    save_results(
        output_path=output_path,
        system_name=args.system,
        aggregate_metrics=aggregate_metrics,
        per_query_results=per_query_results,
        metadata=system.get_metadata()
    )
    
    print(f"\n✅ Evaluation complete!")
    print(f"📄 Results: {output_path}")


if __name__ == "__main__":
    main()
