#!/usr/bin/env python3
"""
Template 2b: Generation Evaluation with Ideal Retrieval (GT)
=========================================================

Evaluate generation quality and citation behavior when the LLM is given
an ideal retrieval list (ground-truth articles). This isolates Layer 2.

Compares:
- prompt: Zero-shot-CoT + citation + JSON (no grammar constraints)
- rdg: RDG (grammar-constrained JSON) with the same prompt

Usage:
    # RDG (ideal retrieval)
    python3 evaluation/evaluate_generation.py \
        --corpus benchmark/COLIEE/civil.xml \
        --queries benchmark/COLIEE/simple/simple_R06_jp.xml \
        --mode rdg

    # Prompt baseline (ideal retrieval)
    python3 evaluation/evaluate_generation.py \
        --corpus benchmark/COLIEE/civil.xml \
        --queries benchmark/COLIEE/simple/simple_R06_jp.xml \
        --mode prompt

    # Fast smoke test (10 queries)
    python3 evaluation/evaluate_generation.py --mode rdg --sample-queries 10
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from statistics import mean, stdev

from langchain_core.documents import Document

# Add evaluation/ and src/ to path for local imports
import sys
EVAL_PATH = Path(__file__).parent
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(EVAL_PATH))
sys.path.insert(0, str(SRC_PATH))

# Local imports
from evaluate_citation import (
    load_coliee_corpus,
    load_coliee_queries,
    normalize_yn_answer,
    calculate_citation_metrics,
)

from RDG import (
    get_rdg_pipeline,
    Citation,
    StructuredReasoning,
    _parse_json_output,
    _parse_reasoning,
)


def build_documents_from_gt(corpus: Dict[str, Dict[str, str]], gt_articles: List[str]) -> List[Document]:
    docs = []
    for article_id in gt_articles:
        item = corpus.get(article_id)
        if not item:
            continue
        docs.append(Document(
            page_content=item['text'],
            metadata={
                'contract_id': article_id,
                'section_num': article_id,
                'section_title': 'Article',
            }
        ))
    return docs


def extract_contract_id(ref: str) -> str:
    """
    Robustly extract the contract_id (Article Number) from various citation formats.
    Handles:
    - Full Ref: "123_Section_1_Title" -> "123"
    - Short Ref: "123_sec_1" -> "123"
    - Raw ID: "123" -> "123"
    """
    if "_Section_" in ref:
        return ref.split("_Section_")[0]
    if "_sec_" in ref:
        return ref.split("_sec_")[0]
    return ref.strip()


def extract_article_ids_from_citations(citations: List[Citation]) -> List[str]:
    ids = []
    for c in citations:
        # Use the robust extraction on the raw reference_id or contract_id
        if c.contract_id and c.contract_id != "Unknown":
             ids.append(extract_contract_id(c.contract_id))
        elif c.reference_id:
             ids.append(extract_contract_id(c.reference_id))
    return sorted(set(ids))


def prompt_only_generate(rdg, question: str, documents: List[Document], max_tokens: int = 1024, temperature: float = 0.0):
    # FAIRNESS: Use the exact same context construction as RDG (Full Headers)
    # This ensures both models see the same information (including Section Titles).
    valid_refs_full = rdg.extractor.extract_references(documents)

    context_parts = []
    for doc in documents:
        # Reconstruct Full Ref (same as RDG.py)
        c_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
        s_num = doc.metadata.get('section_num', 'N/A')
        s_title = (doc.metadata.get('section_title') or 'Unknown').replace(' ', '_').strip()
        full_ref = f"{c_id}_Section_{s_num}_{s_title}"
        context_parts.append(f"[{full_ref}]\n{doc.page_content.strip()}")

    # Build prompt using Full Refs
    # (The prompt text asks for Short IDs, but we will accept Full IDs in post-processing)
    prompt = rdg._build_prompt(
        question,
        valid_refs_full,
        "\n\n---\n\n".join(context_parts),
        answer_mode="yn",
    )

    output = rdg.llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=1.12,
        stop=["<|eot_id|>", "<|end_of_text|>"]
    )

    raw_text = output['choices'][0]['text'].strip()
    output_dict = _parse_json_output(raw_text)
    
    if not output_dict:
        # Fallback for broken JSON
        return {
            'answer': raw_text, 
            'reasoning': StructuredReasoning([]),
            'citations': []
        }

    reasoning = _parse_reasoning(output_dict.get('reasoning', ''))

    # COLLECT CITATIONS
    # We accept tokens that loosely match known document IDs
    collected_refs = set()
    
    # Helper to clean strings
    def looks_like_ref(text: str) -> bool:
        return isinstance(text, str) and (len(text) > 0)

    # 1. From Reasoning
    if isinstance(reasoning, StructuredReasoning) and reasoning.steps:
        for step in reasoning.steps:
            for cit in (step.citations or []):
                if looks_like_ref(cit):
                    collected_refs.add(cit)
                    
    # 2. From Citations field
    if 'citations' in output_dict and isinstance(output_dict['citations'], list):
         for cit in output_dict['citations']:
             if looks_like_ref(cit):
                 collected_refs.add(cit)

    # Convert strings to Citation objects
    # Note: We create Citations with the raw string as reference_id
    # The 'extract_article_ids_from_citations' function will handle the parsing logic.
    validated_citations = [Citation(ref, "Unknown", "Unknown", ref) for ref in collected_refs]

    return {
        'answer': output_dict.get('answer', ''),
        'reasoning': reasoning,
        'citations': validated_citations
    }


def evaluate_generation(
    mode: str,
    corpus: Dict[str, Dict[str, str]],
    queries: List[Dict[str, Any]],
    llm_model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    max_tokens: int,
    sample_size: int = None,
):
    if sample_size:
        queries = queries[:sample_size]
        print(f"\n📊 Testing on {len(queries)} sample queries")
    else:
        print(f"\n📊 Evaluating on {len(queries)} queries")

    # Validate Layer 2 setup: retrieved should equal relevant_articles (GT context)
    print(f"✓ Layer 2: Perfect retrieval (GT context isolation)")

    rdg = get_rdg_pipeline(model_path=llm_model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)

    per_query_results = []
    hallucination_rates = []
    citation_precisions = []
    citation_recalls = []
    answer_correctness = []
    generation_times = []

    for i, query_data in enumerate(queries, 1):
        query_id = query_data['query_id']
        question = query_data['question']
        ground_truth = query_data['ground_truth_answer']
        relevant_articles = query_data['relevant_articles']

        print(f"\n[{i}/{len(queries)}] {query_id}")

        # Build documents from GT (Layer 2: Perfect retrieval)
        retrieved = sorted(set(relevant_articles))
        assert len(retrieved) > 0, f"❌ Layer 2 BROKEN: Query {query_id} has no GT articles! Parser fix may have failed."
        docs = build_documents_from_gt(corpus, retrieved)

        t_start = time.time()
        if mode == "rdg":
            structured = rdg.generate(question=question, documents=docs, max_tokens=max_tokens, answer_mode="yn")
            answer = structured.answer
            citations = extract_article_ids_from_citations(structured.citations)
        else:
            baseline = prompt_only_generate(rdg, question, docs, max_tokens=max_tokens, temperature=0.0)
            answer = baseline['answer']
            citations = extract_article_ids_from_citations(baseline['citations'])

        generation_time = time.time() - t_start

        predicted_label = normalize_yn_answer(answer)
        gt_label = (ground_truth or 'N').strip().upper()
        is_correct = None if predicted_label is None else (1 if predicted_label == gt_label else 0)

        metrics = calculate_citation_metrics(citations, retrieved, relevant_articles)
        
        print(f"  Answer: {gt_label} → {predicted_label or '?'} {'✓' if is_correct == 1 else '✗' if is_correct == 0 else ''}")
        print(f"  Hallucination: {metrics['citation_hallucination_rate']*100:.1f}% | Precision: {metrics['citation_precision']*100:.1f}% | Recall: {metrics['citation_recall']*100:.1f}%")
        print(f"  Time: {generation_time:.2f}s")

        hallucination_rates.append(metrics['citation_hallucination_rate'])
        citation_precisions.append(metrics['citation_precision'])
        citation_recalls.append(metrics['citation_recall'])
        if is_correct is not None:
            answer_correctness.append(is_correct)
        generation_times.append(generation_time)

        # Store result (minimal necessary fields)
        result_entry = {
            'query_id': query_id,
            'ground_truth_answer': ground_truth,
            'predicted_answer': predicted_label,
            'answer_correct': is_correct,
            'gt_citations': relevant_articles,
            'llm_citations': citations,
            'hallucination_rate': metrics['citation_hallucination_rate'],
            'citation_precision': metrics['citation_precision'],
            'citation_recall': metrics['citation_recall'],
            'generation_time': generation_time
        }
        
        # Include raw answer only if normalization changed it
        if answer != predicted_label:
            result_entry['raw_answer'] = answer

        per_query_results.append(result_entry)

    aggregate_metrics = {
        'citation_hallucination_rate': {
            'mean': mean(hallucination_rates),
            'std': stdev(hallucination_rates) if len(hallucination_rates) > 1 else 0.0,
        },
        'citation_precision': {
            'mean': mean(citation_precisions),
            'std': stdev(citation_precisions) if len(citation_precisions) > 1 else 0.0,
        },
        'citation_recall': {
            'mean': mean(citation_recalls),
            'std': stdev(citation_recalls) if len(citation_recalls) > 1 else 0.0,
        },
        'answer_accuracy': {
            'mean': mean(answer_correctness) if answer_correctness else 0.0,
            'count_correct': sum(answer_correctness),
            'count_wrong': len(answer_correctness) - sum(answer_correctness),
            'total': len(answer_correctness)
        },
        'generation_time': {
            'mean': mean(generation_times) if generation_times else 0.0,
            'std': stdev(generation_times) if len(generation_times) > 1 else 0.0,
            'total': sum(generation_times)
        }
    }

    return aggregate_metrics, per_query_results


def main():
    parser = argparse.ArgumentParser(description="Generation Evaluation with Ideal Retrieval (GT)")
    parser.add_argument('--corpus', type=str, default='benchmark/COLIEE/civil.xml')
    parser.add_argument('--queries', type=str, default='benchmark/COLIEE/simple/simple_R06_jp.xml')
    parser.add_argument('--mode', type=str, default='rdg', choices=['rdg', 'prompt'])
    parser.add_argument('--llm-model', type=str, default='models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
    parser.add_argument('--n-ctx', type=int, default=16384)
    parser.add_argument('--n-gpu-layers', type=int, default=-1)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--sample-queries', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    random.seed(args.seed)

    corpus = load_coliee_corpus(Path(args.corpus))
    queries = load_coliee_queries(Path(args.queries))

    if args.sample_queries > 0:
        queries = random.sample(queries, min(args.sample_queries, len(queries)))

    aggregate_metrics, per_query_results = evaluate_generation(
        mode=args.mode,
        corpus=corpus,
        queries=queries,
        llm_model_path=args.llm_model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        max_tokens=args.max_tokens,
        sample_size=None
    )

    output = {
        'metadata': {
            'mode': args.mode,
            'llm_model': args.llm_model,
            'num_queries': len(per_query_results),
        },
        'aggregate_metrics': aggregate_metrics,
        'per_query_results': per_query_results
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / f"evaluation_results_layer2_generation_{args.mode}_gt.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Layer-2 evaluation complete: {output_path}")


if __name__ == '__main__':
    main()
