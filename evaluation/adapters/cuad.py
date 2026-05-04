"""CUAD dataset adapter."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Corpus, DatasetAdapter, Query, resolve_path


def _normalize_cuad_doc_id(relative_path: Path) -> str:
    """Return the canonical CUAD corpus-unit ID.

    We keep CUAD as whole-document corpus units, not section-level units.
    The file path namespace is preserved so the benchmark snippets can be
    traced back to the originating document unambiguously.
    """
    return f"cuad/{relative_path.as_posix()}"


def _normalize_snippet_file_path(file_path: str) -> str:
    """Normalize CUAD snippet file path with Unicode NFC normalization.
    
    Ensures that file paths with combining characters (e.g., LECLANCHÉ) are
    normalized to NFC form, matching the canonical form in corpus file IDs.
    """
    path = file_path.strip().replace("\\", "/")
    # Apply Unicode NFC normalization to handle combining characters
    path = unicodedata.normalize('NFC', path)
    if path.startswith("cuad/"):
        return path
    if path.startswith("./cuad/"):
        return path[2:]
    if path.startswith("/cuad/"):
        return path[1:]
    return f"cuad/{path.lstrip('./')}"


class CuadAdapter(DatasetAdapter):
    name = "cuad"

    def load_corpus(self, workspace_root: Path, corpus_path: Optional[str] = None) -> Corpus:
        path = resolve_path(workspace_root, corpus_path, "benchmark/LegalBench-RAG/corpus/cuad")
        if path.is_file():
            raise ValueError("CUAD corpus_path must point to a directory of contract text files")

        corpus: Corpus = {}
        for text_path in sorted(path.rglob("*.txt")):
            if not text_path.is_file():
                continue
            relative_path = text_path.relative_to(path)
            # Apply Unicode NFC normalization to doc_id for consistency with benchmark file paths
            relative_path_str = unicodedata.normalize('NFC', relative_path.as_posix())
            doc_id = f"cuad/{relative_path_str}"
            text = text_path.read_text(encoding="utf-8", errors="ignore")
            corpus[doc_id] = {
                "caption": text_path.stem,
                "text": text,
                "metadata": {
                    "source_path": str(text_path),
                    "relative_path": relative_path.as_posix(),
                    "unit_type": "document",
                    "dataset": self.name,
                },
            }
        return corpus

    def load_queries(self, workspace_root: Path, queries_path: Optional[str] = None) -> List[Query]:
        path = resolve_path(workspace_root, queries_path, "benchmark/LegalBench-RAG/benchmarks/cuad.json")
        data = json.loads(path.read_text(encoding="utf-8"))

        tests = data.get("tests") if isinstance(data, dict) else data
        if not isinstance(tests, list):
            raise ValueError("CUAD benchmark must contain a 'tests' array")

        queries: List[Query] = []
        for idx, item in enumerate(tests, start=1):
            if not isinstance(item, dict):
                continue

            snippets = item.get("snippets") or []
            normalized_snippets: List[Dict[str, Any]] = []
            relevant_articles: List[str] = []
            evidence_answers: List[str] = []

            for snippet in snippets:
                if not isinstance(snippet, dict):
                    continue
                file_path = _normalize_snippet_file_path(str(snippet.get("file_path") or ""))
                span = snippet.get("span")
                answer = snippet.get("answer")

                if file_path:
                    relevant_articles.append(file_path)

                normalized_snippets.append(
                    {
                        "file_path": file_path,
                        "span": span,
                        "answer": answer,
                    }
                )
                if isinstance(answer, str) and answer.strip():
                    evidence_answers.append(answer.strip())

            query_text = str(item.get("query") or "").strip()
            query_id = str(item.get("id") or f"cuad_{idx:06d}")

            queries.append(
                {
                    "query_id": query_id,
                    "question": query_text,
                    "ground_truth_answer": None,
                    "relevant_articles": sorted(set(relevant_articles)),
                    "metadata": {
                        "ground_truth_snippets": normalized_snippets,
                        "evidence_answers": evidence_answers,
                        "benchmark": "LegalBench-RAG/CUAD",
                    },
                }
            )
        return queries