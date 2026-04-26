"""Housing QA dataset adapter."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Corpus, DatasetAdapter, Query, resolve_path


class HousingQAAdapter(DatasetAdapter):
    name = "housing_qa"

    def load_corpus(self, workspace_root: Path, corpus_path: Optional[str] = None) -> Corpus:
        path = resolve_path(workspace_root, corpus_path, "benchmark/housing_qa/data/statutes.tsv")

        max_int = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_int)
                break
            except OverflowError:
                max_int = int(max_int / 10)

        corpus: Corpus = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                idx = (row.get("idx") or "").strip()
                if not idx:
                    continue
                citation = (row.get("citation") or idx).strip()
                text = row.get("text") or ""
                corpus[idx] = {
                    "caption": citation,
                    "text": text,
                }
        return corpus

    def load_queries(self, workspace_root: Path, queries_path: Optional[str] = None) -> List[Query]:
        path = resolve_path(workspace_root, queries_path, "benchmark/housing_qa/data/questions.json")
        data = json.loads(path.read_text(encoding="utf-8"))

        queries: List[Query] = []
        for i, item in enumerate(data, start=1):
            statutes = item.get("statutes") or []
            relevant: List[str] = []
            for s in statutes:
                if not isinstance(s, dict):
                    continue
                idx = s.get("statute_idx")
                if idx is not None:
                    relevant.append(str(idx))

            qid = item.get("id")
            if qid is None:
                qid = f"housing_{i:06d}"

            queries.append(
                {
                    "query_id": str(qid),
                    "question": str(item.get("question") or ""),
                    "ground_truth_answer": str(item.get("answer") or "").strip().upper() or None,
                    "relevant_articles": sorted(set(relevant)),
                    "metadata": {
                        "state": item.get("state"),
                    },
                }
            )
        return queries
