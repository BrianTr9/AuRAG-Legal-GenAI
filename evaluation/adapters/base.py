"""Common dataset adapter interfaces for evaluation scripts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

Corpus = Dict[str, Dict[str, str]]
Query = Dict[str, Any]


class DatasetAdapter(ABC):
    """Abstract adapter that maps a dataset into the evaluation schema."""

    name: str

    @abstractmethod
    def load_corpus(self, workspace_root: Path, corpus_path: Optional[str] = None) -> Corpus:
        """Load corpus as {doc_id: {'caption': str, 'text': str}}."""

    @abstractmethod
    def load_queries(self, workspace_root: Path, queries_path: Optional[str] = None) -> List[Query]:
        """Load queries in the canonical format expected by evaluation scripts.

        Required fields per query:
        - query_id: str
        - question: str
        - relevant_articles: list[str]

        Optional fields:
        - ground_truth_answer: str | None
        - metadata: dict
        """

    def metadata(self) -> Dict[str, Any]:
        """Return optional adapter metadata for reporting."""
        return {"dataset": self.name}


def resolve_path(workspace_root: Path, provided: Optional[str], default_rel: str) -> Path:
    """Resolve a possibly-relative path against workspace root."""
    raw = provided if provided else default_rel
    p = Path(raw)
    if p.is_absolute():
        return p
    return workspace_root / p
