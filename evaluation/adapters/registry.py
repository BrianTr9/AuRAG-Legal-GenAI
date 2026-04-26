"""Dataset adapter registry for standardized evaluation loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Corpus, Query
from .coliee import ColieeAdapter
from .housing_qa import HousingQAAdapter

_ADAPTERS = {
    "coliee": ColieeAdapter,
    "housing_qa": HousingQAAdapter,
}


def available_datasets() -> List[str]:
    return sorted(_ADAPTERS.keys())


def load_dataset(
    dataset: str,
    workspace_root: Path,
    corpus_path: Optional[str] = None,
    queries_path: Optional[str] = None,
) -> Tuple[Corpus, List[Query], Dict[str, Any]]:
    key = (dataset or "coliee").strip().lower()
    if key not in _ADAPTERS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {', '.join(available_datasets())}")

    adapter = _ADAPTERS[key]()
    corpus = adapter.load_corpus(workspace_root=workspace_root, corpus_path=corpus_path)
    queries = adapter.load_queries(workspace_root=workspace_root, queries_path=queries_path)
    metadata = adapter.metadata()
    return corpus, queries, metadata
