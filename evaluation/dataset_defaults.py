"""Dataset-specific default paths for evaluators."""

from pathlib import Path
from typing import Tuple, Optional

DATASET_DEFAULTS = {
    "coliee": {
        "corpus": "benchmark/COLIEE/civil.xml",
        "queries": "benchmark/COLIEE/simple/simple_R06_jp.xml",
    },
    "housing_qa": {
        "corpus": "benchmark/housing_qa/data/statutes.tsv",
        "queries": "benchmark/housing_qa/data/questions.json",
    },
}


def get_dataset_defaults(dataset: str) -> Tuple[str, str]:
    """Get default corpus and queries paths for a dataset.
    
    Args:
        dataset: Dataset name (coliee, housing_qa, etc.)
        
    Returns:
        (corpus_path, queries_path)
    """
    if dataset not in DATASET_DEFAULTS:
        dataset = "coliee"  # Fallback to COLIEE
    
    config = DATASET_DEFAULTS[dataset]
    return config["corpus"], config["queries"]


def resolve_dataset_paths(
    dataset: str,
    workspace_root: Path,
    provided_corpus: Optional[str] = None,
    provided_queries: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve dataset corpus/queries paths intelligently.
    
    Logic:
    1. If user provides explicit path, try to use it
    2. If provided path doesn't exist, fall back to dataset default
    3. Return (corpus_path, queries_path)
    
    Args:
        dataset: Dataset name
        workspace_root: Project root directory
        provided_corpus: Optional user-provided corpus path
        provided_queries: Optional user-provided queries path
        
    Returns:
        (corpus_path, queries_path) - absolute or relative paths
    """
    default_corpus, default_queries = get_dataset_defaults(dataset)
    
    # Resolve corpus path
    corpus_path = provided_corpus or default_corpus
    corpus_full = Path(corpus_path) if Path(corpus_path).is_absolute() else workspace_root / corpus_path
    if not corpus_full.exists():
        corpus_path = default_corpus
    
    # Resolve queries path
    queries_path = provided_queries or default_queries
    queries_full = Path(queries_path) if Path(queries_path).is_absolute() else workspace_root / queries_path
    if not queries_full.exists():
        queries_path = default_queries
    
    return corpus_path, queries_path
