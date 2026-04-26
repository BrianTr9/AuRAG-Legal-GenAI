"""System registry for standardized end-to-end evaluation."""

from __future__ import annotations

from typing import Callable, Dict, List

from .AuRAGSystem import AuRAGSystem
from .base import RAGSystem


def _build_aurag() -> RAGSystem:
    return AuRAGSystem(system_name="aurag")


_SYSTEM_BUILDERS: Dict[str, Callable[[], RAGSystem]] = {
    "aurag": _build_aurag,
}


def available_systems() -> List[str]:
    return sorted(_SYSTEM_BUILDERS.keys())


def create_system(system_name: str) -> RAGSystem:
    key = (system_name or "").strip().lower()
    if key not in _SYSTEM_BUILDERS:
        raise ValueError(f"Unknown system '{system_name}'. Available: {', '.join(available_systems())}")
    return _SYSTEM_BUILDERS[key]()
