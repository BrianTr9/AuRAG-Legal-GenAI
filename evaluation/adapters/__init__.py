"""Dataset adapters for standardized evaluation pipelines."""

from .registry import available_datasets, load_dataset

__all__ = ["available_datasets", "load_dataset"]
