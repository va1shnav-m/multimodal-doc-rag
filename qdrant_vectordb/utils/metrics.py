"""
Central metrics tracker for production RAG pipeline.
Supports CLI terminal logging and execution profiling.
"""
from typing import Dict, Any

_METRICS: Dict[str, Any] = {}


def record_metric(key: str, value: Any) -> None:
    """Record an execution metric."""
    _METRICS[key] = value


def get_metrics() -> Dict[str, Any]:
    """Return a copy of the current metrics dictionary."""
    return dict(_METRICS)


def clear_metrics() -> None:
    """Reset all recorded metrics."""
    _METRICS.clear()
