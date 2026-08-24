from dataclasses import dataclass, field
from typing import Dict, List
import time
from contextlib import contextmanager

@dataclass
class Benchmark:
    """
    Stores benchmark metrics for document processing.
    """

    # ----------------------------------------
    # Document information
    # ----------------------------------------

    document_name: str = ""
    document_type: str = ""
    file_size_mb: float = 0.0
    total_pages: int = 0

    # ----------------------------------------
    # Timing
    # ----------------------------------------

    total_time: float = 0.0

    stage_times: Dict[str, float] = field(
        default_factory=dict
    )

    # ----------------------------------------
    # Parsing
    # ----------------------------------------

    pymupdf_pages: int = 0
    docling_pages: int = 0

    # ----------------------------------------
    # Images
    # ----------------------------------------

    images_detected: int = 0
    images_processed: int = 0
    images_skipped: int = 0

    # ----------------------------------------
    # Image Analysis (Unified)
    # ----------------------------------------

    images_analyzed: int = 0
    images_cached: int = 0
    images_failed: int = 0

    analysis_times: Dict[str, float] = field(
        default_factory=dict
    )

    # ----------------------------------------
    # Tables
    # ----------------------------------------

    tables_detected: int = 0
    tables_processed: int = 0

    # ----------------------------------------
    # Errors
    # ----------------------------------------

    errors: List[str] = field(
        default_factory=list
    )

@contextmanager
def measure_stage(benchmark, stage_name):
    """
    Measure the execution time of a pipeline stage
    and store it in the Benchmark object.
    """

    start = time.perf_counter()

    try:
        yield

    finally:
        elapsed = time.perf_counter() - start

        benchmark.stage_times[stage_name] = elapsed    