from dataclasses import dataclass, field
from typing import List


@dataclass
class ProcessingReport:

    input_file: str = ""
    output_file: str = ""

    total_pages: int = 0

    pymupdf_pages: int = 0
    docling_pages: int = 0

    tables_found: int = 0
    images_extracted: int = 0

    # Unified image analysis metrics
    images_analyzed: int = 0
    images_cached: int = 0
    images_skipped: int = 0
    images_failed: int = 0

    analysis_time: float = 0.0
    chunking_time: float = 0.0
    parsing_time: float = 0.0
    image_analysis_time: float = 0.0
    merge_time: float = 0.0
    total_time: float = 0.0

    warnings: List[str] = field(default_factory=list)

    def add_warning(self, message):
        self.warnings.append(message)


def print_report(report):
    """
    Print a processing summary to the terminal.
    """

    print("")
    print("=" * 60)
    print("  Document Processing Summary")
    print("=" * 60)

    print(f"  Input File   : {report.input_file}")
    print(f"  Output File  : {report.output_file}")

    # ------------------------------------------
    # Parser Statistics
    # ------------------------------------------

    print("")
    print("  Parser Statistics")
    print("  " + "-" * 40)
    print(f"  Total Pages  : {report.total_pages}")
    print(f"  PyMuPDF      : {report.pymupdf_pages}")
    print(f"  Docling      : {report.docling_pages}")
    print(f"  Tables Found : {report.tables_found}")

    # ------------------------------------------
    # Image Analysis
    # ------------------------------------------

    print("")
    print("  Image Analysis")
    print("  " + "-" * 40)
    print(f"  Images Extracted : {report.images_extracted}")
    print(f"  Analyzed         : {report.images_analyzed}")
    print(f"  Cached           : {report.images_cached}")
    print(f"  Skipped          : {report.images_skipped}")
    print(f"  Failed           : {report.images_failed}")

    # ------------------------------------------
    # Performance
    # ------------------------------------------

    print("")
    print("  Performance")
    print("  " + "-" * 40)

    if report.analysis_time > 0:
        print(f"  PDF Analysis     : {report.analysis_time:.2f}s")

    if report.chunking_time > 0:
        print(f"  PDF Chunking     : {report.chunking_time:.2f}s")

    print(f"  Parsing          : {report.parsing_time:.2f}s")
    print(f"  Image Analysis   : {report.image_analysis_time:.2f}s")
    print(f"  Markdown Merge   : {report.merge_time:.2f}s")
    print(f"  Total            : {report.total_time:.2f}s")

    # ------------------------------------------
    # Warnings
    # ------------------------------------------

    if report.warnings:
        print("")
        print("  Warnings")
        print("  " + "-" * 40)
        for warning in report.warnings:
            print(f"  ⚠ {warning}")

    print("=" * 60)