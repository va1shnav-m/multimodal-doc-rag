from typing import List, Dict

MAX_CHUNK_SIZE = 20

# Docling is memory-heavy (images, tables, OCR).
# Smaller chunks prevent bad_alloc crashes.
MAX_DOCLING_CHUNK_SIZE = 5


def create_execution_plan(analysis: List[Dict]):
    """
    Creates parser-aware execution chunks.

    Uses smaller chunks for Docling (image/table heavy)
    to prevent memory exhaustion (bad_alloc).

    Returns:
    [
        {
            "parser": "pymupdf",
            "start_page": 1,
            "end_page": 10,
            "page_count": 10
        },
        ...
    ]
    """
    
    execution_plan = []

    if not analysis:
        return execution_plan

    current_parser = analysis[0]["parser"]

    start_page = analysis[0]["page"]
    current_count = 1

    for i in range(1, len(analysis)):

        parser = analysis[i]["parser"]

        page_number = analysis[i]["page"]

        # Pick the right chunk limit based on parser
        if current_parser == "docling":
            chunk_limit = MAX_DOCLING_CHUNK_SIZE
        else:
            chunk_limit = MAX_CHUNK_SIZE

        # same parser and within chunk size
        if (
            parser == current_parser
            and current_count < chunk_limit
        ):
            current_count += 1

        else:

            execution_plan.append(
                {
                    "parser": current_parser,
                    "start_page": start_page,
                    "end_page": analysis[i - 1]["page"],
                    "page_count": current_count,
                }
            )

            current_parser = parser
            start_page = page_number
            current_count = 1

    execution_plan.append(
        {
            "parser": current_parser,
            "start_page": start_page,
            "end_page": analysis[-1]["page"],
            "page_count": current_count,
        }
    )

    return execution_plan


import fitz
from pathlib import Path


def create_temp_pdf(
    source_pdf,
    start_page,
    end_page,
    output_dir,
    chunk_name,
):
    """
    Creates a temporary PDF containing only the requested pages.

    Parameters
    ----------
    source_pdf : str | Path
    start_page : int      # 1-based
    end_page : int        # 1-based
    output_dir : str | Path
    chunk_name : str

    Returns
    -------
    Path
        Path to the generated temporary PDF.
    """

    source_pdf = Path(source_pdf)
    output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    output_pdf = output_dir / f"{chunk_name}.pdf"

    src = fitz.open(source_pdf)
    dst = fitz.open()

    # PyMuPDF uses 0-based indexing
    dst.insert_pdf(
        src,
        from_page=start_page - 1,
        to_page=end_page - 1,
    )

    dst.save(output_pdf)

    dst.close()
    src.close()

    return output_pdf