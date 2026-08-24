import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional

from modules.pptx_parser import parse_pptx
from modules.image_analyzer import analyze_images
from modules.markdown_merge import merge_markdown
from modules.reporting import ProcessingReport
from modules.ui import ConsoleUI


def process_pptx_pipeline(
    converted_path: str | Path,
    output_dir: str | Path,
    temp_assets_dir: str | Path,
    document_index: int = 1,
    ui: Optional[ConsoleUI] = None,
    skip_image_analysis: bool = False,
) -> Dict[str, Any]:
    """
    Process a PowerPoint presentation (.pptx) into structured Markdown.

    Parameters
    ----------
    converted_path : str | Path
        Path to the PPTX file.
    output_dir : str | Path
        Directory for the final markdown file.
    temp_assets_dir : str | Path
        Temporary directory for extracted slide images.
    document_index : int
        Current document index in batch.
    ui : ConsoleUI, optional
    skip_image_analysis : bool

    Returns
    -------
    dict
        Pipeline results including markdown path, timings, and report.
    """
    converted_path = Path(converted_path)
    output_dir = Path(output_dir)
    temp_assets_dir = Path(temp_assets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_assets_dir.mkdir(parents=True, exist_ok=True)

    if ui is None:
        ui = ConsoleUI()

    pipeline_start = time.perf_counter()
    report = ProcessingReport()
    report.input_file = converted_path.name

    # ----------------------------------------------------
    # 1. Parse PPTX
    # ----------------------------------------------------
    ui.subheader("PowerPoint Parsing")
    parse_start = time.perf_counter()

    parsed = parse_pptx(
        pptx_path=converted_path,
        assets_dir=temp_assets_dir,
        document_prefix=f"doc_{document_index:04d}",
    )

    parse_time = time.perf_counter() - parse_start
    total_images = parsed["image_count"]
    slide_count = parsed["slide_count"]
    raw_markdown = parsed["markdown"]

    report.total_pages = slide_count
    ui.info(f"Parsed {slide_count} slides with {total_images} images in {parse_time:.2f}s")

    raw_markdown_path = output_dir / f"document_{document_index:04d}_raw.md"
    raw_markdown_path.write_text(raw_markdown, encoding="utf-8")

    # ----------------------------------------------------
    # 2. Image Analysis
    # ----------------------------------------------------
    image_analysis_time = 0.0
    image_analysis = {}
    analysis_times = {}

    if total_images > 0 and not skip_image_analysis:
        ui.subheader("Image Analysis")
        ia_start = time.perf_counter()

        ia_result = analyze_images(temp_assets_dir)
        image_analysis = ia_result["results"]
        analysis_times = ia_result.get("timings", {})

        report.images_analyzed = ia_result["generated"]
        report.images_cached = ia_result["cached"]
        report.images_skipped = ia_result["skipped"]
        report.images_failed = ia_result["failed"]

        image_analysis_time = time.perf_counter() - ia_start
        ui.info(f"Analyzed {report.images_analyzed} images in {image_analysis_time:.2f}s")
    else:
        report.images_skipped = total_images

    # ----------------------------------------------------
    # 3. Markdown Merge
    # ----------------------------------------------------
    ui.subheader("Markdown Merge")
    merge_start = time.perf_counter()

    final_markdown_path = output_dir / f"document_{document_index:04d}.md"
    named_output_path = output_dir / f"{converted_path.stem}.md"

    merge_markdown(
        markdown_path=raw_markdown_path,
        assets_dir=temp_assets_dir,
        image_analysis=image_analysis,
        output_path=named_output_path,
    )
    shutil.copy2(named_output_path, final_markdown_path)

    # Clean up intermediate raw markdown
    if raw_markdown_path.exists():
        try:
            raw_markdown_path.unlink()
        except Exception:
            pass

    # Copy assets to persistent output folder
    assets_dest = output_dir / "assets"
    if temp_assets_dir.exists():
        assets_dest.mkdir(parents=True, exist_ok=True)
        for img_file in temp_assets_dir.glob("*.*"):
            if img_file.is_file():
                shutil.copy2(img_file, assets_dest / img_file.name)

    merge_time = time.perf_counter() - merge_start

    total_time = time.perf_counter() - pipeline_start

    ui.success(f"Generated Markdown: {named_output_path.name}")

    return {
        "markdown": named_output_path,
        "image_count": total_images,
        "slide_count": slide_count,
        "timings": {
            "total": total_time,
            "analysis": 0.0,
            "chunking": 0.0,
            "parsing": parse_time,
            "image_analysis": image_analysis_time,
            "merge": merge_time,
        },
        "report": report,
        "analysis_times": analysis_times,
    }
