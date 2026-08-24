from pathlib import Path
import shutil
import time
from modules.document_chunker import chunk_pdf
from modules.docling_parser import parse_document
from modules.markdown_combiner import combine_markdowns
from modules.image_analyzer import analyze_images
from modules.markdown_merge import merge_markdown
from modules.reporting import ProcessingReport
from modules.ui import ConsoleUI


def process_docling_pipeline(
    converted_path,
    output_dir,
    temp_assets_dir,
    temp_chunks_dir,
    document_index,
    ui=None,
):

    pipeline_start = time.perf_counter()
    analysis_time = 0.0
    chunking_time = 0.0
    parsing_time = 0.0
    image_analysis_time = 0.0
    merge_time = 0.0
    if ui is None:
        ui = ConsoleUI()
    total_images = 0

    report = ProcessingReport()

    report.input_file = converted_path.name
    # ----------------------------------------
    # Document Parsing
    # ----------------------------------------
    doc_prefix = f"doc_{document_index:04d}"
    if converted_path.suffix.lower() == ".pdf":
        ui.subheader("PDF Chunking")
        chunk_start = time.perf_counter()
        chunk_paths = chunk_pdf(
            pdf_path=converted_path,
            output_folder=temp_chunks_dir,
            chunk_size=25,
            prefix=f"{doc_prefix}_chunk",
        )
        chunk_end = time.perf_counter()
        chunking_time = chunk_end - chunk_start
        ui.info(
            f"Chunking Time : {chunking_time:.2f} seconds"
        )

        ui.success(f"Created {len(chunk_paths)} chunks.")

        # Parse each chunk using Docling
        ui.subheader("Docling Processing")
        parsing_start = time.perf_counter()
        progress = ui.progress(0)
        total_chunks = len(chunk_paths)

        for index, chunk_path in enumerate(chunk_paths):

            chunk_name = Path(chunk_path).stem
            chunk_stage_start = time.perf_counter()
            result = parse_document(
                input_path=chunk_path,
                output_dir=output_dir,
                assets_dir=temp_assets_dir,
                page_name=chunk_name,
            )
            chunk_stage_end = time.perf_counter()

            total_images += result["image_count"]
            ui.write(
                f"{chunk_name} | "
                f"{chunk_stage_end - chunk_stage_start:.2f} sec"
            )
            progress.progress((index + 1) / total_chunks)

        parsing_end = time.perf_counter()
        parsing_time = parsing_end - parsing_start
        ui.info(
            f"Docling Processing Time : {parsing_time:.2f} seconds"
        )
        progress.empty()

        # Combine all chunk markdowns
        raw_markdown_name = f"document_{document_index:04d}_raw.md"
        markdown_file = combine_markdowns(
            output_dir=output_dir,
            output_name=raw_markdown_name,
            pattern=f"{doc_prefix}_chunk_*.md",
        )

        # Clean up intermediate chunk markdown files for this document
        for chunk_f in output_dir.glob(f"{doc_prefix}_chunk_*.md"):
            try:
                chunk_f.unlink()
            except Exception:
                pass

    else:
        ui.subheader("Docling Processing")
        parsing_start = time.perf_counter()

        result = parse_document(
            input_path=converted_path,
            output_dir=output_dir,
            assets_dir=temp_assets_dir,
            page_name=f"doc_{document_index:04d}",
        )

        raw_markdown_name = f"document_{document_index:04d}_raw.md"
        markdown_file = output_dir / raw_markdown_name
        if result["markdown"].exists() and result["markdown"] != markdown_file:
            result["markdown"].replace(markdown_file)

        total_images = result["image_count"]

        parsing_end = time.perf_counter()
        parsing_time = parsing_end - parsing_start
        ui.info(
            f"Docling Processing Time : {parsing_time:.2f} seconds"
        )

    # ----------------------------------------
    # Image Analysis (Unified)
    # ----------------------------------------
    ui.subheader("Image Analysis")
    ia_start = time.perf_counter()

    ia_result = analyze_images(
        temp_assets_dir
    )

    image_analysis = ia_result["results"]

    report.images_analyzed = ia_result["generated"]
    report.images_cached = ia_result["cached"]
    report.images_skipped = ia_result["skipped"]
    report.images_failed = ia_result["failed"]

    ia_end = time.perf_counter()
    image_analysis_time = ia_end - ia_start

    ui.info(
        f"Image Analysis Time : {image_analysis_time:.2f} seconds"
    )

    # ----------------------------------------
    # Final Markdown
    # ----------------------------------------

    final_output = output_dir / f"document_{document_index:04d}.md"
    ui.subheader("Generating Final Markdown")
    merge_start = time.perf_counter()
    final_markdown = merge_markdown(
        markdown_path=markdown_file,
        assets_dir=temp_assets_dir,
        image_analysis=image_analysis,
        output_path=final_output,
    )

    # Clean up intermediate raw markdown file
    if markdown_file != final_markdown and markdown_file.exists():
        try:
            markdown_file.unlink()
        except Exception:
            pass

    # Copy assets to persistent output folder so markdown image links work
    assets_dest = output_dir / "assets"
    if temp_assets_dir.exists():
        assets_dest.mkdir(parents=True, exist_ok=True)
        for img_file in temp_assets_dir.glob("*.*"):
            if img_file.is_file():
                shutil.copy2(img_file, assets_dest / img_file.name)

    merge_end = time.perf_counter()
    merge_time = merge_end - merge_start
    ui.info(
        f"Markdown Merge Time : {merge_time:.2f} seconds"
    )
    pipeline_end = time.perf_counter()
    report.output_file = final_markdown.name

    report.images_extracted = total_images

    report.analysis_time = analysis_time
    report.chunking_time = chunking_time
    report.parsing_time = parsing_time
    report.image_analysis_time = image_analysis_time
    report.merge_time = merge_time
    report.total_time = pipeline_end - pipeline_start

    return {
        "markdown": final_markdown,
        "image_count": total_images,
        "timings": {
            "analysis": analysis_time,
            "chunking": chunking_time,
            "parsing": parsing_time,
            "image_analysis": image_analysis_time,
            "merge": merge_time,
            "total": pipeline_end - pipeline_start,
        },
        "report": report,
        "analysis_times": ia_result["analysis_times"],
    }