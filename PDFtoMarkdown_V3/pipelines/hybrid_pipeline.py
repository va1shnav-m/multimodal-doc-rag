from pathlib import Path
import shutil
import time

from modules.adaptive_chunker import create_execution_plan
from modules.adaptive_chunker import create_temp_pdf
from modules.pdf_analyzer import analyze_pdf
from modules.docling_parser import parse_document
from modules.pymupdf_parser import parse_document as parse_document_pymupdf
from modules.markdown_combiner import combine_markdowns
from modules.image_analyzer import analyze_images
from modules.markdown_merge import merge_markdown
from modules.reporting import ProcessingReport
from modules.ui import ConsoleUI



def process_hybrid_pipeline(
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
    report = ProcessingReport()

    report.input_file = converted_path.name
    # ----------------------------
    # PDF Analysis 
    # ----------------------------
    if converted_path.suffix.lower() == ".pdf":

        ui.subheader("PDF Analysis")

        analysis_start = time.perf_counter()

        
        analysis = analyze_pdf(converted_path)
        # --------------------------------------------
        # Report Statistics
        # --------------------------------------------

        report.total_pages = len(analysis)

        report.docling_pages = sum(
            1 for page in analysis
            if page["parser"] == "docling"
        )

        report.pymupdf_pages = sum(
            1 for page in analysis
            if page["parser"] == "pymupdf"
        )

        report.tables_found = sum(
            page["table_count"]
            for page in analysis
        )
        execution_plan = create_execution_plan(analysis)

        

        doc_prefix = f"doc_{document_index:04d}"
        adaptive_chunk_paths = []

        for index, chunk in enumerate(execution_plan):

            chunk_name = f"{doc_prefix}_chunk_{index:04d}"

            temp_pdf = create_temp_pdf(
                source_pdf=converted_path,
                start_page=chunk["start_page"],
                end_page=chunk["end_page"],
                output_dir=temp_chunks_dir,
                chunk_name=chunk_name,
            )

            adaptive_chunk_paths.append(
                {
                    "pdf": temp_pdf,
                    "parser": chunk["parser"],
                }
            )

            ui.write(
                f"{chunk_name} | "
                f"{chunk['parser']} | "
                f"Pages {chunk['start_page']} - {chunk['end_page']} | "
                f"({chunk['page_count']} pages)"
            )

        analysis_end = time.perf_counter()
        analysis_time = analysis_end - analysis_start

        ui.info(
            f"PDF Analysis Time : {analysis_time:.2f} seconds"
        )

        ui.success(f"Created {len(adaptive_chunk_paths)} adaptive chunks.")

    # ----------------------------
    # Parsing Processing
    # ----------------------------

    ui.subheader("Parser Processing")

    parsing_start = time.perf_counter()

    total_images = 0

    progress = ui.progress(0)

    if converted_path.suffix.lower() == ".pdf":

        total_chunks = len(adaptive_chunk_paths)

        total_images = 0

        for index, chunk in enumerate(adaptive_chunk_paths):

            chunk_pdf_path = chunk["pdf"]
            parser = chunk["parser"]

            chunk_name = Path(chunk_pdf_path).stem

            chunk_start = time.perf_counter()

            if parser == "docling":

                result = parse_document(
                    input_path=chunk_pdf_path,
                    output_dir=output_dir,
                    assets_dir=temp_assets_dir,
                    page_name=chunk_name,
                )
                total_images += result["image_count"]

            else:

                result = parse_document_pymupdf(
                    input_path=chunk_pdf_path,
                    output_dir=output_dir,
                    assets_dir=temp_assets_dir,
                    page_name=chunk_name,
                )
                total_images += result["image_count"]

            chunk_end = time.perf_counter()

            ui.write(
                f"{chunk_name} | "
                f"{parser} | "
                f"{chunk_end - chunk_start:.2f} sec"
            )

            

            progress.progress(
                (index + 1) / total_chunks
            )

        parsing_end = time.perf_counter()
        parsing_time = parsing_end - parsing_start
        
        ui.info(
            f"Parsing Time : {parsing_time:.2f} seconds"
        )

        ui.success("Document parsing completed.")    
        raw_markdown_name = f"document_{document_index:04d}_raw.md"

        markdown_file = combine_markdowns(
            output_dir,
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

        result = parse_document(
            input_path=converted_path,
            output_dir=output_dir,
            assets_dir=temp_assets_dir,
            page_name=f"doc_{document_index:04d}",
        )

        markdown_file = result["markdown"]

        total_images = result["image_count"]

        parsing_end = time.perf_counter()
        parsing_time = parsing_end - parsing_start

        ui.info(
            f"Parsing Time : {parsing_time:.2f} seconds"
        )
        
    progress.empty()    

    # ----------------------------
    # Image Analysis (Unified)
    # ----------------------------

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

    # ----------------------------
    # Markdown Merge
    # ----------------------------

    ui.subheader("Generating Final Markdown")

    merge_start = time.perf_counter()

    batch_markdown = output_dir / f"document_{document_index:04d}.md"
    
    final_markdown = merge_markdown(
        markdown_path=markdown_file,
        assets_dir=temp_assets_dir,
        image_analysis=image_analysis,
        output_path=batch_markdown,
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

    analysis_times = ia_result["analysis_times"]

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
        "analysis_times": analysis_times,
    } 