import argparse
import shutil
import time
from pathlib import Path

from modules.ui import ConsoleUI
from modules.doc_converter import convert_doc
from modules.utils import clear_folder
from modules.markdown_combiner import combine_final_documents
from modules.reporting import print_report



DOC_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
SUPPORTED_EXTENSIONS = DOC_EXTENSIONS | AUDIO_EXTENSIONS


def main():

    parser = argparse.ArgumentParser(
        description="Document and Audio to Markdown Converter"
    )

    subparsers = parser.add_subparsers(dest="command")

    # -------------------------------------------
    # convert command
    # -------------------------------------------

    convert = subparsers.add_parser(
        "convert",
        help="Convert documents and audio files to markdown"
    )

    convert.add_argument(
        "inputs",
        nargs="+",
        help="One or more input folders or file paths"
    )

    convert.add_argument(
        "--out",
        required=True,
        help="Output folder"
    )

    convert.add_argument(
        "--pipeline",
        choices=["hybrid", "docling"],
        default="hybrid",
        help="Processing pipeline for documents (default: hybrid)"
    )

    convert.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium"],
        default="base",
        help="Whisper model size for audio transcription (default: base)"
    )

    convert.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip LLM summary for audio files"
    )

    convert.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip image analysis"
    )

    convert.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output"
    )

    convert.add_argument(
        "--rag",
        action="store_true",
        help="Ingest converted markdown documents into Qdrant RAG pipeline"
    )

    convert.add_argument(
        "--clear-index",
        action="store_true",
        help="Clear existing vector collection and BM25 index before ingestion"
    )

    convert.add_argument(
        "--collection-name",
        default="production_rag",
        help="Target Qdrant collection name (default: production_rag)"
    )

    convert.add_argument(
        "--rag-pipeline",
        choices=["hybrid", "fast", "adaptive"],
        default="hybrid",
        help="RAG retrieval strategy: hybrid (Dense+BM25+Rerank+LLMLingua) or fast (default: hybrid)"
    )

    convert.add_argument(
        "--llm",
        choices=["qwen", "openai"],
        default="qwen",
        help="LLM backend for generation (default: qwen)"
    )

    convert.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a question immediately after RAG ingestion"
    )

    convert.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive terminal chat session after conversion"
    )

    # -------------------------------------------
    # rag command
    # -------------------------------------------

    rag_parser = subparsers.add_parser(
        "rag",
        help="Query, chat, or manage the Qdrant RAG knowledge base"
    )

    rag_parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a question against the indexed documents"
    )

    rag_parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch an interactive terminal chat session"
    )

    rag_parser.add_argument(
        "--status",
        action="store_true",
        help="Show Qdrant database status and indexed document list"
    )

    rag_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all vectors, nodes, and BM25 index"
    )

    rag_parser.add_argument(
        "--ingest",
        nargs="+",
        help="Ingest one or more markdown files or directories directly into RAG"
    )

    rag_parser.add_argument(
        "--collection-name",
        default="production_rag",
        help="Target Qdrant collection name (default: production_rag)"
    )

    rag_parser.add_argument(
        "--pipeline",
        choices=["hybrid", "fast", "adaptive"],
        default="hybrid",
        help="RAG retrieval mode (default: hybrid)"
    )

    rag_parser.add_argument(
        "--llm",
        choices=["qwen", "openai"],
        default="qwen",
        help="LLM backend for generation (default: qwen)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "convert":
        run_convert(args)
    elif args.command == "rag":
        run_rag(args)


def run_convert(args):
    from pipelines.hybrid_pipeline import process_hybrid_pipeline
    from pipelines.docling_pipeline import process_docling_pipeline
    from pipelines.audio_pipeline import process_audio_pipeline
    from pipelines.pptx_pipeline import process_pptx_pipeline
    from benchmark.benchmark import Benchmark
    from benchmark.report_generator import generate_benchmark_report

    output_folder = Path(args.out)
    ui = ConsoleUI()

    # -------------------------------------------
    # Refresh Output Directory & Atomic Archival of Prior Runs
    # -------------------------------------------
    if output_folder.exists() and any(output_folder.iterdir()):
        try:
            archive_root = Path("storage/archive")
            archive_root.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_dir = archive_root / f"run_{timestamp}"

            # Atomic directory move — moves the entire folder in a single OS operation
            shutil.move(str(output_folder), str(archive_dir))
        except Exception as e:
            # If archiving fails, NEVER delete files unsafely. Preserve existing data.
            print(f"Warning: Could not atomically archive previous output ({e}). Existing files preserved.")

    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "assets").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------
    # Collect documents and audio files
    # -------------------------------------------

    supported_extensions = SUPPORTED_EXTENSIONS
    documents = []
    seen_paths = set()

    for path_str in args.inputs:
        input_path = Path(path_str)

        if not input_path.exists():
            print(f"Warning: Input path not found, skipping: {input_path}")
            continue

        if input_path.is_file():
            if input_path.suffix.lower() not in supported_extensions:
                print(f"Warning: Unsupported file type, skipping: {input_path}")
                continue
            resolved_path = input_path.resolve()
            if resolved_path not in seen_paths:
                documents.append(input_path)
                seen_paths.add(resolved_path)
        else:
            files_in_folder = sorted([
                f for f in input_path.rglob("*")
                if f.suffix.lower() in supported_extensions
            ])
            for f in files_in_folder:
                resolved_path = f.resolve()
                if resolved_path not in seen_paths:
                    documents.append(f)
                    seen_paths.add(resolved_path)

    if not documents:
        print("Error: No supported documents found to process.")
        exit(1)

    # -------------------------------------------
    # Print header
    # -------------------------------------------

    print("")
    print("=" * 60)
    print("  Document to Markdown Converter")
    print("=" * 60)
    print(f"  Inputs   : {', '.join(args.inputs)}")
    print(f"  Output   : {output_folder}")
    print(f"  Pipeline : {args.pipeline}")
    print(f"  Documents: {len(documents)}")
    print("=" * 60)

    # -------------------------------------------
    # Temp directories
    # -------------------------------------------

    temp_assets_dir = Path("temp_assets")
    temp_chunks_dir = Path("temp_chunks")

    # -------------------------------------------
    # Process each document
    # -------------------------------------------

    batch_start = time.perf_counter()
    generated_markdowns = []

    for index, document in enumerate(documents, start=1):

        print("")
        print("=" * 60)
        print(f"  Document {index}/{len(documents)}")
        print(f"  {document.name}")
        print("=" * 60)

        # Create benchmark
        benchmark = Benchmark()
        benchmark.document_name = document.name
        benchmark.document_type = document.suffix.lower()
        benchmark.file_size_mb = (
            document.stat().st_size / (1024 * 1024)
        )

        # Clear temp folders
        clear_folder("temp_assets")
        clear_folder("temp_chunks")

        for chunk_file in output_folder.glob("chunk_*.md"):
            chunk_file.unlink()

        # Convert legacy .doc → .docx or .ppt → .pptx if needed
        converted_path = document

        if document.suffix.lower() in {".doc", ".ppt"}:
            converted_path = Path(
                convert_doc(document)
            )

        # Run pipeline
        try:

            # -------------------------------------------
            # Audio files pipeline
            # -------------------------------------------
            if document.suffix.lower() in AUDIO_EXTENSIONS:

                audio_result = process_audio_pipeline(
                    audio_path=document,
                    output_dir=output_folder,
                    ui=ui,
                    whisper_model=args.whisper_model,
                    generate_summary=not args.no_summary,
                )

                generated_markdowns.append(Path(audio_result["output_path"]))

                print(f"\n  Markdown  : {audio_result['output_path']}")
                print(f"  Duration  : {audio_result['duration_formatted']} ({audio_result['duration']:.1f}s)")
                print(f"  Language  : {audio_result['language']}")
                print(f"  Speed     : {audio_result['transcription_time']}s")
                continue

            if args.pipeline == "docling":

                result = process_docling_pipeline(
                    converted_path=converted_path,
                    output_dir=output_folder,
                    temp_assets_dir=temp_assets_dir,
                    temp_chunks_dir=temp_chunks_dir,
                    document_index=index,
                    ui=ui,
                )

            elif converted_path.suffix.lower() == ".pptx":

                result = process_pptx_pipeline(
                    converted_path=converted_path,
                    output_dir=output_folder,
                    temp_assets_dir=temp_assets_dir,
                    document_index=index,
                    ui=ui,
                    skip_image_analysis=args.no_analysis,
                )

            else:

                result = process_hybrid_pipeline(
                    converted_path=converted_path,
                    output_dir=output_folder,
                    temp_assets_dir=temp_assets_dir,
                    temp_chunks_dir=temp_chunks_dir,
                    document_index=index,
                    ui=ui,
                )

            final_markdown = result["markdown"]
            generated_markdowns.append(Path(final_markdown))

            total_images = result["image_count"]
            timings = result["timings"]
            report = result["report"]
            analysis_times = result["analysis_times"]

            # -------------------------------------------
            # Print processing report
            # -------------------------------------------

            print_report(report)

            # -------------------------------------------
            # Populate benchmark
            # -------------------------------------------

            benchmark.total_time = timings["total"]
            benchmark.images_detected = total_images

            benchmark.stage_times = {
                "analysis": timings["analysis"],
                "chunking": timings["chunking"],
                "parsing": timings["parsing"],
                "image_analysis": timings["image_analysis"],
                "markdown_merge": timings["merge"],
            }

            benchmark.analysis_times = analysis_times

            benchmark.images_analyzed = report.images_analyzed
            benchmark.images_cached = report.images_cached
            benchmark.images_skipped = report.images_skipped
            benchmark.images_failed = report.images_failed

            benchmark.images_processed = (
                benchmark.images_detected
                - benchmark.images_skipped
            )

            # -------------------------------------------
            # Save benchmark HTML report
            # -------------------------------------------

            benchmark_html = generate_benchmark_report(benchmark)

            benchmark_filename = (
                f"{document.stem}_benchmark.html"
            )

            benchmark_path = output_folder / benchmark_filename

            benchmark_path.write_text(
                benchmark_html,
                encoding="utf-8"
            )

            print(f"\n  Benchmark : {benchmark_path}")
            print(f"  Markdown  : {final_markdown}")

        except Exception as e:

            print(f"\n  Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------
    # Combine all documents
    # -------------------------------------------

    batch_time = time.perf_counter() - batch_start

    if len(documents) > 0:

        final_batch = combine_final_documents(output_folder)

        print("")
        print("=" * 60)
        print("  Batch Complete")
        print("=" * 60)
        print(f"  Documents Processed : {len(documents)}")
        print(f"  Total Time          : {batch_time:.2f} sec")
        print(f"  Combined Output     : {final_batch}")
        print("=" * 60)

        # Clean up temporary working folders after batch finishes
        clear_folder("temp_assets")
        clear_folder("temp_chunks")

    # -------------------------------------------
    # Optional RAG Pipeline Ingestion & Query
    # -------------------------------------------

    if getattr(args, "rag", False) and generated_markdowns:
        from modules.rag_bridge import (
            ingest_markdown_outputs,
            query_rag,
            start_terminal_chat,
            print_query_result,
        )

        ingest_markdown_outputs(
            markdown_paths=generated_markdowns,
            collection_name=args.collection_name,
            clear_index=args.clear_index,
        )

        if getattr(args, "query", None):
            query_res = query_rag(
                question=args.query,
                collection_name=args.collection_name,
                pipeline_type=args.rag_pipeline,
                llm_choice=args.llm,
            )
            print_query_result(query_res)

        if getattr(args, "chat", False):
            start_terminal_chat(
                collection_name=args.collection_name,
                pipeline_type=args.rag_pipeline,
                llm_choice=args.llm,
            )


def run_rag(args):
    """Handler for the standalone 'rag' subcommand."""
    from modules.rag_bridge import (
        ingest_markdown_outputs,
        query_rag,
        start_terminal_chat,
        print_query_result,
        get_database_status,
        clear_database,
    )

    if args.status:
        status = get_database_status(collection_name=args.collection_name)
        print("\n" + "=" * 60)
        print("  QDRANT VECTOR DATABASE STATUS")
        print("=" * 60)
        for k, v in status.items():
            if isinstance(v, list):
                print(f"  {k:20s}: {len(v)} files")
                for item in v:
                    print(f"     - {item}")
            else:
                print(f"  {k:20s}: {v}")
        print("=" * 60 + "\n")
        return

    if args.clear:
        clear_database(collection_name=args.collection_name)
        return

    if args.ingest:
        md_files = []
        for inp in args.ingest:
            p = Path(inp)
            if not p.exists():
                print(f"  Warning: Path not found: {p}")
                continue
            if p.is_file() and p.suffix.lower() == ".md":
                md_files.append(p)
            elif p.is_dir():
                md_files.extend(sorted(list(p.rglob("*.md"))))
        if not md_files:
            print("  Error: No .md files found to ingest.")
            return
        ingest_markdown_outputs(
            markdown_paths=md_files,
            collection_name=args.collection_name,
        )

    if args.query:
        query_res = query_rag(
            question=args.query,
            collection_name=args.collection_name,
            pipeline_type=args.pipeline,
            llm_choice=args.llm,
        )
        print_query_result(query_res)
        return

    if args.chat or not (args.status or args.clear or args.ingest or args.query):
        start_terminal_chat(
            collection_name=args.collection_name,
            pipeline_type=args.pipeline,
            llm_choice=args.llm,
        )


if __name__ == "__main__":
    main()