"""
RAG Bridge Module
Connects PDFtoMarkdown_V3 conversion pipelines directly with the qdrant_vectordb RAG pipeline.
100% Terminal & CLI-native execution without frontend dependencies.
"""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure qdrant_vectordb is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
QDRANT_ROOT = PROJECT_ROOT / "qdrant_vectordb"

if str(QDRANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QDRANT_ROOT))


_VECTOR_STORE_CACHE = {}


def get_vector_store(collection_name: Optional[str] = None):
    """Initialize or load VectorStore with optional collection name."""
    key = collection_name or "default"
    if key not in _VECTOR_STORE_CACHE:
        from retrieval.vector_store import VectorStore
        _VECTOR_STORE_CACHE[key] = VectorStore(collection_name=collection_name)
    return _VECTOR_STORE_CACHE[key]


def ingest_markdown_outputs(
    markdown_paths: List[Path | str],
    collection_name: Optional[str] = None,
    clear_index: bool = False,
) -> Dict[str, Any]:
    """
    Ingest a list of converted Markdown files directly into Qdrant & BM25.
    """
    from ingestion.ingestion_pipeline import IngestionPipeline
    from retrieval.vector_store import VectorStore
    from retrieval.node_store import NodeStore
    from utils.metrics import clear_metrics, get_metrics

    clear_metrics()
    pipeline = IngestionPipeline(collection_name=collection_name)

    if clear_index:
        print("\n  [RAG] Clearing existing knowledge base index...")
        pipeline.vector_store.clear()
        NodeStore.clear()

    valid_paths = [Path(p) for p in markdown_paths if Path(p).exists()]
    if not valid_paths:
        print("  [RAG Warning] No valid markdown files found to ingest.")
        return {}

    print(f"\n  [RAG] Ingesting {len(valid_paths)} Markdown document(s) into vector pipeline...")
    start_time = time.perf_counter()

    documents = []
    for p in valid_paths:
        doc = pipeline.ingest_markdown_file(p)
        documents.append(doc)

    result = pipeline.build_pipeline(documents)
    total_time = time.perf_counter() - start_time

    stats = pipeline.vector_store.get_collection_statistics()
    indexed_docs = pipeline.vector_store.list_documents()

    print("\n" + "=" * 60)
    print("  RAG INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Collection Name : {stats['collection_name']}")
    print(f"  Total Vectors   : {stats['vectors']}")
    print(f"  Documents Total : {len(indexed_docs)}")
    print(f"  Ingestion Time  : {total_time:.2f}s")
    if result.get("report_path"):
        print(f"  Inspection File : {result['report_path']}")
    print("=" * 60)

    return {
        "collection_name": stats["collection_name"],
        "vectors": stats["vectors"],
        "documents": indexed_docs,
        "report_path": result.get("report_path"),
        "total_time": round(total_time, 2),
        "metrics": get_metrics(),
    }


def query_rag(
    question: str,
    collection_name: Optional[str] = None,
    pipeline_type: str = "hybrid",
    llm_choice: str = "qwen",
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Execute a full RAG query against the indexed Qdrant knowledge base.
    """
    from retrieval.node_store import NodeStore
    from retrieval.retriever import ProductionRetriever
    from retrieval.hybrid_search import HybridSearch
    from retrieval.simple_search import SimpleSearch
    from retrieval.router import AdaptiveRouter
    from llm.qwen import QwenLLM
    from llm.openai import OpenAILLM
    from utils.metrics import clear_metrics, get_metrics

    clear_metrics()
    overall_start = time.perf_counter()

    # Load LLM early in case of direct mode
    if llm_choice.lower() == "openai":
        llm = OpenAILLM()
    else:
        llm = QwenLLM()

    # Fast direct check for greetings & chitchat
    from retrieval.route_selector import CHITCHAT_REGEX
    cleaned_q = question.strip()
    if len(cleaned_q) < 80 and CHITCHAT_REGEX.search(cleaned_q):
        answer = llm.generate(question=question, context="", history=history)
        overall_time = time.perf_counter() - overall_start
        return {
            "question": question,
            "answer": answer,
            "supporting_evidence": [],
            "documents_used": [],
            "context": "",
            "analytics": {"pipeline": "direct", "reason": "Chitchat / greeting detected (0ms retrieval)"},
            "metrics": get_metrics(),
            "total_time": round(overall_time, 3),
        }

    # Load vector store & index
    vstore = get_vector_store(collection_name=collection_name)
    nodes = NodeStore.load()

    if not nodes:
        # Fallback to direct generation if no documents are indexed
        answer = llm.generate(question=question, context="", history=history)
        overall_time = time.perf_counter() - overall_start
        return {
            "question": question,
            "answer": answer,
            "supporting_evidence": [],
            "documents_used": [],
            "context": "",
            "analytics": {"pipeline": "direct", "reason": "No documents indexed - answered with general knowledge"},
            "metrics": get_metrics(),
            "total_time": round(overall_time, 3),
        }

    index = vstore.load_index()
    storage_context = vstore.get_storage_context()

    retriever = ProductionRetriever(
        index=index,
        storage_context=storage_context,
        nodes=nodes,
    )

    # Route search
    if pipeline_type == "fast":
        search_engine = SimpleSearch(retriever)
        retrieval_result = search_engine.retrieve(question)
    elif pipeline_type == "hybrid":
        search_engine = HybridSearch(retriever)
        retrieval_result = search_engine.retrieve(question)
    else:  # adaptive (default)
        router = AdaptiveRouter(retriever)
        retrieval_result = router.retrieve(question)

    context = retrieval_result.get("context", "")
    supporting_evidence = retrieval_result.get("supporting_evidence", [])
    documents_used = retrieval_result.get("documents_used", [])

    # Generate answer
    answer = llm.generate(
        question=question,
        context=context,
        history=history,
    )

    overall_time = time.perf_counter() - overall_start

    return {
        "question": question,
        "answer": answer,
        "supporting_evidence": supporting_evidence,
        "documents_used": documents_used,
        "context": context,
        "analytics": retrieval_result.get("analytics", {}),
        "metrics": get_metrics(),
        "total_time": round(overall_time, 3),
    }


def print_query_result(result: Dict[str, Any]) -> None:
    """Pretty print a RAG answer with citations and evidence in the CLI."""
    print("\n" + "=" * 70)
    print(f"  QUESTION: {result.get('question', '')}")
    print("=" * 70)
    print("\n" + result.get("answer", "").strip() + "\n")
    print("-" * 70)

    # Documents used / citations
    docs_used = result.get("documents_used", [])
    if docs_used:
        print("  SOURCES & CITATIONS:")
        for doc in docs_used:
            pages = ", ".join(map(str, doc.get("pages", [])))
            print(f"   * {doc.get('filename')} (Pages: {pages} | Chunks: {doc.get('chunks', 1)})")
        print("")

    # Supporting Evidence
    evidence_list = result.get("supporting_evidence", [])
    if evidence_list:
        print("  SUPPORTING EVIDENCE:")
        for i, ev in enumerate(evidence_list[:3], start=1):
            score_str = f"Score: {ev.get('score', 0):.4f}" if "score" in ev else ""
            print(f"   [{i}] {ev.get('filename', '')} (Page {ev.get('page', '')}) {score_str}")
            snippet = ev.get("text", "").replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            print(f"       \"{snippet}\"")
        print("")

    # Metrics & Performance
    metrics = result.get("metrics", {})
    if metrics:
        print(f"  PERFORMANCE METRICS (Total: {result.get('total_time', 0):.2f}s):")
        metrics_summary = [f"{k}: {v}s" if isinstance(v, (int, float)) and "token" not in k.lower() and "ratio" not in k.lower() else f"{k}: {v}" for k, v in metrics.items()]
        print("   " + " | ".join(metrics_summary[:6]))
    print("=" * 70 + "\n")


def start_terminal_chat(
    collection_name: Optional[str] = None,
    pipeline_type: str = "adaptive",
    llm_choice: str = "qwen",
) -> None:
    """Run an interactive CLI chat session with multi-turn conversation memory."""
    from retrieval.node_store import NodeStore

    vstore = get_vector_store(collection_name=collection_name)
    has_docs = vstore.collection_exists() and bool(NodeStore.load())

    stats = vstore.get_collection_statistics() if vstore.collection_exists() else {"collection_name": collection_name or "production_rag", "vectors": 0}
    indexed_docs = vstore.list_documents() if has_docs else []

    print("\n" + "=" * 70)
    print("  PRODUCTION RAG - INTERACTIVE TERMINAL CHAT")
    print("=" * 70)
    print(f"  Collection : {stats.get('collection_name', 'production_rag')} ({stats.get('vectors', 0)} vectors)")
    print(f"  Pipeline   : {pipeline_type.upper()} | LLM: {llm_choice.upper()}")
    if indexed_docs:
        print(f"  Documents  : {', '.join(indexed_docs)}")
    else:
        print("  Documents  : None (Direct AI Chat Mode Active)")
    print("  Commands   : Type 'exit', 'quit', or 'q' to end session.")
    print("=" * 70 + "\n")

    history = []

    while True:
        try:
            question = input("You > ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q", "bye"}:
                print("\nExiting chat session. Goodbye!\n")
                break

            print("\nSearching knowledge base & generating answer...\n")
            result = query_rag(
                question=question,
                collection_name=collection_name,
                pipeline_type=pipeline_type,
                llm_choice=llm_choice,
                history=history,
            )

            print_query_result(result)

            # Record history
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result.get("answer", "")})

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated.")
            break
        except Exception as e:
            print(f"\n[Error during query]: {e}\n")


def get_database_status(collection_name: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve full database status."""
    from retrieval.node_store import NodeStore

    vstore = get_vector_store(collection_name=collection_name)
    exists = vstore.collection_exists()
    if not exists:
        return {"status": "Collection does not exist"}

    stats = vstore.get_collection_statistics()
    docs = vstore.list_documents()
    nodes = NodeStore.load()

    return {
        "collection_name": stats["collection_name"],
        "vectors_count": stats["vectors"],
        "dimension": stats["dimension"],
        "distance": stats["distance"],
        "indexed_documents": docs,
        "total_nodes": len(nodes),
    }


def clear_database(collection_name: Optional[str] = None) -> None:
    """Clear all vectors, nodes, and BM25 store."""
    from retrieval.node_store import NodeStore
    from retrieval.bm25_store import clear_bm25

    vstore = get_vector_store(collection_name=collection_name)
    vstore.clear()
    NodeStore.clear()
    clear_bm25()
    print("  [RAG] Knowledge base, vector collection, nodes, and BM25 index cleared successfully.")
