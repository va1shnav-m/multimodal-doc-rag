from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import json
from collections import Counter
from llama_index.core.node_parser import get_leaf_nodes
from ingestion.metadata_pipeline import MetadataPipeline
from ingestion.chunker import create_chunks
from ingestion.entities import extract_entities
from ingestion.keywords import extract_keywords
from ingestion.topics import extract_topics
from ingestion.knowledge_graph import build_knowledge_graph
from llama_index.core.schema import MetadataMode
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import build_bm25
from utils.logger import logger
from utils.metrics import record_metric, get_metrics
from analytics.document_analysis import (
    build_document_analytics
)
from report.report_generator import ReportGenerator
from ingestion.metadata_pipeline import tokenizer
from llama_index.core.schema import NodeRelationship

class IngestionPipeline:

    def __init__(self, collection_name=None):

        logger.info(
            "Initializing Ingestion Pipeline"
        )

        self.metadata_pipeline = MetadataPipeline()

        self.vector_store = VectorStore(collection_name=collection_name)

        self.index = None

    ##########################################################

    def ingest_pdf(
        self,
        uploaded_file
    ):
        raise NotImplementedError(
            "PDF conversion should be performed via PDFtoMarkdown_V3 / 'main.py convert' "
            "before ingesting the resulting markdown into the RAG vector store."
        )

    def ingest_markdown_file(
        self,
        markdown_path,
        metadata=None
    ):
        path = Path(markdown_path)
        logger.info(f"Ingesting Markdown: {path.name}")

        text = path.read_text(encoding="utf-8")

        doc_metadata = {
            "filename": path.name,
            "source_path": str(path.resolve()),
            "source_type": path.suffix,
        }
        if metadata:
            doc_metadata.update(metadata)

        parsed = {
            "text": text,
            "document_metadata": doc_metadata
        }

        return self.metadata_pipeline.build_document(
            parsed,
            filename=path.name
        )

    def ingest_markdown_text(
        self,
        text,
        filename="document.md",
        metadata=None
    ):
        logger.info(f"Ingesting Markdown Text: {filename}")

        doc_metadata = {
            "filename": filename,
            "source_type": ".md",
        }
        if metadata:
            doc_metadata.update(metadata)

        parsed = {
            "text": text,
            "document_metadata": doc_metadata
        }

        return self.metadata_pipeline.build_document(
            parsed,
            filename=filename
        )

    def ingest_batch_markdown(
        self,
        markdown_files,
        metadata_list=None
    ):
        documents = []
        for i, mf in enumerate(markdown_files):
            meta = metadata_list[i] if (metadata_list and i < len(metadata_list)) else None
            doc = self.ingest_markdown_file(mf, metadata=meta)
            documents.append(doc)
        return self.build_pipeline(documents)
    def process_node(
        self,
        node
    ):

        node.metadata["entities"] = extract_entities(
            node.text
        )

        node.metadata["keywords"] = extract_keywords(
            node.text
        )

        node.metadata["topics"] = extract_topics(
            node.text
        )

        if len(node.text) > 1000:

            node.metadata["knowledge_graph"] = (
                build_knowledge_graph(
                    node.text
                )
            )

        else:

            node.metadata["knowledge_graph"] = {

                "triples": [],

                "entity_graph": {}

            }
        ##################################################
        # Metadata Token Count

        metadata_for_tokens = {
            "entities": node.metadata["entities"],
            "keywords": node.metadata["keywords"],
            "topics": node.metadata["topics"],
            "knowledge_graph": node.metadata["knowledge_graph"]
        }

        metadata_text = json.dumps(
            metadata_for_tokens,
            default=str
        )

        node.metadata["metadata_tokens"] = len(
            # tokenizer.encode(
            #     metadata_text,
            #     add_special_tokens=False
            # )
            tokenizer.encode(metadata_text)
        )

        logger.info("=" * 80)
        logger.info(f"NODE ID: {node.node_id}")
        logger.info(f"Document: {node.metadata.get('filename')}")
        logger.info(f"Chunk Length: {len(node.text)} characters")
        logger.info("=" * 80)

        # -----------------------------
        # Metadata Summary
        # -----------------------------

        logger.info("Metadata Summary")
        logger.info(f"Entities   : {len(node.metadata['entities'])}")
        logger.info(f"Keywords   : {len(node.metadata['keywords'])}")
        logger.info(f"Topics     : {len(node.metadata['topics'])}")
        logger.info(
            f"KG Triples : {len(node.metadata['knowledge_graph']['triples'])}"
        )
        logger.info(
            f"KG Nodes   : {len(node.metadata['knowledge_graph']['entity_graph'])}"
        )

        logger.info("=" * 80)

        logger.info(
            node.get_content(
                metadata_mode=MetadataMode.ALL
            )
        )

        logger.info("=" * 80)

        return node

        
    ##########################################################

    def build_pipeline(
        self,
        documents
    ):
        if not documents:

            logger.info("No new documents received.")

            return {
                "index": self.vector_store.get_index(),
                "storage_context": self.vector_store.get_storage_context(),
                "nodes": []
            }

        logger.info(
            "Creating hierarchical nodes"
        )

        start = time.perf_counter()

        all_nodes = create_chunks(
            documents
        )
        node_lookup = {
            node.node_id: node
            for node in all_nodes
        }
        chunk_time = time.perf_counter() - start

        logger.info(
            f"Chunking: {chunk_time:.2f}s"
        )

        record_metric("Chunking", round(chunk_time, 3))
        ##################################################
        # Leaf Nodes
        ##################################################
        leaf_nodes = get_leaf_nodes(all_nodes)
        if leaf_nodes:
            logger.info("=" * 80)
            logger.info("INSPECTING HIERARCHY")
            logger.info(f"Total nodes: {len(all_nodes)}")
            logger.info(f"Leaf nodes: {len(leaf_nodes)}")
            logger.info(f"First Node ID: {all_nodes[0].node_id}")
            logger.info("=" * 80)

        ##################################################
        # Document Analytics
        ##################################################

        document_statistics = build_document_analytics(
            leaf_nodes,
            chunk_time
        )

        counter = Counter()

        for node in leaf_nodes:
            counter[node.metadata["filename"]] += 1

        logger.info("=" * 60)
        logger.info("Leaf nodes per document")
        logger.info(counter)
        logger.info("=" * 60)

##################################################
# Metadata Extraction
##################################################

        start = time.perf_counter()

        with ThreadPoolExecutor(
            max_workers=8
        ) as executor:

            list(
                executor.map(
                    self.process_node,
                    leaf_nodes
                )
            )

        metadata_time = time.perf_counter() - start

        record_metric("Metadata Extraction", round(metadata_time, 3))

        ##################################################
        # Inspection Report
        ##################################################

        report = ReportGenerator()

        report.pipeline_summary(

            embedding_model="E5-small-v2",

            embedding_dimension=384,

            vector_db="Qdrant",

            chunk_strategy="HierarchicalNodeParser",

            chunk_size=512,

            overlap=50,

            llm="gpt-4.1"

        )

##################################################
# Document Summary
##################################################

        for filename, stats in document_statistics.items():

            report.document_summary(

                filename=filename,

                pages=len(stats["pages"]),

                processing_time=stats["processing_time"],

                chunks=stats["chunks_created"],

                total_tokens=stats["total_chunk_tokens"],

                average_tokens=stats["average_chunk_tokens"],

                largest_chunk=stats["largest_chunk"],

                smallest_chunk=stats["smallest_chunk"]

            )

        ##################################################
        # Chunk Summary
        ##################################################

        total_chunks = len(leaf_nodes)

        for i, node in enumerate(
            leaf_nodes,
            start=1
        ):

            metadata = node.metadata

            report.chunk_summary(

                chunk_number=i,

                total_chunks=total_chunks,

                filename=metadata.get("filename"),

                page=metadata.get("page_number", "Unknown"),

                characters=len(node.text),

                chunk_tokens=metadata.get("chunk_tokens", 0),

                metadata_tokens=metadata.get("metadata_tokens", 0),

                entities=metadata.get("entities", []),

                keywords=metadata.get("keywords", []),

                topics=metadata.get("topics", []),

                knowledge_graph=node.metadata.get(
                        "knowledge_graph",
                        {}
                    ),

                text=node.text

            )
        ##################################################
        # Vector Index
        ##################################################

        start = time.perf_counter()

        filename = leaf_nodes[0].metadata["filename"]

        logger.info(
            f"Checking document : {filename}"
        )

        if self.vector_store.document_exists(filename):

                logger.info(
                    f"{filename} already exists."
                )

    ##################################################
    # OPTIONAL
    # Skip OR update
    ##################################################

                self.index = self.vector_store.load_index()

        else:

        ##################################################
        # First document?
        ##################################################

            if self.vector_store.collection_exists() and self.vector_store.list_documents():

                logger.info(
                    "Add new document"
                )

                self.index = self.vector_store.add_document(
                    all_nodes
                )

        ##################################################
        # Existing KB
        ##################################################

            else:

                logger.info(
                    "creating initial index"
                )

                self.index = self.vector_store.build_index(
                    all_nodes
                )

        vector_time = time.perf_counter() - start

        logger.info(
            f"Vector Index: {vector_time:.2f}s"
        )

        record_metric("Vector Index", round(vector_time, 3))

        ##################################################
        # BM25
        ##################################################

        bm25_documents = []

        for node in leaf_nodes:

            bm25_documents.append(

                {
                    "text": node.text,
                    "metadata": node.metadata,
                    "node_id": node.node_id
                }

            )

        start = time.perf_counter()

        build_bm25(
            bm25_documents
        )

        logger.info(
            f"BM25 Index: {time.perf_counter()-start:.2f}s"
        )

        logger.info(
            "Pipeline Ready"
        )
        ##################################################
        # Qdrant Inspection
        ##################################################

        points = self.vector_store.preview_vectors(
            limit=100
        )

        report.qdrant_summary(
            points,
            tokenizer
        )

        report_path = report.save()

        return {

            "index": self.index,

            "storage_context":
                self.vector_store.get_storage_context(),

            "nodes": all_nodes,

            "leaf_nodes": leaf_nodes,

            "report_path": report_path,

            "document_statistics": document_statistics,

            "metrics": get_metrics(),

        }