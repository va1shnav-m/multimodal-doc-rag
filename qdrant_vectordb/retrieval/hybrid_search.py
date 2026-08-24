# BM25 + dense fusion.
from collections import defaultdict
import time
from retrieval.retriever import ProductionRetriever
from ranking.reranker import rerank_results
from retrieval.bm25_store import search_bm25
from retrieval.citation import (
    get_documents_used,
    get_supporting_evidence
)
from retrieval.adaptive_topk import AdaptiveTopK
from utils.logger import logger
from retrieval.settings import (
    BM25_TOP_K,
    RERANK_TOP_K,
    FINAL_CONTEXT_K,
    MAX_CHUNKS_PER_DOCUMENT
)
from context_engine import ContextCompressor
# from retrieval.retrieval_uncertainity import (
#     RetrievalUncertaintyEstimator
# # )
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(
#     "Qwen/Qwen2.5-7B"
# )
from retrieval.retrieval_evaluator import RetrievalEvaluator
from retrieval.query_expansion import (
    expand_query
)
from utils.metrics import record_metric
from retrieval.retrieval_statistics import RetrievalStatistics
from retrieval.feedback_engine import FeedbackEngine
from retrieval.automerge_retrieval import AutoMerge
class HybridSearch:

    def __init__(

        self,

        retriever

    ):

        self.retriever = retriever
        self.evaluator = RetrievalEvaluator()
        self.statistics = RetrievalStatistics()
        self.feedback = FeedbackEngine()
        self.context_compressor = ContextCompressor()
    def reciprocal_rank_fusion(

        self,

        dense_results,

        sparse_results,

        k=60

    ):

        logger.info(
            "Running Reciprocal Rank Fusion"
        )

        scores = defaultdict(float)

        node_lookup = {}

        ##################################################

        for rank, node in enumerate(

            dense_results

        ):

            node_id = node.node.node_id

            node_lookup[node_id] = node

            scores[node_id] += 1 / (k + rank + 1)

        ##################################################

        for rank, chunk in enumerate(

            sparse_results

        ):

            node_id = chunk.get(

                "node_id",

                chunk["text"]

            )

            node_lookup[node_id] = chunk

            scores[node_id] += 1 / (k + rank + 1)

        ##################################################

        ranked = sorted(

            scores.items(),

            key=lambda x: x[1],

            reverse=True

        )

        results = []

        for node_id, _ in ranked:

            results.append(

                node_lookup[node_id]

            )

        return results

    #######################################################

    def collect_metadata(

        self,

        fused_results

    ):

        logger.info(

            "Collecting metadata from retrieved chunks"

        )

        metadata_list = []

        ##################################################

        for item in fused_results:

            ##############################################
            # LlamaIndex Node
            ##############################################

            if hasattr(

                item,

                "node"

            ):

                metadata = item.node.metadata

            ##############################################
            # BM25 Result
            ##############################################

            else:

                metadata = item.get(

                    "metadata",

                    {}

                )

            ##############################################

            metadata_list.append(

                metadata

            )
        logger.info(

            f"Collected metadata from {len(metadata_list)} chunks"

        )

        return metadata_list
    def retrieve(

        self,

        query

    ):
        overall_start=time.perf_counter()
        logger.info(

            f"Hybrid Retrieval : {query}"

        )
        ##################################################
        # Stage 1 Retrieval
        ##################################################

        logger.info(
            "Stage 1 Retrieval"
        )
        
        dense_results = self.retriever.retrieve(
            query
        )

        sparse_results = search_bm25(
            query,
            top_k=BM25_TOP_K
        )
        fusion_start = time.perf_counter()

        fused_results = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results
        )
        logger.info("=" * 80)
        logger.info("RRF RESULTS")

        for i, item in enumerate(fused_results):

            if hasattr(item, "node"):      # Dense/Qdrant result
                filename = item.node.metadata.get("filename", "Unknown")
                score = item.score

            else:                          # BM25 result
                filename = item.get("metadata", {}).get("filename", "Unknown")
                score = item.get("score", "N/A")

            logger.info(f"{i+1}. {filename} | Score={score}")

        logger.info("=" * 80)

        fusion_time = time.perf_counter() - fusion_start

        record_metric(
            "RRF Fusion",
            round(fusion_time * 1000, 2)
        )
        
        ##################################################
        # Collect Metadata
        ##################################################

        metadata_list = self.collect_metadata(
            fused_results[:10]
        )

        ##################################################
        # Expand Query
        ##################################################

        if len(query.split()) <= 5:
            query_expansion_start = time.perf_counter()
            expanded_query = expand_query(
                query,
                metadata_list
            )
            query_expansion_time = (
                time.perf_counter() - query_expansion_start
            )


            record_metric(
                "Query Expansion",
                round(query_expansion_time, 3)
            )
            logger.info(f"Expanded Query: {expanded_query}")

        else:

            expanded_query = query

            logger.info("Skipping query expansion")
        logger.info(
        "Stage 2 Retrieval"
    )
        
        dense_results = self.retriever.retrieve(
            expanded_query
        )
        logger.info("=" * 80)
        logger.info("DENSE RESULTS")

        for i, node in enumerate(dense_results):

            logger.info(
                f"{i+1}. "
                f"{node.node.metadata.get('filename', 'Unknown')} | "
                f"Score={node.score:.4f}"
            )

        logger.info("=" * 80)
        

        sparse_results = search_bm25(
            expanded_query,
            top_k=BM25_TOP_K
        )
        logger.info("=" * 80)
        logger.info("BM25 RESULTS")

        for i, item in enumerate(sparse_results):

            metadata = item.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            score = item.get("score", "N/A")

            logger.info(
                f"{i+1}. {filename} | Score={score}"
            )

        logger.info("=" * 80)
        
        fused_results = self.reciprocal_rank_fusion(
            dense_results,
            sparse_results)
        

        rerank_documents = []

        for item in fused_results:

            ##################################################

            if hasattr(item, "node"):

                rerank_documents.append(

                    {

                        "text": item.node.text,
                        "metadata": item.node.metadata,
                        "node_id": item.node.node_id,
                        "node": item.node

                    }

                )

            ##################################################

            else:

                rerank_documents.append(

                    {

                        "text": item["text"],
                        "metadata": item.get(
                                "metadata",
                                {}
                            )

                    }

                )

        ##################################################
        rerank_start = time.perf_counter()
        rerank_documents = rerank_documents
        auto_merge = AutoMerge(

                 self.retriever.node_lookup

             )

        rerank_documents = auto_merge.merge(
             rerank_documents,
            min_children=2,
        )
        reranked = rerank_results(

            expanded_query,

            rerank_documents,

            RERANK_TOP_K

        )
        rerank_time = time.perf_counter() - rerank_start

        record_metric(
            "Reranking",
            round(rerank_time, 3)
        )
        ##########################################################
        # Retrieval Uncertainty
        ##########################################################

        # uncertainty = self.uncertainty.compute(
        #     dense_results=dense_results,
        #     bm25_results=sparse_results,
        #     reranked_results=reranked
        # )

        # logger.info(
        #     f"Retrieval Confidence : {uncertainty['overall']:.3f}"
        # )

        # st.session_state.metrics["Retrieval Confidence"] = round(
        #     uncertainty["overall"],
        #     3
        # )
        # st.session_state.metrics["Softmax"] = round(
        #     uncertainty["softmax"],
        #     3
        # )

        # st.session_state.metrics["Margin"] = round(
        #     uncertainty["margin"],
        #     3
        # )

        # st.session_state.metrics["Entropy"] = round(
        #     uncertainty["entropy"],
        #     3
        # )

        # st.session_state.metrics["Consensus"] = round(
        #     uncertainty["consensus"],
        #     3
        # )

        # st.session_state.metrics["Reranker Stability"] = round(
        #     uncertainty["reranker_stability"],
        #     3
        # )
        ##################################################
        # Remove duplicate chunks
        ##################################################

        unique_docs = []

        seen = set()

        for doc, score in reranked:

            text = doc["text"][:200]

            if text in seen:
                continue

            seen.add(text)

            unique_docs.append(
                (doc, score)
            )

        ##################################################
        # Use unique chunks
        ##################################################

        reranked = unique_docs

        ##################################################
        # Document diversity
        ##################################################

        selected = []

        document_count = {}


        for doc, score in reranked:

            filename = doc["metadata"].get(
                "filename",
                "Unknown"
            )

            if document_count.get(filename, 0) >= MAX_CHUNKS_PER_DOCUMENT:
                continue

            selected.append(
                (doc, score)
            )

            document_count[filename] = (
                document_count.get(filename, 0) + 1
            )

            if len(selected) == FINAL_CONTEXT_K:
                break

        reranked = selected
        evaluation = self.evaluator.evaluate(

            query=query,
            reranked_results=reranked,
            dense_results=dense_results,

            bm25_results=sparse_results)
    
        
        ##################################################

        self.statistics.update(

            evaluation

        )

        statistics = self.statistics.compute()

        feedback = self.feedback.decide(

            evaluation,

            statistics

        )
                
        documents_used = get_documents_used(

            reranked

        )

        supporting_evidence = get_supporting_evidence(

            reranked

        )

        ##################################################
        context_start = time.perf_counter()
        context_parts = []

        for doc, score in reranked:

            metadata = doc.get("metadata", {})

            filename = metadata.get(
                "filename",
                "Unknown"
            )

            page = metadata.get(
                "page",
                metadata.get("page_number", "Unknown")
            )

            context_parts.append(

                f"""
        Document: {filename}
        Page: {page}

        {doc["text"]}

        ----------------------------------------
        """
            )

        context = "\n".join(context_parts)
        compression_result = self.context_compressor.compress(context)
        context = compression_result["compressed_context"]
        record_metric(
            "Compression Time",
            round(compression_result["compression_time"], 3)
        )
        record_metric(
            "Compression Ratio",
            compression_result["compression_ratio"]
        )
        record_metric(
            "Original Tokens",
            compression_result["original_tokens"]
        )
        record_metric(
            "Compressed Tokens",
            compression_result["compressed_tokens"]
        )
        chunk_tokens = []
        metadata_tokens = []
        metadata_ratios = []
        similarity_scores = []

        for doc, score in reranked:

            metadata = doc.get("metadata", {})

            chunk_tokens.append(
                metadata.get("chunk_tokens", 0)
            )

            metadata_tokens.append(
                metadata.get("metadata_tokens", 0)
            )

            metadata_ratios.append(
                metadata.get("metadata_ratio", 0)
            )

            similarity_scores.append(score)

        ##########################################################

        original_context_tokens = compression_result["original_tokens"]

        compressed_context_tokens = compression_result["compressed_tokens"]

        analytics = {

            "retrieved_chunks": len(reranked),

            "documents_used": len(documents_used),

            "original_context_tokens": original_context_tokens,
            "compressed_context_tokens": compressed_context_tokens,
            "compression_ratio": compression_result["compression_ratio"],
            "average_chunk_tokens": round(

                sum(chunk_tokens) /

                max(len(chunk_tokens), 1),

                2

            ),

            "average_metadata_tokens": round(

                sum(metadata_tokens) /

                max(len(metadata_tokens), 1),

                2

            ),

            "average_metadata_ratio": round(

                sum(metadata_ratios) /

                max(len(metadata_ratios), 1),

                2

            ),

            "average_similarity": round(

                sum(similarity_scores) /

                max(len(similarity_scores), 1),

                4

            ),
            #"retrieval_uncertainity": uncertainty

        }
        logger.info(
            f"Context Construction Time : {time.perf_counter() - context_start:.3f}s"
        )

        logger.info(
            "Hybrid Search Completed"
        )
        overall_time = time.perf_counter() - overall_start

        logger.info("=" * 60)
        logger.info(f"Total Hybrid Search Time : {overall_time:.3f}s")
        logger.info("=" * 60)
        return {

    "context": context,

    "reranked_results": reranked,

    "documents_used": documents_used,

    "supporting_evidence": supporting_evidence,

    "dense_results": dense_results,

    "bm25_results": sparse_results,

    "expanded_query": expanded_query,

    "retrieval_report": evaluation,

    "feedback": feedback,

    #########################################
    # NEW
    #########################################

    "analytics": analytics

}