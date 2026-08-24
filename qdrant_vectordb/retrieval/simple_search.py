import time

from retrieval.citation import (
    get_documents_used,
    get_supporting_evidence,
)
from utils.logger import logger
from utils.metrics import record_metric
#from transformers import AutoTokenizer
from retrieval.retrieval_evaluator import RetrievalEvaluator
# tokenizer = AutoTokenizer.from_pretrained(
#     "Qwen/Qwen2.5-7B"
# )
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4.1")
class SimpleSearch:

    def __init__(self, retriever):

        self.retriever = retriever
        self.evaluator = RetrievalEvaluator()

    def retrieve(self, query):

        overall_start = time.perf_counter()

        dense_results = self.retriever.retrieve(query)
        logger.info("=" * 60)
        logger.info("Simple Retrieval Scores")

        for i, node in enumerate(dense_results[:5], start=1):
            logger.info(
                f"{i}. Score={node.score:.4f} | "
                f"{node.node.metadata.get('filename', 'Unknown')}"
            )

        logger.info("=" * 60)

        documents = []

        for item in dense_results:

            documents.append(
                {
                    "text": item.node.text,
                    "metadata": item.node.metadata,
                    "node": item.node,
                }
            )

        context_parts = []

        for doc in documents[:5]:

            metadata = doc["metadata"]

            filename = metadata.get(
                "filename",
                "Unknown"
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"""
Document : {filename}
Page : {page}

{doc['text']}

---------------------------------------
"""
            )

        context = "\n".join(context_parts)

        documents_used = get_documents_used(
            [(d,1.0) for d in documents[:5]]
        )

        supporting_evidence = get_supporting_evidence(
            [(d,1.0) for d in documents[:5]]
        )

        # context_tokens = len(
        #     tokenizer.encode(
        #         context,
        #         add_special_tokens=False
        #     )
        # )
        context_tokens = len(tokenizer.encode(context))

        

        analytics = {

            "retrieved_chunks": len(documents),

            "documents_used": len(documents_used),

            "original_context_tokens": context_tokens,

            "compressed_context_tokens": context_tokens,

            "compression_ratio": "1.0x",

            "average_similarity": "-",

            }

        
        retrieval_report = self.evaluator.evaluate(
            query=query,
            reranked_results=[(doc, 1.0) for doc in documents[:5]],
            dense_results=dense_results,
            bm25_results=[]
        )

        record_metric(
            "Simple Retrieval",
            round(time.perf_counter() - overall_start, 3)
        )

        return {

            "context":context,

            "reranked_results":[
                (doc,1.0)
                for doc in documents[:5]
            ],

            "documents_used":
                documents_used,

            "supporting_evidence":
                supporting_evidence,

            "dense_results":
                dense_results,

            "bm25_results":None,

            "expanded_query":
                query,

            "retrieval_report":retrieval_report,

            "feedback":None,
            "analytics":analytics,
            "routing": {
                    "pipeline": "fast"}
           

        }