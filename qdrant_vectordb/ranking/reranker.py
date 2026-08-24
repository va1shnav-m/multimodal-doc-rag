from functools import lru_cache
from sentence_transformers import CrossEncoder
import time
from retrieval.settings import RERANKER_MODEL
from utils.logger import logger
from utils.metrics import record_metric


@lru_cache(maxsize=1)
def load_reranker():
    logger.info("Loading BGE Reranker")
    return CrossEncoder(RERANKER_MODEL)


def rerank_results(
    expanded_query,
    documents,
    top_k=5
):
    if not documents:
        return []

    reranker = load_reranker()

    pairs = [
        [expanded_query, doc["text"]]
        for doc in documents
    ]
    start = time.perf_counter()
    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )
    rerank_time = time.perf_counter() - start
    record_metric("Re-ranking", round(rerank_time, 3))
    ranked = sorted(

        zip(documents, scores),

        key=lambda x: x[1],

        reverse=True

    )

    return ranked[:top_k]