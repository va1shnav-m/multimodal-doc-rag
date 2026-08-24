from rank_bm25 import BM25Okapi
import numpy as np
import pickle
import time
from retrieval.settings import STORAGE_ROOT
from utils.logger import logger
from utils.metrics import record_metric


##################################################
# Storage
##################################################

BM25_DIR = STORAGE_ROOT / "bm25"

BM25_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BM25_FILE = BM25_DIR / "bm25.pkl"

CHUNKS_FILE = BM25_DIR / "chunks.pkl"


##################################################
# Globals
##################################################

bm25 = None

chunks_store = []


##################################################
# Build
##################################################

def build_bm25(chunks):

    global bm25, chunks_store

    tokenized_chunks = []

    for chunk in chunks:

        document = f"""
        {chunk.get("title", "")}

        {chunk["text"]}
        """

        tokenized_chunks.append(

            document.lower().split()

        )

    bm25 = BM25Okapi(

        tokenized_chunks

    )

    chunks_store = chunks

    ##################################################
    # Persist
    ##################################################

    with open(BM25_FILE, "wb") as f:

        pickle.dump(

            bm25,

            f

        )

    with open(CHUNKS_FILE, "wb") as f:

        pickle.dump(

            chunks_store,

            f

        )

    logger.info("=" * 80)
    logger.info("BM25 persisted successfully.")
    logger.info("=" * 80)


##################################################
# Load
##################################################

def load_bm25():

    global bm25, chunks_store

    if not BM25_FILE.exists():

        logger.warning(

            "BM25 file not found."

        )

        return

    if not CHUNKS_FILE.exists():

        logger.warning(

            "Chunk store not found."

        )

        return

    with open(BM25_FILE, "rb") as f:

        bm25 = pickle.load(f)

    with open(CHUNKS_FILE, "rb") as f:

        chunks_store = pickle.load(f)

    logger.info("=" * 80)
    logger.info("BM25 loaded successfully.")
    logger.info(f"Chunks Loaded : {len(chunks_store)}")
    logger.info("=" * 80)


##################################################
# Search
##################################################

def search_bm25(

    expanded_query,

    top_k=20

):

    global bm25

    if bm25 is None:

        load_bm25()

    if bm25 is None:

        raise RuntimeError(

            "BM25 index not found."

        )

    tokenized_query = (

        expanded_query.lower().split()

    )

    start = time.perf_counter()

    scores = bm25.get_scores(

        tokenized_query

    )

    bm25_time = time.perf_counter() - start

    record_metric(
        "BM25 Retrieval",
        round(bm25_time, 3)
    )

    top_indices = np.argsort(

        scores

    )[::-1][:top_k]

    return [

        chunks_store[i]

        for i in top_indices

    ]


##################################################
# Delete
##################################################

def delete_document(filename):

    global bm25, chunks_store

    ##################################################
    # Remove chunks
    ##################################################

    remaining_chunks = [

        chunk

        for chunk in chunks_store

        if chunk["metadata"].get("filename") != filename

    ]

    ##################################################
    # Rebuild
    ##################################################

    build_bm25(

        remaining_chunks

    )

    logger.info(

        f"{filename} removed from BM25."

    )


def clear_bm25():
    global bm25, chunks_store
    bm25 = None
    chunks_store = []
    if BM25_FILE.exists():
        try:
            BM25_FILE.unlink()
        except Exception:
            pass
    if CHUNKS_FILE.exists():
        try:
            CHUNKS_FILE.unlink()
        except Exception:
            pass
    logger.info("BM25 store cleared.")