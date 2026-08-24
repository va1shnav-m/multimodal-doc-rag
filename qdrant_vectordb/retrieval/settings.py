from pathlib import Path

# Base storage directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = PROJECT_ROOT / "storage"
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Qdrant
# ==========================================================

QDRANT_PATH = STORAGE_ROOT / "qdrant"
QDRANT_PATH.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "production_rag"

VECTOR_SIZE = 384

DISTANCE = "Cosine"
# ==========================================================
# Chunking
# ==========================================================

PARENT_CHUNK_SIZE = 1500
MIDDLE_CHUNK_SIZE = 1000
CHILD_CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

CHUNK_SIZES = [
    PARENT_CHUNK_SIZE,
    MIDDLE_CHUNK_SIZE,
    CHILD_CHUNK_SIZE
]
# ==========================================================
# Hybrid Search
# ==========================================================

ENABLE_BM25 = True

ENABLE_RERANKER = True

ENABLE_QUERY_EXPANSION = True

ENABLE_PARENT_RETRIEVER = True


ENABLE_RECURSIVE_RETRIEVER = True


# ==========================================================
# Metadata
# ==========================================================

MAX_KEYWORDS = 10

MAX_ENTITIES = 20


# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = "INFO"
# ==========================================================
# Retrieval
# ==========================================================
TOP_K = 15               # was 20(answers ok)
BM25_TOP_K = 10           # was 10

RERANK_TOP_K = 10          # was 8

FINAL_CONTEXT_K = 8        # was 5

MAX_TOP_K = 30

MAX_CHUNKS_PER_DOCUMENT = 3    # was 2

AUTO_MERGE_THRESHOLD = 0.5

# Models


EMBEDDING_MODEL = "intfloat/e5-small-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"
MAX_CHUNKS_PER_DOCUMENT = 2
ENABLE_CONTEXT_COMPRESSION = True

COMPRESSION_RATE = 0.5

LLMLINGUA_MODEL = (
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
)