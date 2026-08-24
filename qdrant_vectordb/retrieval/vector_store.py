from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    Filter,
    FilterSelector,
    FieldCondition,
    MatchValue,
    HnswConfigDiff
)
from functools import lru_cache
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex
)
from llama_index.core.node_parser import (
    get_leaf_nodes
)
from retrieval.node_store import NodeStore

from llama_index.core.storage.docstore import (
    SimpleDocumentStore
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding
)

from llama_index.vector_stores.qdrant import (
    QdrantVectorStore
)

from retrieval.settings import *

from utils.logger import logger


@lru_cache(maxsize=1)
def load_embedding(model_name=EMBEDDING_MODEL):
    return HuggingFaceEmbedding(
        model_name=model_name
    )


_SHARED_QDRANT_CLIENT = None


def get_shared_qdrant_client(path=None):
    global _SHARED_QDRANT_CLIENT
    if _SHARED_QDRANT_CLIENT is None:
        target_path = path or str(QDRANT_PATH)
        _SHARED_QDRANT_CLIENT = QdrantClient(path=target_path)
    return _SHARED_QDRANT_CLIENT


class VectorStore:

    ##########################################################
    # Initialization
    ##########################################################

    def __init__(
        self,

        collection_name=None,

        hnsw_config=None
        ):

            logger.info(
                "Initializing Production Vector Store"
            )
            self.collection_name = (
            collection_name
            if collection_name
            else COLLECTION_NAME
        )

            self.hnsw_config = (
                hnsw_config
                if hnsw_config
                else HnswConfigDiff(
                    m=16,
                    ef_construct=100
                )
            )

        ##################################################
        # Qdrant Client
        ##################################################

            self.client = get_shared_qdrant_client(
                path=str(QDRANT_PATH)
            )

            if not self.collection_exists():

                logger.info(
                    "Creating Qdrant Collection"
                )

                self.client.create_collection(

                    collection_name=self.collection_name,

                    vectors_config=VectorParams(

                        size=VECTOR_SIZE,

                        distance=Distance.COSINE

                    ),
                    hnsw_config=self.hnsw_config

                )

            ##################################################
            # Vector Store
            ##################################################

            self.vector_store = QdrantVectorStore(

                client=self.client,

                collection_name=self.collection_name

            )

            ##################################################
            # Doc Store
            ##################################################

            self.docstore = SimpleDocumentStore()

            ##################################################
            # Storage Context
            ##################################################

            self.storage_context = StorageContext.from_defaults(

                vector_store=self.vector_store,

                docstore=self.docstore

            )

            ##################################################

            self.index = None

            logger.info(
                "Vector Store Ready"
            )
            ##########################################################
            # Initial Index Creation
            ##########################################################

    def _ensure_embed_model(self):
        Settings.embed_model = load_embedding()

    def build_index(
        self,
        all_nodes
    ):
        self._ensure_embed_model()
        logger.info(
            "Building Initial Vector Index"
        )

        leaf_nodes = get_leaf_nodes(
            all_nodes
        )

        logger.info(
            f"Total Nodes : {len(all_nodes)}"
        )

        logger.info(
            f"Leaf Nodes : {len(leaf_nodes)}"
        )

        ##################################################
        # Store Nodes
        ##################################################

        for node in all_nodes:

            try:

                self.docstore.add_documents(
                    [node]
                )

            except Exception as e:

                logger.warning(
                    f"Could not add node {node.node_id}: {e}"
                )

        logger.info(
            "Document hierarchy stored"
        )

        ##################################################
        # Sample Log
        ##################################################

        sample_node = leaf_nodes[0]

        logger.info("=" * 80)
        logger.info("TEXT SENT TO EMBEDDING MODEL")
        logger.info("=" * 80)

        logger.info(sample_node.text)

        logger.info("=" * 80)
        logger.info("METADATA")
        logger.info(sample_node.metadata)
        logger.info("=" * 80)

        ##################################################
        # Create Index
        ##################################################

        self.index = VectorStoreIndex(

            leaf_nodes,

            storage_context=self.storage_context

        )

        ##################################################

        collection = self.client.get_collection(
            self.collection_name
        )

        logger.info("=" * 60)
        logger.info(
            f"Vectors Stored : {collection.points_count}"
        )
        logger.info("=" * 60)

        logger.info(
            "Initial Index Ready"
        )
        points, _ = self.client.scroll(

            collection_name=self.collection_name,

            limit=1,

            with_payload=True,

            with_vectors=False

        )

        logger.info("=" * 80)
        logger.info("FIRST POINT IN QDRANT")
        logger.info(points[0])
        logger.info("=" * 80)
        NodeStore.save(all_nodes)
        return self.index
    def add_document(

        self,

        all_nodes

    ):
        self._ensure_embed_model()
        logger.info(
            "Adding New Document"
        )

        leaf_nodes = get_leaf_nodes(
            all_nodes
        )

        ##################################################
        # Store hierarchy
        ##################################################

        for node in all_nodes:

            try:

                self.docstore.add_documents(
                    [node]
                )

            except Exception as e:

                logger.warning(
                    f"Could not add node {node.node_id}: {e}"
                )

        ##################################################
        # Load index if required
        ##################################################

        if self.index is None:

            logger.info(
                "Loading Existing Index"
            )

            self.load_index()

        ##################################################
        # Insert New Nodes
        ##################################################

        self.index.insert_nodes(
            leaf_nodes
        )

        ##################################################

        collection = self.client.get_collection(
            self.collection_name
        )

        logger.info("=" * 60)
        logger.info(
            f"Vectors Stored : {collection.points_count}"
        )
        logger.info("=" * 60)

        logger.info(
            "Document Added Successfully"
        )
        NodeStore.append(all_nodes)
        return self.index
    def update_document(

        self,

        filename,

        all_nodes

    ):

        logger.info(
            f"Updating {filename}"
        )
        self.delete_document(filename)
        self.add_document(all_nodes)
        NodeStore.replace_document(
            filename,
            all_nodes
        )
        

    def load_index(self):
        self._ensure_embed_model()
        logger.info("Loading Existing Index")

        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context
        )

        return self.index
    ##########################################################
    # Document Operations
    ##########################################################

    def document_exists(
        self,
        filename
    ):

        logger.info(
            f"Checking if {filename} exists"
        )

        ##################################################
        # TODO
        #
        # Change payload key after inspecting Qdrant payload.
        ##################################################

        points, _ = self.client.scroll(

            collection_name=self.collection_name,

            limit=1,

            with_payload=True,

            with_vectors=False,

            scroll_filter=Filter(

                must=[

                    FieldCondition(

                        key="filename",

                        match=MatchValue(
                            value=filename
                        )

                    )

                ]

            )

        )

        exists = len(points) > 0

        logger.info(
            f"Document Exists : {exists}"
        )

        return exists
    def list_documents(self):

        logger.info(
            "Listing Indexed Documents"
        )

        documents = set()

        offset = None

        while True:

            points, offset = self.client.scroll(

                collection_name=self.collection_name,

                limit=100,

                with_payload=True,

                with_vectors=False,

                offset=offset

            )

            if not points:

                break

            for point in points:

                payload = point.payload

                ##################################################
                # TODO
                #
                # Update payload key if required.
                ##################################################

                filename = payload.get(
                    "filename"
                )

                if filename:

                    documents.add(
                        filename
                    )

            if offset is None:

                break

        documents = sorted(
            list(documents)
        )

        logger.info("=" * 60)
        logger.info(
            f"Indexed Documents : {documents}"
        )
        logger.info("=" * 60)

        return documents
    def delete_document(
        self,
        filename
    ):

        logger.info(
            f"Deleting {filename}"
        )

        ##################################################
        # TODO
        #
        # Change payload key after checking payload.
        ##################################################

        self.client.delete(

            collection_name=self.collection_name,

            points_selector=Filter(

                must=[

                    FieldCondition(

                        key="filename",

                        match=MatchValue(
                            value=filename
                        )

                    )

                ]

            )

        )

        collection = self.client.get_collection(
            self.collection_name
        )
        logger.info("=" * 60)
        logger.info(
            f"Remaining Vectors : {collection.points_count}"
        )
        logger.info("=" * 60)

        logger.info(
            "Document Deleted"
        )
        nodes = NodeStore.load()

        nodes = [
            node
            for node in nodes
            if node.metadata.get("filename") != filename
        ]

        NodeStore.save(nodes)
    def get_index(self):
        return self.index


    def get_storage_context(self):
        return self.storage_context


    def get_docstore(self):
        return self.docstore


    def get_vector_store(self):
        return self.vector_store
    ##########################################################
    # Collection Management
    ##########################################################

    def collection_exists(self):

        try:

            collections = self.client.get_collections()

            names = [

                collection.name

                for collection in collections.collections

            ]

            return self.collection_name in names

        except Exception:

            return False
    ##########################################################

    def delete_collection(self):

        logger.warning(
            "Deleting Collection"
        )

        try:

            self.client.delete_collection(
                self.collection_name
            )

        except Exception as e:

            logger.exception(e)
    ##########################################################
    # Collection Statistics
    ##########################################################

    def get_collection_statistics(self):

        collection = self.client.get_collection(
            self.collection_name
        )

        return {

            "collection_name": self.collection_name,

            "vectors": collection.points_count,

            "dimension": VECTOR_SIZE,

            "distance": "Cosine"

        }
    def preview_vectors(self, limit=10):

        points, _ = self.client.scroll(

            collection_name=self.collection_name,

            limit=limit,

            with_payload=True,

            with_vectors=True

        )

        print(points[0])

        return points

    def clear(self):

        logger.warning(
            "Clearing Knowledge Base"
        )

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=Filter())
            )
        except Exception:
            pass

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                ),
                hnsw_config=self.hnsw_config
            )
        except Exception as e:
            logger.info(f"Collection status after clear: {e}")

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name
        )

        self.docstore = SimpleDocumentStore()

        self.storage_context = StorageContext.from_defaults(

            vector_store=self.vector_store,

            docstore=self.docstore

        )

        self.index = None

        logger.info(
            "Knowledge Base Reset"
        )
        NodeStore.clear()