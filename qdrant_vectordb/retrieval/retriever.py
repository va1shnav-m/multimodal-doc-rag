from utils.logger import logger
from utils.metrics import record_metric
from retrieval.settings import *
import time 
from retrieval.metadata_filter import (
    MetadataFilterBuilder
)
#from retrieval.adaptive_topk import AdaptiveTopK
class ProductionRetriever:

    ##########################################################

    def __init__(

        self,

        index,

        storage_context,
        nodes
        

    ):

        logger.info(
            "Initializing Retriever"
        )

        self.index = index
        #self.adaptive_topk = AdaptiveTopK()
        self.storage_context = storage_context
        self.nodes = nodes
        self.node_lookup = {

            node.node_id: node

            for node in nodes

        }
        ##################################################

        self.vector_retriever = (

            self.index.as_retriever(

                similarity_top_k=MAX_TOP_K

            )

        )

        ##################################################

        # self.auto_merging_retriever = (

        #     AutoMergingRetriever(

#                 self.vector_retriever,

#                 storage_context=self.storage_context,

#                 simple_ratio_thresh=AUTO_MERGE_THRESHOLD,

#                 verbose=True

# )

#         )

    ##########################################################

    def retrieve(

        self,

        query

    ):

        logger.info(

            f"Retrieving : {query}"

        )
        start = time.perf_counter()
        nodes = self.vector_retriever.retrieve(query)
        logger.info("=" * 80)
        logger.info("DENSE RETRIEVAL RESULTS")

        for i, node in enumerate(nodes):
            logger.info(
                f"{i+1}. "
                f"{node.metadata.get('filename')} | "
                f"Score={node.score:.4f}"
            )

        logger.info("=" * 80)

        initial_count = len(nodes)

       # nodes = self.adaptive_topk.select(nodes)

        selected_count = len(nodes)
        retrieval_time = time.perf_counter() - start

        logger.info(

            f"Retrieved {len(nodes)} nodes"

        )
        record_metric("Qdrant Retrieval", round(retrieval_time, 3))
        return nodes

    ##########################################################

    def retrieve_with_filters(

        self,

        query,

        metadata_filters

    ):

        logger.info(

            "Applying Metadata Filters"

        )

        retriever = self.index.as_retriever(

            similarity_top_k=MAX_TOP_K,

            filters=metadata_filters

        )

        # retriever = AutoMergingRetriever(

        #     retriever,

        #     storage_context=self.storage_context,

        #     simple_ratio_thresh=AUTO_MERGE_THRESHOLD

        # )
        # retriever = AutoMergingRetriever(

        #     retriever,

        #     storage_context=self.storage_context,

        #     simple_ratio_thresh=AUTO_MERGE_THRESHOLD

        # )
        retriever = self.index.as_retriever(
            similarity_top_k=MAX_TOP_K,
            filters=metadata_filters
        )
        start = time.perf_counter()

        nodes = retriever.retrieve(query)
        retrieval_time = time.perf_counter() - start

        nodes = retriever.retrieve(

            query

        )

        logger.info(

            f"{len(nodes)} filtered nodes retrieved"

        )
        logger.info(

                f"Filtered Retrieval Time : {retrieval_time:.3f}s"

            )

        return nodes

    ##########################################################

    def retrieve_by_filename(

        self,

        filename,

        query

    ):

        builder = MetadataFilterBuilder()

        builder.add_filter(

            "filename",

            filename

        )

        return self.retrieve_with_filters(

            query,

            builder.build()

        )

    ##########################################################

    def retrieve_by_entity(

        self,

        entity,

        query

    ):

        builder = MetadataFilterBuilder()

        builder.add_filter(

            "entities",

            entity

        )

        return self.retrieve_with_filters(

            query,

            builder.build()

        )

    ##########################################################

    def retrieve_by_keyword(

        self,

        keyword,

        query

    ):

        builder = MetadataFilterBuilder()

        builder.add_filter(

            "keywords",

            keyword

        )

        return self.retrieve_with_filters(

            query,

            builder.build()

        )

    ##########################################################

    def retrieve_by_topic(

        self,

        topic,

        query

    ):

        builder = MetadataFilterBuilder()

        builder.add_filter(

            "topics",

            topic

        )

        return self.retrieve_with_filters(

            query,

            builder.build()

        )

    ##########################################################

    def get_vector_retriever(

        self

    ):

        return self.vector_retriever

    ##########################################################

    # def get_auto_merging_retriever(

    #     self

    # ):

    #     return self.auto_merging_retriever