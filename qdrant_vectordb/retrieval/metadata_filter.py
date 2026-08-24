from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from utils.logger import logger


class MetadataFilterBuilder:

    def __init__(self):

        self.filters = []
    def add_filter(
        self,
        key,

        value,

        operator=FilterOperator.EQ,

    ):

        self.filters.append(

            MetadataFilter(

                key=key,

                value=value,

                operator=operator,

            )

        )

    ######################################################

    def build(self):

        logger.info(

            f"Building {len(self.filters)} metadata filters"

        )

        return MetadataFilters(

            filters=self.filters

        )

    ######################################################

    def clear(self):

        self.filters = []