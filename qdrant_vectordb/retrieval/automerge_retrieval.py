from collections import defaultdict
from llama_index.core.schema import NodeRelationship
from utils.logger import logger


class AutoMerge:

    def __init__(self, node_lookup):
        """
        node_lookup:
            {
                node_id -> BaseNode
            }
        """
        self.node_lookup = node_lookup

    def merge(
        self,
        retrieved_documents,
        min_children=2,
    ):

        logger.info("=" * 60)
        logger.info("AUTOMERGE")
        logger.info("=" * 60)

        parent_groups = defaultdict(list)

        standalone = []

        ##################################################
        # Group retrieved chunks by parent
        ##################################################

        for item in retrieved_documents:

            node = item.get("node")

            if node is None:
                standalone.append(item)
                continue

            parent = node.relationships.get(
                NodeRelationship.PARENT
            )

            if parent is None:
                standalone.append(item)
                continue

            parent_groups[parent.node_id].append(item)

        merged = []

        ##################################################
        # Merge siblings
        ##################################################

        for parent_id, children in parent_groups.items():

            logger.info(
                f"Parent {parent_id} -> "
                f"{len(children)} retrieved children"
            )

            ##################################################

            if len(children) < min_children:

                merged.extend(children)
                continue

            ##################################################

            parent_node = self.node_lookup.get(parent_id)

            if parent_node is None:

                merged.extend(children)
                continue

            ##################################################

            logger.info(
                f"Merged into parent {parent_id}"
            )

            merged.append(

                {

                    "text": parent_node.text,

                    "metadata": parent_node.metadata,

                    "node_id": parent_node.node_id,

                    "node": parent_node,

                }

            )

        ##################################################

        merged.extend(standalone)

        logger.info(
            f"Before merge : {len(retrieved_documents)}"
        )

        logger.info(
            f"After merge : {len(merged)}"
        )

        logger.info("=" * 60)

        return merged