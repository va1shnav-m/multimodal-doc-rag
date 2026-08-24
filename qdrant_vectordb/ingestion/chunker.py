# HierarchicalNodeParser implementation.
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
)

from retrieval.settings import CHUNK_SIZES
from utils.logger import logger
#from transformers import AutoTokenizer
import statistics
import time
from collections import defaultdict
# tokenizer = AutoTokenizer.from_pretrained(
#     "Qwen/Qwen2.5-7B"
# )
import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4.1")
from llama_index.core.schema import NodeRelationship
def create_chunks(documents):
    """
    Convert LlamaIndex Documents into hierarchical nodes.

    Parameters
    ----------
    documents : List[Document]

    Returns
    -------
    all_nodes : List[BaseNode]
        Parent + child nodes.
    """

    logger.info("Creating hierarchical nodes")
    start = time.perf_counter()
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=CHUNK_SIZES
    )

    all_nodes = parser.get_nodes_from_documents(
        documents
    )
    logger.info("=" * 80)
    logger.info("NODE RELATIONSHIPS")

    # Find one leaf node
    leaf_nodes = get_leaf_nodes(all_nodes)

    if leaf_nodes:
        node = leaf_nodes[0]

        logger.info(f"Leaf Node ID: {node.node_id}")
        logger.info(f"Text: {node.text[:100]}")

        logger.info("Relationships:")
        for rel, value in node.relationships.items():
            logger.info(f"{rel} -> {value}")

        current = node
        level = 0

        logger.info("Parent Chain")

        while NodeRelationship.PARENT in current.relationships:

            parent_info = current.relationships[NodeRelationship.PARENT]

            parent = next(
                (n for n in all_nodes if n.node_id == parent_info.node_id),
                None
            )

            if parent is None:
                break

            level += 1

            logger.info(
                f"Level {level}: {parent.node_id}"
            )

            logger.info(
                f"Length: {len(parent.text)}"
            )

            current = parent

    leaf_nodes = get_leaf_nodes(
        all_nodes
    )
    processing_time = time.perf_counter() - start
    chunk_token_counts = []

    chunk_word_counts = []

    chunk_character_counts = []
    for node in leaf_nodes:

        # tokens = len(
        #     tokenizer.encode(
        #         node.text,
        #         add_special_tokens=False
        #     )
        # )
        tokens = len(tokenizer.encode(node.text))

        words = len(
            node.text.split()
        )

        characters = len(
            node.text
        )

        ##########################################
        # Save in metadata
        ##########################################

        node.metadata["chunk_tokens"] = tokens

        node.metadata["chunk_words"] = words

        node.metadata["chunk_characters"] = characters

        ##########################################

        chunk_token_counts.append(tokens)

        chunk_word_counts.append(words)

        chunk_character_counts.append(characters)
    if chunk_token_counts:
        chunk_statistics = {

        "total_chunks": len(leaf_nodes),

        "average_chunk_tokens": round(
            statistics.mean(chunk_token_counts),
            2
        ),

        "minimum_chunk_tokens": min(chunk_token_counts),

        "maximum_chunk_tokens": max(chunk_token_counts),

        "median_chunk_tokens": statistics.median(
            chunk_token_counts
        ),

        "std_chunk_tokens": round(
            statistics.pstdev(
                chunk_token_counts
            ),
            2
        ),

        "average_words": round(
            statistics.mean(
                chunk_word_counts
            ),
            2
        ),

        "average_characters": round(
            statistics.mean(
                chunk_character_counts
            ),
            2
        )

    }
    else:

        chunk_statistics = {}
    logger.info("=" * 60)

    logger.info("Chunk Statistics")

    for key, value in chunk_statistics.items():

        logger.info(
            f"{key}: {value}"
        )

    logger.info("=" * 60)

    logger.info(
        f"Created {len(all_nodes)} total nodes"
    )

    logger.info(
        f"Leaf nodes: {len(leaf_nodes)}"
    )

    return all_nodes