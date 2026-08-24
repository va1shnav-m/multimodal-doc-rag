from collections import defaultdict
# from transformers import AutoTokenizer

# # Same tokenizer used throughout the project
# tokenizer = AutoTokenizer.from_pretrained(
#     "Qwen/Qwen2.5-7B"
# )
import tiktoken
tokenizer = tiktoken.encoding_for_model("gpt-4.1")
def build_document_analytics(
    leaf_nodes,
    processing_time
):
    """
    Build document-level analytics from leaf nodes.

    Parameters
    ----------
    leaf_nodes : List[BaseNode]
        Leaf chunks created by HierarchicalNodeParser.

    processing_time : float
        Time taken to create chunks.

    Returns
    -------
    dict
        Analytics grouped by document.
    """

    document_stats = defaultdict(
        lambda: {

            "processing_time": processing_time,

            "chunks_created": 0,

            "total_chunk_tokens": 0,

            "largest_chunk": 0,

            "smallest_chunk": float("inf"),

            "pages": set(),

            "chunks": []

        }
    )

    #######################################################
    # Process every chunk
    #######################################################

    for i, node in enumerate(leaf_nodes, start=1):

        filename = node.metadata.get(
            "filename",
            "Unknown"
        )

        page = node.metadata.get(
            "page",
            node.metadata.get(
                "page_number",
                "Unknown"
            )
        )

        chunk_tokens = len(

            # tokenizer.encode(

            #     node.text,

            #     add_special_tokens=False

            # )
            tokenizer.encode(node.text)

        )

        ###################################################
        # Save token info into metadata
        ###################################################

        node.metadata["chunk_tokens"] = chunk_tokens

        ###################################################

        stats = document_stats[filename]

        stats["chunks_created"] += 1

        stats["total_chunk_tokens"] += chunk_tokens

        stats["largest_chunk"] = max(

            stats["largest_chunk"],

            chunk_tokens

        )

        stats["smallest_chunk"] = min(

            stats["smallest_chunk"],

            chunk_tokens

        )

        stats["pages"].add(page)

        ###################################################
        # Save chunk information
        ###################################################

        stats["chunks"].append({

            "chunk_number": i,

            "page": page,

            "tokens": chunk_tokens,

            "preview": node.text[:250]

        })

    #######################################################
    # Compute averages
    #######################################################

    for filename in document_stats:

        stats = document_stats[filename]

        chunks = stats["chunks_created"]

        stats["average_chunk_tokens"] = round(

            stats["total_chunk_tokens"] /

            max(chunks, 1),

            2

        )

        stats["pages"] = sorted(

            list(stats["pages"])

        )

    return dict(document_stats)