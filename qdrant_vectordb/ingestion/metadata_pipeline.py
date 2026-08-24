from functools import lru_cache
from llama_index.core import Document
from utils.logger import logger
import json
import tiktoken


@lru_cache(maxsize=1)
def load_tokenizer():
    return tiktoken.encoding_for_model("gpt-4.1")


tokenizer = load_tokenizer()
#tokenizer = load_tokenizer()
from ingestion.entities import extract_entities
from ingestion.keywords import extract_keywords
from ingestion.topics import extract_topics
from ingestion.knowledge_graph import (
    build_knowledge_graph
)


class MetadataPipeline:

    def build_document(
        self,
        parsed_document,
        filename=""
    ):

        logger.info(
            "Building Document"
        )

        metadata = dict(
            parsed_document["document_metadata"]
        )

        metadata["filename"] = filename

        return Document(
            text=parsed_document["text"],
            metadata=metadata
        )

    ######################################################

    def process(
    self,
    document
    ):

        logger.info(
            "Extracting Metadata"
        )

        entities = extract_entities(
                document.text
            )

        document.metadata["entities"] = entities[:5]

        keywords = extract_keywords(
            document.text
        )

        document.metadata["keywords"] = keywords[:5]

        topics = extract_topics(
            document.text
        )

        document.metadata["topics"] = topics[:5]
        knowledge_graph = build_knowledge_graph(
            document.text
        )

        knowledge_graph["triples"] = knowledge_graph.get(
            "triples",
            []
        )[:3]

        document.metadata["knowledge_graph"] = knowledge_graph
        ######################################################
        # Metadata Analytics
        ######################################################

        chunk_tokens = document.metadata.get(
            "chunk_tokens",
            len(
                # tokenizer.encode(
                #     document.text,
                #     add_special_tokens=False
                # )
                tokenizer.encode(document.text)
            )
        )

        metadata_text = json.dumps(
            document.metadata,
            default=str,
            ensure_ascii=False
        )

        metadata_tokens = len(
            # tokenizer.encode(
            #     metadata_text,
            #     add_special_tokens=False
            # )
            tokenizer.encode(metadata_text)
        )

        metadata_characters = len(
            metadata_text
        )

        metadata_fields = len(
            document.metadata
        )

        total_tokens = (
            chunk_tokens +
            metadata_tokens
        )

        metadata_ratio = round(

            (metadata_tokens / chunk_tokens) * 100,

            2

        ) if chunk_tokens else 0
        document.metadata["metadata_tokens"] = metadata_tokens

        document.metadata["metadata_characters"] = metadata_characters

        document.metadata["metadata_fields"] = metadata_fields

        document.metadata["total_tokens"] = total_tokens

        document.metadata["metadata_ratio"] = metadata_ratio

        document.metadata["metadata_size_kb"] = round(
            metadata_characters / 1024,
            2
        )

        return document