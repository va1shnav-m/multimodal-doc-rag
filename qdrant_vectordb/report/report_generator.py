from pathlib import Path
from datetime import datetime
import json
import tiktoken
from retrieval.settings import STORAGE_ROOT

tokenizer = tiktoken.encoding_for_model("gpt-4.1")
class ReportGenerator:

    def __init__(self):

        self.lines = []

    #########################################################

    def line(self):

        self.lines.append("=" * 100)

    #########################################################

    def add(self, text=""):

        self.lines.append(str(text))

    #########################################################

    def section(self, title):

        self.line()

        self.add(title.upper())

        self.line()

    #########################################################

    def pipeline_summary(

        self,

        embedding_model,

        embedding_dimension,

        vector_db,

        chunk_strategy,

        chunk_size,

        overlap,

        llm

    ):

        self.section("Pipeline Configuration")

        self.add(f"Embedding Model      : {embedding_model}")

        self.add(f"Embedding Dimension  : {embedding_dimension}")

        self.add(f"Vector Database      : {vector_db}")

        self.add(f"Chunk Strategy       : {chunk_strategy}")

        self.add(f"Chunk Size           : {chunk_size}")

        self.add(f"Chunk Overlap        : {overlap}")

        self.add(f"LLM                  : {llm}")

        self.add()

    #########################################################

    def document_summary(

        self,

        filename,

        pages,

        processing_time,

        chunks,

        total_tokens,

        average_tokens,

        largest_chunk,

        smallest_chunk

    ):

        self.section(f"Document Summary : {filename}")

        self.add(f"Pages                 : {pages}")

        self.add(f"Processing Time       : {processing_time:.2f} s")

        self.add(f"Chunks Created        : {chunks}")

        self.add(f"Total Chunk Tokens    : {total_tokens}")

        self.add(f"Average Chunk Tokens  : {average_tokens}")

        self.add(f"Largest Chunk         : {largest_chunk}")

        self.add(f"Smallest Chunk        : {smallest_chunk}")

        self.add()

    #########################################################

    def chunk_summary(

        self,

        chunk_number,

        total_chunks,

        filename,

        page,

        characters,

        chunk_tokens,

        metadata_tokens,

        entities,

        keywords,

        topics,

        knowledge_graph,

        text,

        chunk_target=512

    ):

        utilization = (

            chunk_tokens /

            chunk_target

        ) * 100

        self.section(

            f"Chunk {chunk_number}/{total_chunks}"

        )

        self.add(f"Document             : {filename}")

        self.add(f"Page                 : {page}")

        self.add(f"Characters           : {characters}")

        self.add(f"Tokens               : {chunk_tokens}")

        self.add(f"Chunk Target         : {chunk_target}")

        self.add(f"Utilization          : {utilization:.2f}%")

        self.add(f"Metadata Tokens      : {metadata_tokens}")

        self.add(f"Entities ({len(entities)})")

        for entity in entities:
            self.add(f"   - {entity}")

        self.add("")

        self.add(f"Keywords             : {len(keywords)}")

        self.add(f"Topics               : {len(topics)}")

        triples = knowledge_graph.get(
            "triples",
            []
        )

        self.add(
            f"KG Triples : {len(triples)}"
        )

        for triple in triples:

            self.add(

                f"   - {triple['subject']} "
                f"--{triple['relation']}--> "
                f"{triple['object']}"

            )

        self.add("")

        self.add()

        self.add("Chunk Text")

        self.add("-" * 80)

        self.add(text)

        self.add()

    #########################################################

    def save(self):
        reports_dir = STORAGE_ROOT / "reports"
        reports_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            reports_dir / f"inspection_report_{timestamp}.txt"
        )

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(
                "\n".join(self.lines)
            )

        return str(filename)
    def qdrant_summary(
        self,
        points,
        tokenizer
    ):

        self.add("=" * 80)
        self.add("QDRANT VECTOR DATABASE")
        self.add("=" * 80)
        self.add("")

        self.add(f"Vectors Stored : {len(points)}")
        self.add("")

        for i, point in enumerate(points, start=1):

            payload = point.payload or {}

            vector = point.vector

            self.add("-" * 80)
            self.add(f"VECTOR {i}")
            self.add("-" * 80)

            self.add(f"ID : {point.id}")

            self.add(f"Embedding Dimension : {len(vector)}")
            self.add("Embedding Preview")

            preview = vector[:10]

            for value in preview:
                self.add(value)

            self.add("")

            payload_text = json.dumps(payload, default=str)

            payload_tokens = len(
                # tokenizer.encode(
                #     payload_text,
                #     add_special_tokens=False
                # )
                    tokenizer.encode(payload_text)
                )
            

            self.add(f"Payload Tokens : {payload_tokens}")
            self.add("")

            self.add("Payload")

            for key, value in payload.items():

                value_tokens = len(
                    tokenizer.encode(
                        json.dumps(value, default=str)
                    )
                )

                self.add(f"{key}")
                self.add(f"Tokens : {value_tokens}")
                self.add(str(value))
                self.add("")