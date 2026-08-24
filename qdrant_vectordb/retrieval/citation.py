from utils.logger import logger
from collections import defaultdict

def get_documents_used(reranked_results):

    logger.info("Collecting documents used")

    documents = defaultdict(
        lambda: {
            "chunks": 0,
            "pages": set()
        }
    )

    for doc, score in reranked_results:

        metadata = doc.get("metadata", {})

        filename = metadata.get(
            "filename",
            "Unknown"
        )

        page = metadata.get(
            "page",
            metadata.get(
                "page_number",
                "Unknown"
            )
        )

        documents[filename]["chunks"] += 1
        documents[filename]["pages"].add(page)

    results = []

    for filename, info in documents.items():

        results.append({

            "filename": filename,

            "chunks": info["chunks"],

            "pages": sorted(info["pages"])

        })

    return results
def get_supporting_evidence(

    reranked_results

):

    logger.info(

        "Collecting supporting evidence"

    )

    evidence = []

    for doc, score in reranked_results:

        metadata = doc.get(

            "metadata",

            {}

        )

        evidence.append(

            {

                "filename": metadata.get(

                    "filename",

                    "Unknown"

                ),

                "page": metadata.get(

                    "page_number",

                    "Unknown"

                ),

                "score": float(score),

                "text": doc["text"],

                "metadata": metadata

            }

        )

    return evidence