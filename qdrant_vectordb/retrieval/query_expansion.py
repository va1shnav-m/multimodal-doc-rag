# Metadata/KG query expansion.
import spacy
from utils.logger import logger
try:

    nlp = spacy.load(
        "en_core_web_sm"
    )

except Exception:

    raise RuntimeError(

        "Install spaCy model:\n"
        "python -m spacy download en_core_web_sm"

    )


##########################################################

def expand_query(

    query,

    metadata_list

):

    logger.info(

        "Expanding Query"

    )

    ######################################################

    expanded_terms = set()

    ######################################################
    # Original Query
    ######################################################

    expanded_terms.add(

        query

    )

    ######################################################
    # Named Entities
    ######################################################

    doc = nlp(

        query

    )

    for entity in doc.ents:

        expanded_terms.add(

            entity.text

        )

    ######################################################
    # Nouns
    ######################################################

    for token in doc:

        if token.pos_ in (

            "NOUN",

            "PROPN"

        ):

            expanded_terms.add(

                token.lemma_

            )

        ######################################################
        # Retrieved Metadata
        ######################################################

    for metadata in metadata_list:

        entities = metadata.get("entities", [])
        expanded_terms.update(entities)

        keywords = metadata.get("keywords", [])
        expanded_terms.update(keywords)

        topics = metadata.get("topics", [])
        expanded_terms.update(topics)

        knowledge_graph = metadata.get(
            "knowledge_graph",
            {}
        )

        triples = knowledge_graph.get(
            "triples",
            []
        )

        for triple in triples:

            expanded_terms.add(triple["subject"])
            expanded_terms.add(triple["relation"])
            expanded_terms.add(triple["object"])

        entity_graph = knowledge_graph.get(
            "entity_graph",
            {}
        )

        for neighbours in entity_graph.values():

            expanded_terms.update(neighbours)

    expanded_query = " ".join(
        sorted(expanded_terms)
    )

    logger.info(
        f"Expanded Query : {expanded_query}"
    )

    return expanded_query


        # for entity in list(

        #         expanded_terms

        #     ):

        #         if entity in entity_graph:

        #             expanded_terms.update(

        #                 entity_graph[entity]

        #         )

        # ##################################################

        # for triple in triples:

        #         expanded_terms.add(

        #             triple["subject"]

        #         )

        #         expanded_terms.add(

        #             triple["relation"]

        #         )

        #         expanded_terms.add(

        #             triple["object"]

        #         )

    ######################################################

    expanded_query = " ".join(

        sorted(

            expanded_terms

        )

    )

    logger.info(

        f"Expanded Query : {expanded_query}"

    )

    return expanded_query