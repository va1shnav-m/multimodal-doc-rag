from utils.logger import logger

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy not available for Knowledge Graph ({e}). Skipping KG triples.")
            _nlp = "UNAVAILABLE"
    return None if _nlp == "UNAVAILABLE" else _nlp


# ==========================================================
# Dependency labels to ignore
# ==========================================================

IGNORE_DEPS = {

    "punct",

    "det",

    "cc",

    "mark",

    "case"

}


# ==========================================================
# Extract subject-verb-object triples
# ==========================================================

def extract_knowledge_graph(text):

    logger.info(
        "Extracting Knowledge Graph"
    )

    nlp = get_nlp()
    if not nlp:
        return []

    try:
        doc = nlp(text)
    except Exception:
        return []

    triples = []

    seen = set()

    ######################################################

    for sent in doc.sents:

        subject = None

        relation = None

        obj = None

        ##################################################

        for token in sent:

            if token.dep_ in ("nsubj", "nsubjpass"):

                subject = token.text

            elif token.pos_ == "VERB":

                relation = token.lemma_

            elif token.dep_ in (

                "dobj",

                "pobj",

                "attr",

                "dative",

                "oprd"

            ):

                obj = token.text

        ##################################################

        if (

            subject

            and

            relation

            and

            obj

        ):

            triple = (

                subject.strip(),

                relation.strip(),

                obj.strip()

            )

            if triple not in seen:

                triples.append(

                    {

                        "subject": triple[0],

                        "relation": triple[1],

                        "object": triple[2]

                    }

                )

                seen.add(triple)

    ######################################################

    logger.info(

        f"{len(triples)} triples extracted"

    )

    return triples


# ==========================================================
# Entity Graph
# ==========================================================

def build_entity_graph(text):

    nlp = get_nlp()
    if not nlp:
        return {}

    try:
        doc = nlp(text)
    except Exception:
        return {}

    graph = {}

    ######################################################

    entities = [

        ent.text

        for ent in doc.ents

    ]

    ######################################################

    for entity in entities:

        graph[entity] = []

    ######################################################

    for sentence in doc.sents:

        sentence_entities = [

            ent.text

            for ent in sentence.ents

        ]

        ##################################################

        for entity in sentence_entities:

            neighbours = [

                e

                for e in sentence_entities

                if e != entity

            ]

            graph.setdefault(

                entity,

                []

            )

            graph[entity].extend(

                neighbours

            )

    ######################################################

    for entity in graph:

        graph[entity] = list(

            set(graph[entity])

        )

    return graph


# ==========================================================
# Combined API
# ==========================================================

def build_knowledge_graph(text):

    return {

        "triples": extract_knowledge_graph(

            text

        ),

        "entity_graph": build_entity_graph(

            text

        )

    }