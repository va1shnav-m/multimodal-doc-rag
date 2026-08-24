import spacy
from utils.logger import logger

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text):
    logger.info("Extracting Entities")

    nlp = get_nlp()
    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        if ent.text not in entities:

            entities.append(ent.text)

    return entities