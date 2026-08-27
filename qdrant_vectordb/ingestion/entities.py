import re
from utils.logger import logger

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy not available ({e}). Using regex entity extraction.")
            _nlp = "UNAVAILABLE"
    return None if _nlp == "UNAVAILABLE" else _nlp


def extract_entities(text):
    logger.info("Extracting Entities")

    nlp = get_nlp()
    if nlp:
        try:
            doc = nlp(text)
            entities = []
            for ent in doc.ents:
                if ent.text not in entities:
                    entities.append(ent.text)
            return entities
        except Exception:
            pass

    # Pure-Python fallback for environments where spaCy DLL is blocked
    pattern = r'\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*\b'
    matches = re.findall(pattern, text)
    entities = []
    for m in matches:
        if m not in entities and len(m) > 2:
            entities.append(m)
    return entities[:50]