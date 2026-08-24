from keybert import KeyBERT
from utils.logger import logger

_model = None


def get_model():
    global _model
    if _model is None:
        _model = KeyBERT()
    return _model


def extract_keywords(
    text,
    top_n=5
):
    logger.info(
        "Extracting Keywords"
    )

    model = get_model()
    keywords = model.extract_keywords(

        text,

        keyphrase_ngram_range=(1, 2),

        stop_words="english",

        top_n=top_n

    )

    return [

        keyword

        for keyword, score in keywords

    ]