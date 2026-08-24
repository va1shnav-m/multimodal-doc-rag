import yake
from utils.logger import logger

# Initialize YAKE keyword extractor
kw_extractor = yake.KeywordExtractor(
    lan="en",
    n=2,
    top=10
)


def extract_topics(
    text
):
    """
    Extract high-level topics from text using YAKE.
    """

    logger.info(
        "Extracting Topics"
    )

    keywords = kw_extractor.extract_keywords(
        text
    )

    topics = []

    for keyword, score in keywords:

        if keyword not in topics:

            topics.append(
                keyword
            )

    return topics