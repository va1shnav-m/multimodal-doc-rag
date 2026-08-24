import re
from langchain_core.output_parsers import PydanticOutputParser

from retrieval.router_schema import (
    RouteDecision,
    Pipeline,
)

from llm.qwen import load_qwen
# from llm.openai import load_openai
from utils.logger import logger


CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|greetings|howdy|hola|bonjour)\b",
    r"^(good morning|good afternoon|good evening|good day)\b",
    r"^(thank you|thanks|thx|appreciate it|cheers)\b",
    r"^(who are you|what is your name|what can you do|how can you help)\b",
    r"^(bye|goodbye|see you|cya|exit|quit)\b",
    r"^(how are you|how's it going|what's up)\b",
]

CHITCHAT_REGEX = re.compile("|".join(CHITCHAT_PATTERNS), re.IGNORECASE)


ROUTER_PROMPT = """
You are an intelligent routing agent for an enterprise AI system.

Your task is to decide whether to answer DIRECTLY (without document retrieval) or retrieve information from internal indexed documents (FAST or DEEP).

----------------------------------------
DIRECT (No Retrieval)
----------------------------------------
Use this when the question:
- is general conversation, greetings, small talk, or polite phrases
- asks general programming/coding questions (e.g., "how to sort a list in Python")
- asks basic math or general world knowledge questions
- does NOT require or reference private/internal uploaded documents

----------------------------------------
FAST Retrieval
----------------------------------------
Use this when the question requires private/internal documents for:
- factual lookups or specific definitions
- questions about a single concept or section
- direct lookup in uploaded documents

----------------------------------------
DEEP Retrieval
----------------------------------------
Use this when the question requires private/internal documents for:
- comparisons across multiple documents or sections
- complex reasoning or synthesis of multiple ideas
- analytical, architectural, or multi-step questions

Return ONLY a valid JSON object.

{format_instructions}

Question:
{query}
"""


class RouteSelector:

    def __init__(self):

        self.llm = load_qwen(
            model="qwen2.5:7b",
            temperature=0
        )

        self.parser = PydanticOutputParser(
            pydantic_object=RouteDecision
        )

    ##########################################################

    def select(
        self,
        query: str
    ) -> RouteDecision:
        cleaned_query = query.strip()

        # Fast rule-based check for common chitchat and greetings
        if len(cleaned_query) < 80 and CHITCHAT_REGEX.search(cleaned_query):
            logger.info("=" * 60)
            logger.info("Chitchat detected -> Routing to DIRECT")
            logger.info("=" * 60)
            return RouteDecision(
                pipeline=Pipeline.DIRECT,
                confidence=1.0,
                reason="General conversational greeting or small-talk detected."
            )

        prompt = ROUTER_PROMPT.format(
            query=query,
            format_instructions=self.parser.get_format_instructions()
        )

        logger.info("=" * 60)
        logger.info("Selecting Retrieval Pipeline")
        logger.info("=" * 60)

        response = self.llm.invoke(prompt)

        text = response.content.strip()

        # Remove markdown fences if the model returns them
        if text.startswith("```"):
            text = (
                text.replace("```json", "")
                    .replace("```", "")
                    .strip()
            )

        try:

            decision = self.parser.parse(text)

        except Exception as e:

            logger.exception("Failed to parse router output.")
            logger.exception(e)

            decision = RouteDecision(
                pipeline=Pipeline.DEEP,
                confidence=0.0,
                reason="Parser failure. Falling back to Deep Retrieval."
            )

        logger.info(f"Pipeline   : {decision.pipeline}")
        logger.info(f"Confidence : {decision.confidence}")
        logger.info(f"Reason     : {decision.reason}")
        logger.info("=" * 60)

        return decision