import os
import re
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser

from retrieval.router_schema import (
    RouteDecision,
    Pipeline,
)
from utils.logger import logger

load_dotenv()

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

    def __init__(self, llm_choice: str = None):
        self.parser = PydanticOutputParser(
            pydantic_object=RouteDecision
        )
        self.llm = None

        try:
            choice = (llm_choice or "").lower()
            if choice == "gemini" or (not choice and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))):
                from llm.gemini import load_gemini
                self.llm = load_gemini(model="gemini-2.5-flash", temperature=0)
            elif choice == "openai" or (not choice and os.getenv("OPENAI_API_KEY")):
                from llm.openai import load_openai
                self.llm = load_openai(model="gpt-4o-mini", temperature=0)
            else:
                from llm.qwen import load_qwen
                self.llm = load_qwen(model="qwen2.5:7b", temperature=0)
        except Exception as e:
            logger.warning(f"Could not initialize RouteSelector LLM ({e}). Will default to DEEP retrieval.")
            self.llm = None

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

        if not self.llm:
            return RouteDecision(
                pipeline=Pipeline.DEEP,
                confidence=1.0,
                reason="Defaulting to Deep Hybrid Retrieval."
            )

        prompt = ROUTER_PROMPT.format(
            query=query,
            format_instructions=self.parser.get_format_instructions()
        )

        logger.info("=" * 60)
        logger.info("Selecting Retrieval Pipeline")
        logger.info("=" * 60)

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                text = response.content
                if isinstance(text, list):
                    text = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in text])
                text = str(text).strip()
            else:
                text = str(response).strip()

            # Remove markdown fences if the model returns them
            if text.startswith("`"):
                text = (
                    text.replace("`json", "")
                        .replace("`", "")
                        .strip()
                )

            decision = self.parser.parse(text)
        except Exception as e:
            logger.warning(f"Failed to route with LLM ({e}). Falling back to Deep Retrieval.")
            decision = RouteDecision(
                pipeline=Pipeline.DEEP,
                confidence=0.0,
                reason="Router fallback to Deep Retrieval."
            )

        logger.info(f"Pipeline   : {decision.pipeline}")
        logger.info(f"Confidence : {decision.confidence}")
        logger.info(f"Reason     : {decision.reason}")
        logger.info("=" * 60)

        return decision
