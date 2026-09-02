import os
import time
from functools import lru_cache
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from utils.logger import logger
    from utils.metrics import record_metric, get_metrics
except ImportError:
    from qdrant_vectordb.utils.logger import logger
    from qdrant_vectordb.utils.metrics import record_metric, get_metrics

load_dotenv()


@lru_cache(maxsize=4)
def load_gemini(model: str = None, temperature: float = 0.1):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY not found in environment or .env file. "
            "Please set your Gemini API key in the .env file (e.g., GEMINI_API_KEY=AIzaSy...)."
        )

    model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    logger.info(f"Loading Gemini Model: {model_name}")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    )


class GeminiLLM:
    def __init__(
        self,
        model: str = None,
        temperature: float = 0.1,
    ):
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self.llm = load_gemini(self.model_name, temperature)

    def generate(
        self,
        question: str,
        context: str = "",
        history=None,
    ) -> str:
        history_text = ""
        if history:
            for message in history:
                history_text += f"{message.get('role', 'user')}: {message.get('content', '')}\n"

        if not context or not context.strip():
            prompt = f"""You are a helpful, knowledgeable, and precise AI assistant.

Instructions:
- Answer the user's question directly, clearly, and concisely.
- Use formatting (bullet points, code blocks) where helpful.

Conversation History:
{history_text}

Question:
{question}

Answer:
"""
        else:
            prompt = f"""You are an expert AI assistant.

Your job is to answer questions using ONLY the retrieved context.

Instructions:
- Write naturally, clearly, and authoritatively.
- Combine information from multiple retrieved chunks.
- Do NOT copy the context verbatim.
- Explain concepts in your own words.
- Use headings and bullet points where appropriate.
- If the retrieved context includes diagrams, flowcharts, or images (e.g. ![...](assets/...)), explicitly reference the visual diagram, explain its workflow, and cite its figure/image filename.
- Give examples if available.
- If the answer cannot be fully found in the context, clearly state what information is missing.
- End with a concise summary.

Conversation History:
{history_text}

Retrieved Context:
{context}

Question:
{question}

Answer:
"""

        logger.info("Generating Response via Gemini")
        logger.info("=" * 80)
        logger.info(f"Prompt Length : {len(prompt)}")
        logger.info(f"Context Length : {len(context)}")
        logger.info("=" * 80)

        start = time.perf_counter()
        logger.info("BEFORE INVOKE")

        try:
            response = self.llm.invoke(prompt)
            logger.info("AFTER INVOKE")
        except Exception as e:
            # Automatic fallback to gemini-3.6-flash if another model failed
            if self.model_name != "gemini-2.5-flash":
                logger.warning(f"Model {self.model_name} failed ({e}). Retrying with gemini-2.5-flash fallback.")
                fallback_llm = load_gemini("gemini-2.5-flash", self.temperature)
                response = fallback_llm.invoke(prompt)
            else:
                logger.exception(e)
                raise

        llm_time = time.perf_counter() - start
        record_metric("LLM Generation", round(llm_time, 3))
        logger.info(get_metrics())

        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "\n".join(text_parts).strip()
            return str(content)

        return str(response)
