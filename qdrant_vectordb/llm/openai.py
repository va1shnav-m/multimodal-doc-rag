from functools import lru_cache
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils.logger import logger
from utils.metrics import record_metric, get_metrics

load_dotenv()


@lru_cache(maxsize=4)
def load_openai(model, temperature):

    logger.info(f"Loading {model}")

    return ChatOpenAI(
        model=model,
        temperature=temperature
    )
class OpenAILLM:

    def __init__(

        self,

        model="gpt-4.1-mini",

        temperature=0.1

    ):

        self.llm = load_openai(
            model,
            temperature
        )
    def generate(

            self,

            question,

            context,

            history=None

        ):
            

            history_text = ""

            if history:

                for message in history:

                    history_text += (
                        f"{message['role']}: "
                        f"{message['content']}\n"
                    )

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

- Write naturally like ChatGPT.
- Combine information from multiple retrieved chunks.
- Do NOT copy the context verbatim.
- Explain concepts in your own words.
- Use headings.
- Use bullet points where appropriate.
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

            logger.info("Generating Response")

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

                logger.exception(e)

                raise

            llm_time = time.perf_counter() - start

            record_metric("LLM Generation", round(llm_time, 3))
            logger.info(get_metrics())

            if hasattr(response, "content"):

                return response.content

            return str(response)