from functools import lru_cache
from langchain_ollama import ChatOllama

from utils.logger import logger
from utils.metrics import record_metric, get_metrics
import time 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B"
)


@lru_cache(maxsize=4)
def load_qwen(model, temperature):

    logger.info(f"Loading {model}")

    return ChatOllama(
        model=model,
        temperature=temperature,
        num_predict=768,
        num_ctx=4096,
    )

class QwenLLM:

    def __init__(

        self,

        model="qwen2.5:7b",

        temperature=0.1

    ):

        self.llm = load_qwen(
            model,
            temperature
        )

    ##########################################################

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

        ##########################################################
        # Token Analytics
        ##########################################################

        prompt_tokens = len(
            tokenizer.encode(
                prompt,
                add_special_tokens=False
            )
        )

        context_tokens = len(
            tokenizer.encode(
                context,
                add_special_tokens=False
            )
        )

        question_tokens = len(
            tokenizer.encode(
                question,
                add_special_tokens=False
            )
        )

        history_tokens = len(
            tokenizer.encode(
                history_text,
                add_special_tokens=False
            )
        )

        MAX_CONTEXT = 128000

        remaining_tokens = MAX_CONTEXT - prompt_tokens

        usage_percent = round(
            (prompt_tokens / MAX_CONTEXT) * 100,
            2
        )

        ##########################################################

        record_metric("Prompt Tokens", prompt_tokens)
        record_metric("Context Tokens", context_tokens)
        record_metric("Question Tokens", question_tokens)
        record_metric("History Tokens", history_tokens)
        record_metric("Remaining Context", remaining_tokens)
        record_metric("Context Usage", usage_percent)

        logger.info("=" * 80)
        logger.info(f"Prompt Tokens      : {prompt_tokens}")
        logger.info(f"Context Tokens     : {context_tokens}")
        logger.info(f"Question Tokens    : {question_tokens}")
        logger.info(f"History Tokens     : {history_tokens}")
        logger.info(f"Remaining Context  : {remaining_tokens}")
        logger.info(f"Context Usage      : {usage_percent}%")
        logger.info("=" * 80)

        start = time.perf_counter()

        logger.info("BEFORE INVOKE")

        try:

            response = self.llm.invoke(prompt)
            response_tokens = len(
                tokenizer.encode(
                    response.content,
                    add_special_tokens=False
                )
            )

            record_metric("Response Tokens", response_tokens)

            logger.info("AFTER INVOKE")

        except Exception as e:

            logger.exception(e)

            raise

        llm_time = time.perf_counter() - start

        record_metric("LLM Generation", round(llm_time, 3))
        logger.info(get_metrics())

        if hasattr(response, "content"):
            record_metric("Total Tokens", prompt_tokens + response_tokens)
            return response.content

        return str(response)