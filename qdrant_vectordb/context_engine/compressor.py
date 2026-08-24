"""
Context Compression Engine using LLMLingua-2

This module is responsible for compressing the retrieved context
before it is passed to the LLM.

Author: Manasi Sabnis
"""

import time
from typing import Dict

from llmlingua import PromptCompressor
from retrieval.settings import (
        COMPRESSION_RATE,
        LLMLINGUA_MODEL,
        ENABLE_CONTEXT_COMPRESSION
)

class ContextCompressor:
    """
    Wrapper around Microsoft's LLMLingua-2 prompt compressor.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device: str = "cpu",
    ):
        self.compression_rate = COMPRESSION_RATE

        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device,
        )

    def compress(self, context: str) -> Dict:
        """
        Compress retrieved context.

        Args:
            context (str):
                Complete retrieved context.

        Returns:
            dict:
                {
                    "compressed_context": str,
                    "original_tokens": int,
                    "compressed_tokens": int,
                    "compression_ratio": str,
                    "compression_rate": str,
                    "compression_time": float
                }
        """

        start = time.perf_counter()

        result = self.compressor.compress_prompt(
            context,
            rate=self.compression_rate,
        )

        end = time.perf_counter()

        return {
            "compressed_context": result["compressed_prompt"],
            "original_tokens": result["origin_tokens"],
            "compressed_tokens": result["compressed_tokens"],
            "compression_ratio": result["ratio"],
            "compression_rate": result["rate"],
            "compression_time": round(end - start, 3),
        }