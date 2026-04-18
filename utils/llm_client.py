# -*- coding: utf-8 -*-
"""
Generic LLM Client — supports any OpenAI-compatible API.
Pre-configured with Groq API credentials.
"""

import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Groq API Configuration (hardcoded) ──────────────────────────────────
GROQ_API_KEY = "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"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"


class LLMClient:
    """
    Abstraction layer for any OpenAI-compatible LLM API.
    Provider-agnostic: swap base_url to switch between OpenAI, Together, Groq, etc.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = None,
        model: str = None,
    ):
        # Use Groq by default, allow override
        self.api_key = api_key or GROQ_API_KEY
        self.base_url = (base_url or GROQ_BASE_URL).rstrip("/")
        self.model = model or GROQ_MODEL
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            self._client = None

    def update_key(self, api_key: str):
        """Hot-swap the API key (called when user updates from UI)."""
        self.api_key = api_key
        self._init_client()

    def chat(self, system_prompt: str, user_message: str, temperature: float = 0.3, max_retries: int = 3) -> str:
        """
        Send a chat request with exponential backoff retry on rate limits.
        Returns the assistant reply as a string.
        Raises ValueError on auth errors, RuntimeError on other failures.
        """
        if not self._client:
            raise RuntimeError("LLM client not initialised. Check that openai is installed.")
        if not self.api_key:
            raise ValueError("No API key provided. Please enter your API key in the sidebar.")

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=1024,
                )
                
                # Handle different response formats
                if isinstance(response, str):
                    return response.strip()
                elif hasattr(response, 'choices') and len(response.choices) > 0:
                    return response.choices[0].message.content.strip()
                elif isinstance(response, dict) and 'choices' in response:
                    return response['choices'][0]['message']['content'].strip()
                else:
                    raise ValueError(f"Unexpected response format from API: {type(response)}")

            except Exception as e:
                err = str(e)
                logger.error(f"Attempt {attempt + 1}: {err}", exc_info=True)
                
                # Auth errors — never retry
                if "401" in err or "invalid_api_key" in err.lower() or "authentication" in err.lower():
                    raise ValueError("Invalid API key. Please verify it matches your provider (OpenAI, OpenRouter, etc).")
                
                # Response format errors from wrong API endpoint
                if "choices" in err or "unexpected response" in err.lower():
                    raise ValueError(
                        "API endpoint mismatch: The base URL may not support OpenAI-compatible format. "
                        "Try: https://api.openai.com/v1 or https://api.openrouter.ai/v1"
                    )
                
                # Rate limit — retry with backoff
                if "429" in err or "rate" in err.lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s backoff
                        logger.warning(f"Rate limit hit. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError("Rate limit reached. Please wait a few minutes before trying again.")
                
                # Server errors — retry
                if "500" in err or "502" in err or "503" in err:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Server error. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError("API unavailable. Please try again later.")
                
                # Other errors
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"API error. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"API error: {err}")


# Singleton — replaced by app.py on each new API key entry
_default_client: Optional[LLMClient] = None


def get_client(api_key: Optional[str] = None, **kwargs) -> LLMClient:
    global _default_client
    if api_key or _default_client is None:
        _default_client = LLMClient(api_key=api_key, **kwargs)
    return _default_client
