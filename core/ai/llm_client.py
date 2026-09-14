"""
core.ai.llm_client
==================
Configurable multi-provider LLM client with strict security, timeouts,
retries, token budgeting (<1500 tokens), and deterministic offline fallback.
Never exposes API keys or credentials to the client.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Environment configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "auto").lower()
AI_MODEL = os.getenv("AI_MODEL", "")
AI_API_KEY = os.getenv("AI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

REQUEST_TIMEOUT_SEC = 15
MAX_RETRIES = 2
MAX_CONTEXT_TOKENS = 1500


class LLMClient:
    """Enterprise LLM Client supporting OpenAI, Gemini, and resilient offline analytical execution."""

    @classmethod
    def get_configured_provider(cls) -> str:
        """Determines active AI provider based on environment and available keys."""
        if AI_PROVIDER != "auto":
            return AI_PROVIDER
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "offline"

    @classmethod
    def call_llm(
        cls,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Executes server-side LLM call with retry, timeout, and fallback.
        Returns: {"success": bool, "text": str, "provider": str, "model": str, "error": Optional[str]}
        """
        provider = cls.get_configured_provider()

        if provider == "offline" or not AI_API_KEY:
            return {
                "success": False,
                "text": "",
                "provider": "offline",
                "model": "deterministic-engine",
                "error": "No external AI API key configured. Using verified deterministic engine."
            }

        # Truncate prompt context if it exceeds safety limit
        safe_prompt = prompt[:MAX_CONTEXT_TOKENS * 4]

        # 1. Gemini Client (google.genai)
        if provider == "gemini":
            try:
                from google import genai
                from google.genai import types

                key = os.getenv("GEMINI_API_KEY") or AI_API_KEY
                client = genai.Client(api_key=key)
                model_name = AI_MODEL or "gemini-2.5-flash"

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_instruction or "You are Quantum AI, a senior expert data analyst. Explain clearly in simple terms.",
                )

                for attempt in range(MAX_RETRIES + 1):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=safe_prompt,
                            config=config,
                        )
                        if response and response.text:
                            return {
                                "success": True,
                                "text": response.text.strip(),
                                "provider": "gemini",
                                "model": model_name,
                                "error": None
                            }
                    except Exception as e:
                        if attempt == MAX_RETRIES:
                            raise e
                        time.sleep(1.0 * (attempt + 1))

            except Exception as e:
                logger.warning(f"Gemini API call failed: {e}")
                return {
                    "success": False,
                    "text": "",
                    "provider": "gemini",
                    "model": AI_MODEL or "gemini-2.5-flash",
                    "error": f"Gemini API error: {str(e)}"
                }

        # 2. OpenAI Client (openai>=1.30.0)
        if provider == "openai":
            try:
                from openai import OpenAI

                key = os.getenv("OPENAI_API_KEY") or AI_API_KEY
                client = OpenAI(api_key=key, timeout=REQUEST_TIMEOUT_SEC)
                model_name = AI_MODEL or "gpt-4o-mini"

                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": safe_prompt})

                for attempt in range(MAX_RETRIES + 1):
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        if response.choices and response.choices[0].message.content:
                            return {
                                "success": True,
                                "text": response.choices[0].message.content.strip(),
                                "provider": "openai",
                                "model": model_name,
                                "error": None
                            }
                    except Exception as e:
                        if attempt == MAX_RETRIES:
                            raise e
                        time.sleep(1.0 * (attempt + 1))

            except Exception as e:
                logger.warning(f"OpenAI API call failed: {e}")
                return {
                    "success": False,
                    "text": "",
                    "provider": "openai",
                    "model": AI_MODEL or "gpt-4o-mini",
                    "error": f"OpenAI API error: {str(e)}"
                }

        return {
            "success": False,
            "text": "",
            "provider": provider,
            "model": "unknown",
            "error": f"Unsupported or unconfigured AI provider: {provider}"
        }
