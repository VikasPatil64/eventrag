"""
Gemini LLM provider using the google-genai SDK.

Called from main.py when LLM_PROVIDER=gemini, inside a ctx.step.run() block
so Inngest still provides step-level retry and memoisation.
"""

from app.config.settings import get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        settings = get_settings()
        _client = genai.Client(api_key=settings.gemini_api_key)
        logger.debug("Gemini client initialised (model=%s).", settings.gemini_llm_model)
    return _client


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """
    Call Gemini and return the answer string.

    Raises on API errors so the Inngest step surfaces them cleanly.
    """
    from google.genai import types

    settings = get_settings()
    client = _get_client()

    logger.info("Calling Gemini %s for answer generation.", settings.gemini_llm_model)
    try:
        response = client.models.generate_content(
            model=settings.gemini_llm_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
    except Exception as exc:
        logger.error("Gemini inference failed: %s", exc)
        raise

    answer = response.text.strip()
    logger.info("Gemini answer received (%d chars).", len(answer))
    return answer
