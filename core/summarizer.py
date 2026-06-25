"""
DocFlow — AI Summarizer
LLM-powered document summarization with multiple providers.
Generates: short/detailed summary, bullet points, key insights, entities, Q&A.
"""
from __future__ import annotations

from loguru import logger

from core.config import settings

SYSTEM_PROMPT = """You are a professional document analyst. 
Analyze the provided document text and return a structured JSON response with the following keys:
- summary: A concise 2-3 sentence summary
- detailed_summary: A comprehensive paragraph summary
- bullet_summary: An array of 5-10 key bullet points
- key_insights: An array of the most important insights
- topics: An array of main topics covered
- entities: An array of {name, type} objects for named entities (people, orgs, places, dates)
- questions: An array of 3-5 Q&A pairs {question, answer} the document answers

Respond ONLY with valid JSON, no markdown wrapping."""


class Summarizer:
    def __init__(self, provider: str | None = None):
        self.provider = provider or settings.summarization_provider

    async def summarize(self, text: str, max_chars: int = 12000) -> dict:
        """Generate structured AI summary of the provided text."""
        # Truncate to avoid token limits
        truncated = text[:max_chars]
        if len(text) > max_chars:
            truncated += "\n\n[...document truncated for summarization...]"

        try:
            if self.provider == "anthropic":
                return await self._anthropic(truncated)
            elif self.provider == "ollama":
                return await self._ollama(truncated)
            else:
                return await self._openai(truncated)
        except Exception as exc:
            logger.warning(f"[Summarizer] {self.provider} failed: {exc}")
            return self._empty_result()

    async def _openai(self, text: str) -> dict:
        import json
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Document:\n\n{text}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
        )
        return json.loads(response.choices[0].message.content)

    async def _anthropic(self, text: str) -> dict:
        import json
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        message = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Document:\n\n{text}"}],
        )
        content = message.content[0].text
        # Strip JSON fences if present
        content = content.strip().strip("```json").strip("```").strip()
        return json.loads(content)

    async def _ollama(self, text: str) -> dict:
        import json
        import httpx
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Document:\n\n{text}"},
            ],
            "stream": False,
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return json.loads(content)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "summary": "",
            "detailed_summary": "",
            "bullet_summary": [],
            "key_insights": [],
            "topics": [],
            "entities": [],
            "questions": [],
        }
