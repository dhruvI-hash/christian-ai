r"""
End-to-end LangChain agent test.

Confirms the single agent actually calls rag_tool (and scripture_verify_tool)
for a scripture question like "Explain John 3:16 in context".

Requires a valid API key for the chosen provider in your environment / .env.
Run from the backend directory:

    # default model from settings (often gpt-4o)
    .\.venv\Scripts\python.exe tests\test_agent_e2e.py

    # or pick a model explicitly
    .\.venv\Scripts\python.exe tests\test_agent_e2e.py gemini-3.1-flash-lite
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from config.logging_config import setup_logging  # noqa: E402
from config.settings import get_settings  # noqa: E402
from custom_agents.christianity_agent import run_christianity_agent  # noqa: E402
from llm.langchain_client import detect_provider  # noqa: E402


def _has_key_for(provider: str, settings) -> bool:
    return {
        "openai": bool(settings.openai_api_key),
        "gemini": bool(settings.gemini_api_key),
        "openrouter": bool(settings.openrouter_api_key),
        "ollama": True,
    }.get(provider, False)


async def main() -> int:
    setup_logging(get_settings().log_level)
    settings = get_settings()

    model = sys.argv[1] if len(sys.argv) > 1 else settings.default_llm_model
    provider = detect_provider(model, settings.default_llm_provider)

    if not _has_key_for(provider, settings):
        print(
            f"SKIP: no API key configured for provider '{provider}' (model={model}). "
            "Set the key in .env and re-run."
        )
        return 0

    question = "Explain John 3:16 in context"
    print(f"Model: {model} | Provider: {provider}")
    print(f"Question: {question}\n")

    start = time.time()
    result = await run_christianity_agent(
        user_message=question,
        conversation_id=f"e2e_{int(start)}",
        denomination="Non-denominational",
        model=model,
        use_rag=True,
        vector_db=settings.default_vector_db,
    )
    elapsed = time.time() - start

    print("-" * 60)
    print(result.response)
    print("-" * 60)
    print(f"Elapsed       : {elapsed:.2f}s")
    print(f"RAG used       : {result.rag_used}")
    print(f"Citations      : {result.citations}")
    print(f"Guardrail      : {result.guardrail_triggered} ({result.guardrail_category})")
    print(f"Safety warnings: {result.safety_warnings}")

    if not result.rag_used:
        print("\nFAILURE: agent did NOT use the RAG tool for a scripture question.")
        return 1

    print("\nSUCCESS: agent grounded the answer via the RAG tool call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
