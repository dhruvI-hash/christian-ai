r"""
RAG pipeline smoke test (no LLM required).

Verifies that the vector DB is reachable and returns passages for a query.
Run from the backend directory:

    .\.venv\Scripts\python.exe tests\test_rag_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from config.logging_config import setup_logging  # noqa: E402
from config.settings import get_settings  # noqa: E402
from rag.pipeline import format_search_results_for_context, search  # noqa: E402


async def main() -> int:
    setup_logging(get_settings().log_level)
    settings = get_settings()

    queries = ["John 3:16", "love your neighbor", "the resurrection of Jesus"]
    failures = 0

    for vector_db in {settings.default_vector_db, "chroma"}:
        print(f"\n=== Vector DB: {vector_db} ===")
        for q in queries:
            try:
                results = await search(query=q, top_k=3, vector_db=vector_db)
            except Exception as e:  # noqa: BLE001
                print(f"  [ERROR] '{q}' -> {e}")
                failures += 1
                continue

            print(f"  '{q}' -> {len(results)} result(s)")
            if results:
                preview = format_search_results_for_context(results[:1])[:160]
                print(f"     {preview.replace(chr(10), ' ')}")
            else:
                print("     (no passages — knowledge base may be empty for this DB)")

    if failures:
        print(f"\nFAILURE: {failures} query/db combination(s) errored.")
        return 1

    print("\nSUCCESS: RAG pipeline responded for all queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
