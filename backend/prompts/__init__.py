"""
Prompt loaders — Load XML system prompts and guardrail prompts.
"""

from __future__ import annotations

import os
from pathlib import Path

# Directory containing prompt files
PROMPTS_DIR = Path(__file__).parent


def load_system_prompt() -> str:
    """Load the main XML system prompt."""
    prompt_path = PROMPTS_DIR / "system_prompt.xml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_guardrail_prompt() -> str:
    """Load the guardrail agent XML prompt."""
    prompt_path = PROMPTS_DIR / "guardrail_prompt.xml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_combined_prompt() -> str:
    """Load system + guardrail policy as a single agent prompt."""
    system = load_system_prompt()
    guardrail = load_guardrail_prompt()
    integrated = """
<integrated_guardrail>
Apply the guardrail policy below on every user message before answering or calling tools.

If the message must be blocked:
1. Do NOT call any tools.
2. Start your reply with exactly one line: [GUARDRAIL_REFUSAL:category_id]
   (use a category id from the guardrail policy, e.g. hate_speech, fake_scripture_injection).
3. Then give a brief, pastoral refusal aligned with Christian values.

If the message is allowed, respond normally and use rag_tool for biblical/theological questions.
</integrated_guardrail>
"""
    return f"{system}\n\n{integrated}\n\n{guardrail}"
