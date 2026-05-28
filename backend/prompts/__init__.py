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
