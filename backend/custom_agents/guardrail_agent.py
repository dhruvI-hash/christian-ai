"""
Guardrail Agent — Lightweight safety classifier using OpenAI Agents SDK.
Runs on EVERY user message BEFORE the main agent processes it.
Uses gpt-4o-mini for speed and cost efficiency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from agents import Agent, Runner

from prompts import load_guardrail_prompt
from moderation.safety_layer import REFUSAL_MESSAGES


@dataclass
class GuardrailDecision:
    """Decision from the guardrail agent."""
    blocked: bool
    reason: Optional[str] = None
    category: Optional[str] = None
    confidence: float = 0.0


# Categories recognized by the guardrail agent
GUARDRAIL_CATEGORIES = [
    "hate_speech",
    "fake_scripture_injection",
    "adversarial_rewrite",
    "heresy_promotion",
    "ideology_manipulation",
    "image_violation",
    "off_topic_harmful",
    "any_other_topic_except_bible"
]


# Create the guardrail agent
guardrail_agent = Agent(
    name="GuardrailClassifier",
    instructions=load_guardrail_prompt(),
    model="gpt-4o-mini",
)


async def run_guardrail(user_message: str, model: Optional[str] = None) -> GuardrailDecision:
    """
    Run the guardrail agent on a user message.

    Args:
        user_message: The user's input message.
        model: Optional dynamic model name.

    Returns:
        GuardrailDecision indicating whether the message should be blocked.
    """
    logger.info(f"Running guardrail classification on user message...")
    try:
        agent_to_run = guardrail_agent
        if model and model != "gpt-4o-mini":
            logger.info(f"Dynamically instantiating guardrail agent with model: {model}")
            agent_to_run = Agent(
                name="GuardrailClassifier",
                instructions=load_guardrail_prompt(),
                model=model,
            )

        result = await Runner.run(
            agent_to_run,
            input=f"Evaluate this user message:\n\n{user_message}",
        )

        # Parse the JSON response
        response_text = result.final_output.strip()

        # Try to extract JSON from the response
        try:
            # Handle case where response has markdown code blocks
            if "```" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    response_text = response_text[json_start:json_end]

            data = json.loads(response_text)
            blocked = data.get("blocked", False)
            reason = data.get("reason")
            category = data.get("category")
            confidence = data.get("confidence", 0.0)

            logger.info(f"Guardrail check complete. Blocked: {blocked}, Reason: {reason}, Category: {category}")

            return GuardrailDecision(
                blocked=blocked,
                reason=reason,
                category=category,
                confidence=confidence,
            )
        except json.JSONDecodeError:
            # If we can't parse the response, default to allowing
            logger.warning(f"Failed to parse guardrail JSON response: {response_text}. Defaulting to allow.")
            return GuardrailDecision(blocked=False, confidence=0.0)

    except Exception as e:
        # If guardrail fails, default to allowing (fail open)
        # but log the error
        logger.exception("Guardrail agent error occurred: ")
        return GuardrailDecision(blocked=False, reason=f"Guardrail error: {str(e)}")


def build_refusal_response(decision: GuardrailDecision) -> dict:
    """
    Build a graceful refusal response based on the guardrail decision.

    Args:
        decision: The guardrail decision with category and reason.

    Returns:
        Dict with response text and metadata.
    """
    category = decision.category or "general"

    # Get the appropriate refusal message
    refusal_map = {
        "hate_speech": REFUSAL_MESSAGES["hate_speech"],
        "fake_scripture_injection": REFUSAL_MESSAGES["fake_scripture"],
        "adversarial_rewrite": REFUSAL_MESSAGES["adversarial_rewrite"],
        "heresy_promotion": REFUSAL_MESSAGES["general"],
        "ideology_manipulation": REFUSAL_MESSAGES["adversarial_rewrite"],
        "image_violation": REFUSAL_MESSAGES["image_violation"],
        "off_topic_harmful": REFUSAL_MESSAGES["general"],
    }

    refusal_text = refusal_map.get(category, REFUSAL_MESSAGES["general"])

    return {
        "response": refusal_text,
        "guardrail_triggered": True,
        "guardrail_category": category,
        "guardrail_reason": decision.reason,
        "guardrail_confidence": decision.confidence,
        "citations": [],
        "rag_used": False,
        "image_url": None,
    }
