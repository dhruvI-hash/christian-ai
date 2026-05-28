"""
Christianity Agent — Main OpenAI Agents SDK agent for the AI assistant.
Orchestrates the full conversation flow with guardrails, RAG, and safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from agents import Agent, Runner

from prompts import load_system_prompt
from tools.rag_tool import rag_tool, set_vector_db_context
from tools.scripture_verify_tool import scripture_verify_tool
from tools.image_gen_tool import image_gen_tool
from custom_agents.guardrail_agent import run_guardrail, build_refusal_response, GuardrailDecision
from memory.conversation_store import ConversationStore, ConversationTurn
from moderation.safety_layer import post_process_safety
from config.settings import get_settings


@dataclass
class AgentResponse:
    """Structured response from the Christianity agent."""
    response: str
    citations: list[dict] = field(default_factory=list)
    guardrail_triggered: bool = False
    guardrail_category: Optional[str] = None
    rag_used: bool = False
    image_url: Optional[str] = None
    model_used: str = "gpt-4o"
    conversation_id: str = ""
    safety_warnings: list[str] = field(default_factory=list)


# Create the main Christianity agent
christianity_agent = Agent(
    name="ChristianityAssistant",
    instructions=load_system_prompt(),
    tools=[rag_tool, scripture_verify_tool, image_gen_tool],
    model="gpt-4o",
)

# Conversation store singleton
_store = ConversationStore()


async def run_christianity_agent(
    user_message: str,
    conversation_id: str,
    denomination: str = "Non-denominational",
    model: str = "gpt-4o",
    use_rag: bool = True,
    guardrail_model: Optional[str] = None,
    vector_db: Optional[str] = None,
) -> AgentResponse:
    """
    Run the Christianity agent with full safety pipeline.

    Flow:
    1. Run guardrail agent on user message
    2. If blocked → return graceful refusal
    3. Build context from conversation history
    4. Run main agent with tools
    5. Post-process safety check on output
    6. Store conversation turn
    7. Return structured response

    Args:
        user_message: The user's input message.
        conversation_id: Unique conversation ID.
        denomination: User's denomination for context.
        model: LLM model to use.
        use_rag: Whether to enable RAG search.
        guardrail_model: Optional guardrail model.
        vector_db: Vector DB to use ("qdrant" or "chroma"). Defaults to settings.

    Returns:
        AgentResponse with the agent's reply and metadata.
    """
    settings = get_settings()
    # Override vector DB if specified
    effective_vector_db = vector_db or settings.default_vector_db
    logger.info(f"Using vector DB: {effective_vector_db} (requested: {vector_db})")

    # Step 1: Run guardrail agent
    if settings.enable_guardrail:
        guardrail_result = await run_guardrail(user_message, model=guardrail_model)
        if guardrail_result.blocked:
            refusal = build_refusal_response(guardrail_result)

            # Store the blocked turn
            _store.upsert_turn(conversation_id, ConversationTurn(
                role="user",
                content=user_message,
                guardrail_triggered=True,
                denomination_context=denomination,
            ))
            _store.upsert_turn(conversation_id, ConversationTurn(
                role="assistant",
                content=refusal["response"],
                guardrail_triggered=True,
            ))

            return AgentResponse(
                response=refusal["response"],
                guardrail_triggered=True,
                guardrail_category=guardrail_result.category,
                model_used=model,
                conversation_id=conversation_id,
            )

    # Step 2: Get conversation history
    conversation = _store.get_conversation(conversation_id)
    if conversation and conversation.denomination != denomination:
        conversation.denomination = denomination

    # Build context-enriched input
    context_prefix = f"[User denomination: {denomination}]\n[Vector DB: {effective_vector_db}]\n\n"
    enriched_message = context_prefix + user_message

    # Build conversation history for the agent
    input_messages = []
    if conversation:
        for turn in conversation.turns[-20:]:  # Last 20 turns for context
            input_messages.append({
                "role": turn.role,
                "content": turn.content,
            })

    input_messages.append({
        "role": "user",
        "content": enriched_message,
    })

    # Step 3: Run the main agent
    try:
        # Update agent model if different from default
        agent_to_run = christianity_agent
        if model != "gpt-4o":
            agent_to_run = Agent(
                name="ChristianityAssistant",
                instructions=load_system_prompt(),
                tools=[rag_tool, scripture_verify_tool, image_gen_tool],
                model=model,
            )

        # Set the vector DB context for the rag_tool
        set_vector_db_context(effective_vector_db)

        result = await Runner.run(
            agent_to_run,
            input=input_messages,
        )

        agent_response_text = result.final_output

    except Exception as e:
        logger.exception("Error running main agent: ")
        agent_response_text = (
            "I apologize, but I encountered an issue processing your request. "
            f"Please try again. (Error: {str(e)[:100]})"
        )
        return AgentResponse(
            response=agent_response_text,
            citations=[],
            guardrail_triggered=False,
            rag_used=False,
            image_url=None,
            model_used=model,
            conversation_id=conversation_id,
            safety_warnings=[],
        )

    # Step 4: Post-process safety check
    safety_warnings = []
    if settings.enable_post_safety:
        # Extract tool calls info for safety verification
        tool_calls_info = []
        if hasattr(result, 'raw_responses'):
            for raw in result.raw_responses:
                if hasattr(raw, 'output') and isinstance(raw.output, list):
                    for item in raw.output:
                        if hasattr(item, 'type') and item.type == 'tool_call':
                            tool_calls_info.append({
                                "name": getattr(item, 'name', ''),
                                "arguments": getattr(item, 'arguments', ''),
                            })

        safety_result = await post_process_safety(
            agent_response_text,
            tool_calls_info,
        )

        if not safety_result.passed:
            safety_warnings = safety_result.warnings

        if safety_result.modified_response:
            agent_response_text = safety_result.modified_response

    # Step 5: Extract citations and image URLs from the response
    citations = _extract_citations(agent_response_text)
    image_url = _extract_image_url(agent_response_text)
    rag_used = _check_rag_used(agent_response_text)

    # Step 6: Store conversation turns
    _store.upsert_turn(conversation_id, ConversationTurn(
        role="user",
        content=user_message,
        denomination_context=denomination,
    ))
    _store.upsert_turn(conversation_id, ConversationTurn(
        role="assistant",
        content=agent_response_text,
        citations=citations,
        rag_context=[],
    ))

    return AgentResponse(
        response=agent_response_text,
        citations=citations,
        guardrail_triggered=False,
        rag_used=rag_used,
        image_url=image_url,
        model_used=model,
        conversation_id=conversation_id,
        safety_warnings=safety_warnings,
    )


def _extract_citations(text: str) -> list[dict]:
    """Extract scripture citations from the agent response."""
    import re
    citations = []
    # Match patterns like (John 3:16, NIV) or (Genesis 1:1-3)
    pattern = re.compile(
        r'\((\d?\s*\w+(?:\s+\w+)?)\s+(\d+):(\d+)(?:-(\d+))?\s*(?:,\s*(\w+))?\)'
    )
    for match in pattern.finditer(text):
        citation = {
            "book": match.group(1).strip(),
            "chapter": int(match.group(2)),
            "verse_start": int(match.group(3)),
        }
        if match.group(4):
            citation["verse_end"] = int(match.group(4))
        if match.group(5):
            citation["translation"] = match.group(5)
        citations.append(citation)
    return citations


def _extract_image_url(text: str) -> Optional[str]:
    """Extract generated image URL from the agent response."""
    import re
    url_pattern = re.compile(r'https://[^\s]+\.png|https://oaidalleapiprodscus[^\s]+')
    match = url_pattern.search(text)
    return match.group(0) if match else None


def _check_rag_used(text: str) -> bool:
    """Check if RAG results were used in the response."""
    indicators = [
        "Knowledge Base Results",
        "[Source",
        "knowledge base",
        "VERIFIED:",
    ]
    return any(indicator.lower() in text.lower() for indicator in indicators)


def get_conversation_store() -> ConversationStore:
    """Get the singleton conversation store."""
    return _store
