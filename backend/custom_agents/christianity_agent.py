"""
Christianity Agent — Single LangChain tool-calling agent with integrated guardrails and RAG.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

from config.settings import get_settings
from custom_agents.guardrail_agent import (
    GuardrailDecision,
    build_refusal_response,
    parse_guardrail_refusal,
)
from llm.langchain_client import build_chat_model, detect_provider
from memory.conversation_store import ConversationStore, ConversationTurn
from moderation.safety_layer import post_process_safety
from prompts import load_combined_prompt
from tools.image_gen_tool import image_gen_tool
from tools.rag_tool import rag_tool, set_vector_db_context
from tools.scripture_verify_tool import scripture_verify_tool


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


_store = ConversationStore()

# Trivial messages that never need knowledge-base grounding.
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|yo|thanks|thank you|good (morning|evening|afternoon|night)|bye|goodbye)\b",
    re.IGNORECASE,
)


def _needs_rag(message: str) -> bool:
    """Heuristic: does this message warrant knowledge-base grounding?"""
    m = message.strip()
    if len(m) < 3:
        return False
    if _GREETING_RE.match(m) and len(m.split()) <= 4:
        return False
    return True


def _history_to_messages(conversation_id: str) -> list:
    """Convert stored turns to LangChain message objects."""
    conversation = _store.get_conversation(conversation_id)
    if not conversation:
        return []

    messages = []
    for turn in conversation.turns[-20:]:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        elif turn.role == "assistant":
            messages.append(AIMessage(content=turn.content))
    return messages


def _extract_tool_calls(messages: list) -> list[dict]:
    """Collect tool names and inputs from agent message history (from AIMessage requests)."""
    tool_calls_info = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_info.append({
                    "name": tc.get("name", ""),
                    "arguments": tc.get("args", tc.get("arguments", "")),
                })
    return tool_calls_info


def _log_agent_trace(messages: list) -> None:
    """Emit a structured trace of the agent's reasoning steps for observability."""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                logger.info(
                    f"[agent] tool call -> {tc.get('name', '?')} "
                    f"args={tc.get('args', tc.get('arguments', {}))}"
                )
        elif isinstance(msg, ToolMessage):
            preview = str(msg.content)[:200].replace("\n", " ")
            logger.info(
                f"[agent] tool result <- {msg.name or '?'} "
                f"({len(str(msg.content))} chars): {preview}"
            )


def _final_response_text(messages: list) -> str:
    """Return the last assistant text response from the message list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            text = msg.content
            if isinstance(text, str):
                return text
            if isinstance(text, list):
                parts = [
                    block.get("text", "")
                    for block in text
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "".join(parts)
    return ""


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
    Run the single LangChain Christianity agent.

    Flow:
    1. Build tools (RAG optional) and set vector DB context
    2. Run agent with conversation history
    3. Parse inline guardrail refusals
    4. Post-process safety on output
    5. Store conversation turn
    """
    del guardrail_model  # single agent uses one model; kept for API compatibility

    started = time.perf_counter()
    settings = get_settings()
    effective_vector_db = vector_db or settings.default_vector_db
    provider = detect_provider(model, settings.default_llm_provider)

    logger.info(
        f"[agent] run start | conv={conversation_id} | provider={provider} | "
        f"model={model} | denomination={denomination} | use_rag={use_rag} | "
        f"vector_db={effective_vector_db}"
    )
    logger.debug(f"[agent] user_message: {user_message!r}")

    conversation = _store.get_conversation(conversation_id)
    if conversation and conversation.denomination != denomination:
        conversation.denomination = denomination

    tools = [scripture_verify_tool, image_gen_tool]
    if use_rag:
        tools.insert(0, rag_tool)

    set_vector_db_context(effective_vector_db)

    tool_names = [getattr(t, "name", str(t)) for t in tools]
    logger.info(f"[agent] tools bound: {tool_names}")

    context_prefix = (
        f"[User denomination: {denomination}]\n"
        f"[Vector DB: {effective_vector_db}]\n"
        f"[RAG enabled: {use_rag}. For ANY biblical, scriptural, or theological "
        f"question you MUST call rag_tool first, then scripture_verify_tool before "
        f"citing a specific verse.]\n\n"
    )
    enriched_message = context_prefix + user_message

    chat_history = _history_to_messages(conversation_id)
    system_prompt = load_combined_prompt()

    llm = build_chat_model(model, settings, provider)
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    messages = [*chat_history, HumanMessage(content=enriched_message)]

    try:
        result = await agent_graph.ainvoke({"messages": messages})
        result_messages = result.get("messages", [])
        _log_agent_trace(result_messages)
        agent_response_text = _final_response_text(result_messages)
        tool_calls_info = _extract_tool_calls(result_messages)
        rag_used = any(t["name"] == "rag_tool" for t in tool_calls_info)

        # Deterministic grounding fallback: if the model skipped RAG on a substantive
        # question, retrieve from the knowledge base ourselves and re-run grounded.
        if use_rag and not rag_used and _needs_rag(user_message):
            logger.warning(
                "[agent] model skipped rag_tool; running deterministic grounded retry"
            )
            try:
                kb_context = await rag_tool.ainvoke(
                    {"query": user_message, "top_k": 5}
                )
                grounding_msg = HumanMessage(content=(
                    "Knowledge base results were retrieved for you. Use them as the "
                    "primary source and cite the listed references:\n\n"
                    f"{kb_context}"
                ))
                result2 = await agent_graph.ainvoke(
                    {"messages": [*messages, grounding_msg]}
                )
                result_messages2 = result2.get("messages", [])
                _log_agent_trace(result_messages2)
                grounded_text = _final_response_text(result_messages2)
                if grounded_text:
                    agent_response_text = grounded_text
                tool_calls_info.append(
                    {"name": "rag_tool", "arguments": {"query": user_message}}
                )
                rag_used = True
            except Exception:
                logger.exception("[agent] grounded retry failed; using original answer")

        if not agent_response_text:
            logger.warning("[agent] empty final response from model")
            agent_response_text = (
                "I apologize, but I could not generate a response. Please try again."
            )

    except Exception as e:
        logger.exception("[agent] error running LangChain agent:")
        agent_response_text = (
            "I apologize, but I encountered an issue processing your request. "
            f"Please try again. (Error: {str(e)[:100]})"
        )
        return AgentResponse(
            response=agent_response_text,
            model_used=model,
            conversation_id=conversation_id,
        )

    guardrail_decision: Optional[GuardrailDecision] = None
    if settings.enable_guardrail:
        agent_response_text, guardrail_decision = parse_guardrail_refusal(agent_response_text)

    if guardrail_decision and guardrail_decision.blocked:
        logger.warning(
            f"[agent] guardrail refusal | category={guardrail_decision.category}"
        )
        refusal = build_refusal_response(guardrail_decision)
        response_text = agent_response_text or refusal["response"]

        _store.upsert_turn(conversation_id, ConversationTurn(
            role="user",
            content=user_message,
            guardrail_triggered=True,
            denomination_context=denomination,
        ))
        _store.upsert_turn(conversation_id, ConversationTurn(
            role="assistant",
            content=response_text,
            guardrail_triggered=True,
        ))

        return AgentResponse(
            response=response_text,
            guardrail_triggered=True,
            guardrail_category=guardrail_decision.category,
            model_used=model,
            conversation_id=conversation_id,
        )

    safety_warnings = []
    if settings.enable_post_safety:
        safety_result = await post_process_safety(agent_response_text, tool_calls_info)
        if not safety_result.passed:
            safety_warnings = safety_result.warnings
            logger.warning(f"[agent] post-safety warnings: {safety_warnings}")
        if safety_result.modified_response:
            agent_response_text = safety_result.modified_response

    citations = _extract_citations(agent_response_text)
    image_url = _extract_image_url(agent_response_text)
    rag_used = rag_used or _check_rag_used(agent_response_text)

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        f"[agent] run done | conv={conversation_id} | tool_calls="
        f"{[t['name'] for t in tool_calls_info]} | rag_used={rag_used} | "
        f"citations={len(citations)} | warnings={len(safety_warnings)} | "
        f"image={'yes' if image_url else 'no'} | {elapsed_ms:.0f}ms"
    )

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
    citations = []
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
    url_pattern = re.compile(r'https://[^\s]+\.png|https://oaidalleapiprodscus[^\s]+')
    match = url_pattern.search(text)
    return match.group(0) if match else None


def _check_rag_used(text: str) -> bool:
    """Check if RAG results appear in the response."""
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
