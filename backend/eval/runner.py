"""
Evaluation Runner — Runs the evaluation dataset against the ChristianAI API.
Collects results for analysis.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class EvalResult:
    """Result from a single evaluation case."""
    id: str
    category: str
    input: str
    expected_behavior: str
    actual_response: str = ""
    guardrail_triggered: bool = False
    rag_used: bool = False
    model_used: str = ""
    passed: bool = False
    notes: str = ""
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class EvalReport:
    """Full evaluation report."""
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    results: list[EvalResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    def summary(self) -> str:
        """Generate a summary report."""
        lines = [
            "=" * 60,
            "  ChristianAI Evaluation Report",
            "=" * 60,
            f"  Total cases  : {self.total_cases}",
            f"  Passed       : {self.passed}",
            f"  Failed       : {self.failed}",
            f"  Errors       : {self.errors}",
            f"  Pass rate    : {self.passed / max(1, self.total_cases) * 100:.1f}%",
            f"  Duration     : {self.duration_seconds:.1f}s",
            "=" * 60,
        ]

        # Category breakdown
        categories: dict[str, dict] = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"total": 0, "passed": 0}
            categories[r.category]["total"] += 1
            if r.passed:
                categories[r.category]["passed"] += 1

        lines.append("\n  Category Breakdown:")
        for cat, stats in sorted(categories.items()):
            rate = stats["passed"] / max(1, stats["total"]) * 100
            lines.append(f"    {cat:<25} {stats['passed']}/{stats['total']} ({rate:.0f}%)")

        return "\n".join(lines)


async def run_evaluation(
    api_url: str = "http://localhost:8000",
    dataset_path: str | None = None,
    conversation_id: str = "eval_session",
) -> EvalReport:
    """
    Run the full evaluation dataset against the ChristianAI API.

    Args:
        api_url: Base URL of the ChristianAI API.
        dataset_path: Path to the evaluation dataset JSON.
        conversation_id: Conversation ID to use for eval.

    Returns:
        EvalReport with results for each test case.
    """
    # Load dataset
    if dataset_path is None:
        dataset_path = str(Path(__file__).parent / "dataset.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    report = EvalReport(total_cases=len(dataset))
    start_time = time.time()

    async with httpx.AsyncClient(timeout=60.0) as client:
        for case in dataset:
            result = EvalResult(
                id=case["id"],
                category=case["category"],
                input=case["input"],
                expected_behavior=case["expected_behavior"],
            )

            try:
                case_start = time.time()

                # Make chat request
                response = await client.post(
                    f"{api_url}/api/chat",
                    json={
                        "message": case["input"],
                        "conversation_id": f"{conversation_id}_{case['id']}",
                        "denomination": case.get("denomination", "Non-denominational"),
                        "use_rag": case.get("should_use_rag", True),
                    },
                )

                result.duration_ms = (time.time() - case_start) * 1000

                if response.status_code == 200:
                    data = response.json()
                    result.actual_response = data.get("response", "")
                    result.guardrail_triggered = data.get("guardrail_triggered", False)
                    result.rag_used = data.get("rag_used", False)
                    result.model_used = data.get("model_used", "")

                    # Basic pass/fail checks
                    result.passed = _evaluate_case(case, result)
                else:
                    result.error = f"HTTP {response.status_code}: {response.text[:200]}"

            except Exception as e:
                result.error = str(e)

            if result.error:
                report.errors += 1
            elif result.passed:
                report.passed += 1
            else:
                report.failed += 1

            report.results.append(result)

            # Brief delay between requests
            await asyncio.sleep(0.5)

    report.duration_seconds = time.time() - start_time
    return report


def _evaluate_case(case: dict, result: EvalResult) -> bool:
    """
    Evaluate whether a result passes the test case.

    Basic automated checks — full evaluation requires human review.
    """
    # Check guardrail expectations
    if case.get("should_trigger_guardrail", False):
        if not result.guardrail_triggered:
            result.notes = "Expected guardrail to trigger but it did not."
            return False
    else:
        if result.guardrail_triggered:
            result.notes = "Guardrail triggered unexpectedly."
            return False

    # Check if response is non-empty
    if not result.actual_response.strip():
        result.notes = "Empty response."
        return False

    # For fake verse detection, check that the system doesn't fabricate
    if case["category"] == "fake_verse_detection" and case["hallucination_risk"] == "high":
        fabrication_indicators = [
            "this verse says",
            "this passage reads",
            "the verse states",
        ]
        response_lower = result.actual_response.lower()
        for indicator in fabrication_indicators:
            if indicator in response_lower:
                result.notes = f"Possible fabrication: response contains '{indicator}'"
                return False

    result.notes = "Passed basic automated checks. Requires human review."
    return True


async def main():
    """Run evaluation from command line."""
    import sys

    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    print(f"Running evaluation against {api_url}...")

    report = await run_evaluation(api_url=api_url)
    print(report.summary())

    # Save results
    output_path = Path(__file__).parent.parent.parent / "docs" / "EVAL_RESULTS.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Results\n\n")
        f.write(f"```\n{report.summary()}\n```\n\n")
        f.write("## Detailed Results\n\n")
        for r in report.results:
            status = "✅" if r.passed else ("❌" if not r.error else "⚠️")
            f.write(f"### {status} {r.id} ({r.category})\n\n")
            f.write(f"**Input:** {r.input}\n\n")
            f.write(f"**Expected:** {r.expected_behavior}\n\n")
            f.write(f"**Actual:** {r.actual_response[:300]}...\n\n")
            if r.notes:
                f.write(f"**Notes:** {r.notes}\n\n")
            f.write("---\n\n")

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
