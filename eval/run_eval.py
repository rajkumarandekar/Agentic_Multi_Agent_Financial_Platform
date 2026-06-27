"""
Eval harness — runs all test cases through the live graph and prints a scorecard.

Run from the project root:
    python -m eval.run_eval

Prerequisites:
  - .env populated with GROQ_API_KEY
  - For RAG cases:  python -m ingestion.pdf_ingest data/<file>.pdf
  - For SQL cases:  python scripts/seed_db.py

Scoring:
  Routing accuracy  — actual agent_used == expected_route  (1 or 0)
  Keyword score     — fraction of expected_keywords found (case-insensitive)
                      in the answer.  Empty list → N/A (skipped).
  Pass              — routing correct AND keyword_score >= 0.5 (or N/A)
"""

import asyncio
import os
import sys

# Make sure the project root is on sys.path when run as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from orchestration.graph import graph          # noqa: E402
from orchestration.state import AgentState     # noqa: E402
from eval.test_set import TEST_CASES           # noqa: E402

KEYWORD_PASS_THRESHOLD = 0.5   # fraction of keywords that must be present


# ---------------------------------------------------------------------------
# Per-case scoring helpers
# ---------------------------------------------------------------------------

def _score_keywords(answer: str, keywords: list[str]) -> tuple[int, int]:
    """Return (found, total) keyword counts (case-insensitive substring match)."""
    lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in lower)
    return found, len(keywords)


def _make_initial_state(question: str) -> AgentState:
    return AgentState(
        question=question,
        messages=[HumanMessage(content=question)],
        route="",
        answer="",
        sources=[],
        agent_used="",
        guardrail_results={},
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_all() -> list[dict]:
    """Invoke the graph for every test case and return raw result records."""
    records = []
    for case in TEST_CASES:
        print(f"  [{case['id']}] {case['question'][:60]}...", end=" ", flush=True)
        try:
            state = _make_initial_state(case["question"])
            result = await graph.ainvoke(state)
            actual_route  = result.get("agent_used", "")
            actual_answer = result.get("answer", "")
            error = None
        except Exception as exc:
            actual_route  = "ERROR"
            actual_answer = ""
            error = str(exc)

        route_ok = actual_route == case["expected_route"]
        kw_found, kw_total = _score_keywords(actual_answer, case["expected_keywords"])
        has_keywords = kw_total > 0

        if has_keywords:
            kw_score = kw_found / kw_total
            kw_label = f"{kw_found}/{kw_total} ({kw_score:.0%})"
            passed = route_ok and kw_score >= KEYWORD_PASS_THRESHOLD
        else:
            kw_score = None
            kw_label = "N/A"
            passed = route_ok

        status = "PASS" if passed else "FAIL"
        print(status)

        records.append({
            "id":             case["id"],
            "description":    case["description"],
            "expected_route": case["expected_route"],
            "actual_route":   actual_route,
            "route_ok":       route_ok,
            "kw_label":       kw_label,
            "kw_score":       kw_score,
            "passed":         passed,
            "error":          error,
        })
    return records


# ---------------------------------------------------------------------------
# Scorecard printer
# ---------------------------------------------------------------------------

def _print_scorecard(records: list[dict]) -> None:
    routes = ["rag", "sql", "tool"]

    print("\n" + "=" * 70)
    print("PHASE 3 EVAL SCORECARD")
    print("=" * 70)

    # Per-case table
    header = f"{'ID':<10} {'Expected':<10} {'Actual':<10} {'Route':>6}  {'Keywords':<14}  {'Pass'}"
    print(header)
    print("-" * len(header))
    for r in records:
        route_sym = "YES" if r["route_ok"] else "NO "
        pass_sym  = "PASS" if r["passed"] else "FAIL"
        print(
            f"{r['id']:<10} {r['expected_route']:<10} {r['actual_route']:<10} "
            f"{route_sym:>6}  {r['kw_label']:<14}  {pass_sym}"
        )
        if r["error"]:
            print(f"           ERROR: {r['error'][:80]}")

    # Per-route summary
    print("\n" + "-" * 50)
    print(f"{'Route':<8} {'Cases':>5}  {'Routing':>12}  {'Quality':>12}")
    print("-" * 50)

    total_cases = total_route_ok = total_kw_found = total_kw_total = total_pass = 0

    for route in routes:
        group = [r for r in records if r["expected_route"] == route]
        n = len(group)
        route_ok  = sum(1 for r in group if r["route_ok"])
        kw_cases  = [r for r in group if r["kw_score"] is not None]
        kw_found  = sum(r["kw_score"] * len(TEST_CASES) for r in kw_cases)  # approx; use label
        passes    = sum(1 for r in group if r["passed"])

        # Recompute keyword totals cleanly
        kw_scored = [(r["kw_score"], r["kw_label"]) for r in group if r["kw_score"] is not None]
        if kw_scored:
            avg_kw = sum(s for s, _ in kw_scored) / len(kw_scored)
            quality_label = f"{avg_kw:.0%} ({len(kw_scored)} scored)"
        else:
            quality_label = "N/A (routing only)"

        print(
            f"{route:<8} {n:>5}  "
            f"{route_ok}/{n} ({route_ok/n:.0%}):>12  "
            f"{quality_label:>12}"
        )

        total_cases    += n
        total_route_ok += route_ok
        total_pass     += passes

    print("-" * 50)
    print(
        f"{'TOTAL':<8} {total_cases:>5}  "
        f"{total_route_ok}/{total_cases} ({total_route_ok/total_cases:.0%}) routing"
    )
    print(f"\nOverall pass rate: {total_pass}/{total_cases} ({total_pass/total_cases:.0%})")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    print(f"\nRunning {len(TEST_CASES)} test cases...\n")
    records = await run_all()
    _print_scorecard(records)


if __name__ == "__main__":
    asyncio.run(_main())
