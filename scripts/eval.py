"""
scripts/eval.py — evaluation harness for the multi-agent platform.

RAGAS-style automated eval doesn't generalize to a multi-agent system the
way it does to a single RAG pipeline (there's no single "retrieved context
+ generated answer" pair to score) -- the equivalent here is deterministic,
per-component checks: routing accuracy (does the fast-path router send each
question to the agent that actually owns it) plus honest reporting of the
trained models' real metrics, read straight from the files train_models.py
already wrote (no re-computation, no cherry-picking).

Deliberately does NOT call a live LLM: routing accuracy is measured against
regex_route()/fast_route() (the deterministic pre-filter layer), which is
reproducible across runs -- unlike the LLM router fallback, whose exact
wording can vary between calls. A question the fast path can't confidently
resolve is reported as "no opinion" rather than silently falling through to
an LLM call and grading THAT instead.

Run:
    python scripts/eval.py
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from orchestration.semantic_router import fast_route, regex_route

MODELS_DIR = os.path.join(_ROOT, "models")


# ── Routing eval set ──────────────────────────────────────────────────────────
# (question, expected_route). Deliberately unambiguous, first-turn phrasings
# only -- the same restriction fast_route() itself requires (see
# orchestration/semantic_router.py's module docstring): a follow-up like
# "what about that one" is NOT eligible for fast-path routing at all, so it
# has no place in an eval of the fast path's accuracy.
ROUTING_EVAL_SET: list[tuple[str, str]] = [
    ("show me all customers in Chennai", "sql"),
    ("list the products you have", "sql"),
    ("how many transactions happened last month", "sql"),
    ("what is the profit margin on PRD001", "finance"),
    ("give me a bulk quote for 50 units", "finance"),
    ("what discount does a gold tier customer get", "finance"),
    ("can CUST001 apply for a loan of 50000", "credit"),
    ("what's the EMI on a 100000 rupee loan", "credit"),
    ("check credit eligibility for CUST003", "credit"),
    ("which customers should we chase for overdue payments", "credit"),
    ("is CUST009 at risk of churning", "risk"),
    ("flag any suspicious transactions for CUST001", "risk"),
    ("forecast revenue for the next 3 months", "forecast"),
    ("predict revenue for the next 10 days", "forecast"),
    ("what are the market trends in electronics", "research"),
    ("how do competitors price similar products", "research"),
]


def eval_routing() -> dict:
    """Score regex_route() (fully deterministic) against the labeled set."""
    correct, no_opinion, wrong = 0, 0, []
    for question, expected in ROUTING_EVAL_SET:
        got = regex_route(question)
        if got == expected:
            correct += 1
        elif got is None:
            no_opinion += 1
        else:
            wrong.append((question, expected, got))

    total = len(ROUTING_EVAL_SET)
    return {
        "total": total,
        "correct": correct,
        "no_opinion": no_opinion,
        "wrong": wrong,
        "accuracy_on_attempted": round(correct / max(total - no_opinion, 1), 4),
        "accuracy_overall": round(correct / total, 4),
    }


# ── Model metrics (read, not recomputed) ──────────────────────────────────────

def _read_json(filename: str) -> dict | None:
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def eval_models() -> dict:
    churn    = _read_json("churn_metrics.json")
    forecast = _read_json("forecast_metrics.json")
    return {"churn": churn, "forecast": forecast}


# ── Report ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("EVAL: Routing accuracy (deterministic regex_route(), no LLM call)")
    print("=" * 70)
    routing = eval_routing()
    print(f"  Total questions:        {routing['total']}")
    print(f"  Correct:                {routing['correct']}")
    print(f"  No opinion (would fall through to LLM router in production):"
          f" {routing['no_opinion']}")
    print(f"  Wrong:                  {len(routing['wrong'])}")
    for q, expected, got in routing["wrong"]:
        print(f"    MISROUTED: {q!r} -> expected {expected!r}, got {got!r}")
    print(f"  Accuracy (of attempted): {routing['accuracy_on_attempted']*100:.1f}%")
    print(f"  Accuracy (overall):      {routing['accuracy_overall']*100:.1f}%")

    print("\n" + "=" * 70)
    print("EVAL: Trained model metrics (read from models/*.json, not recomputed)")
    print("=" * 70)
    models = eval_models()

    churn = models["churn"]
    if churn is None:
        print("  Churn model: not found -- run models/train_models.py first.")
    else:
        n_classes = len(churn.get("classes", []))
        baseline  = round(100 / n_classes, 1) if n_classes else None
        print(f"  Churn classifier (LeaveOneOut CV):")
        print(f"    Accuracy:    {churn['accuracy']*100:.1f}%")
        print(f"    Classes:     {churn['classes']}  (random baseline ~{baseline}%)")
        print(f"    n_customers: {churn['n_customers']}")
        print(f"    Top features: "
              f"{sorted(churn['feature_importances'].items(), key=lambda x: -x[1])[:3]}")

    forecast = models["forecast"]
    if forecast is None:
        print("  Forecast model: not found -- run models/train_models.py first.")
    else:
        print(f"  Revenue ARIMA (order {tuple(forecast['best_order'])}):")
        print(f"    Test MAE:  Rs.{forecast['mae']:,.0f}")
        print(f"    Test MAPE: {forecast['mape']:.1f}%")
        print(f"    Trained through: {forecast['last_month']} "
              f"({forecast['train_months']} months of history)")

    print("=" * 70)


if __name__ == "__main__":
    main()
