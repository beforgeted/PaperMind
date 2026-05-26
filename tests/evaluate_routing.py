"""Routing evaluation script.

Usage:
    # Rule-only evaluation (no LLM cost):
    source activate PaperMind && python tests/evaluate_routing.py

    # Full pipeline (includes LLM Router for low-confidence queries):
    source activate PaperMind && python tests/evaluate_routing.py --full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.routing.pre_router import EnhancedRuleRouter
from app.core.config import settings


def load_test_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Test set must be a JSON array")
    for i, item in enumerate(data):
        if "query" not in item or "label" not in item:
            raise ValueError(f"Item {i} missing 'query' or 'label': {item}")
    return data


def rule_only_evaluate(test_set: list[dict]) -> dict:
    """Evaluate using only the rule-based pre-router (zero LLM cost)."""
    router = EnhancedRuleRouter()
    total = len(test_set)
    correct = 0
    high_conf_count = 0
    low_conf_count = 0

    # Per-label breakdown
    label_stats: dict[str, dict] = {}
    for label in ["paper_search", "paper_deep_search", "chunk_qa", "task_status", "paper_profile"]:
        label_stats[label] = {"total": 0, "correct": 0, "wrong_route": {}, "confidences": []}

    # Track "complex" queries wrongly downgraded to chunk_qa
    complex_labels = {"paper_search", "paper_deep_search", "paper_profile", "task_status"}
    complex_total = 0
    complex_downgraded = 0

    results: list[dict] = []

    for item in test_set:
        query = item["query"]
        label = item["label"]
        result = router.route(query)

        is_correct = result.route == label
        is_high_conf = result.confidence >= settings.rule_route_threshold

        if is_correct:
            correct += 1
        if is_high_conf:
            high_conf_count += 1
        else:
            low_conf_count += 1

        label_stats[label]["total"] += 1
        label_stats[label]["confidences"].append(result.confidence)
        if is_correct:
            label_stats[label]["correct"] += 1
        else:
            wrong_to = result.route
            label_stats[label]["wrong_route"][wrong_to] = (
                label_stats[label]["wrong_route"].get(wrong_to, 0) + 1
            )

        if label in complex_labels:
            complex_total += 1
            if result.route == "chunk_qa":
                complex_downgraded += 1

        results.append({
            "query": query,
            "label": label,
            "predicted": result.route,
            "confidence": result.confidence,
            "reason": result.reason,
            "correct": is_correct,
            "high_conf": is_high_conf,
        })

    accuracy = correct / total if total > 0 else 0.0
    high_conf_rate = high_conf_count / total if total > 0 else 0.0
    low_conf_rate = low_conf_count / total if total > 0 else 0.0
    downgrade_rate = complex_downgraded / complex_total if complex_total > 0 else 0.0

    return {
        "mode": "rule_only",
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "high_conf_count": high_conf_count,
        "high_conf_rate": round(high_conf_rate, 4),
        "llm_trigger_count": low_conf_count,
        "llm_trigger_rate": round(low_conf_rate, 4),
        "complex_total": complex_total,
        "complex_downgraded": complex_downgraded,
        "complex_downgrade_rate": round(downgrade_rate, 4),
        "label_stats": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
                "avg_confidence": round(sum(v["confidences"]) / len(v["confidences"]), 4) if v["confidences"] else 0,
                "wrong_routes": v["wrong_route"],
            }
            for k, v in label_stats.items()
        },
        "detail": results,
    }


async def full_pipeline_evaluate(test_set: list[dict]) -> dict:
    """Evaluate with full pipeline: rule pre-router + LLM router for low-confidence.

    Only invokes LLM for queries where rule confidence < threshold.
    """
    from app.agents.routing.llm_router import StructuredLLMRouter

    # First pass: rule evaluation
    rule_report = rule_only_evaluate(test_set)

    llm_router = StructuredLLMRouter()
    llm_calls = 0
    llm_correct = 0
    llm_fallback = 0

    # Re-evaluate low-confidence queries with LLM
    full_results: list[dict] = []
    correct = 0
    for item in rule_report["detail"]:
        if not item["high_conf"]:
            llm_calls += 1
            try:
                pre_ctx = (
                    f"Most likely route: {item['predicted']} "
                    f"(confidence: {item['confidence']:.2f})"
                )
                decision = await llm_router.route(item["query"], pre_ctx)
                final_route = decision.route
                final_confidence = decision.confidence
                final_source = "llm"
                is_fallback = False
            except Exception:
                final_route = item["predicted"]
                final_confidence = item["confidence"]
                final_source = "fallback"
                is_fallback = True
                llm_fallback += 1
        else:
            final_route = item["predicted"]
            final_confidence = item["confidence"]
            final_source = "rule"
            is_fallback = False

        is_correct = final_route == item["label"]
        if is_correct:
            correct += 1
            if final_source == "llm" and not is_fallback:
                llm_correct += 1

        full_results.append({
            **item,
            "final_route": final_route,
            "final_confidence": final_confidence,
            "final_source": final_source,
            "llm_overridden": final_source != "rule",
            "correct": is_correct,
        })

    total = len(test_set)
    accuracy = correct / total if total > 0 else 0.0
    fallback_rate = llm_fallback / llm_calls if llm_calls > 0 else 0.0

    return {
        "mode": "full_pipeline",
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "high_conf_count": rule_report["high_conf_count"],
        "high_conf_rate": rule_report["high_conf_rate"],
        "llm_calls": llm_calls,
        "llm_trigger_rate": rule_report["llm_trigger_rate"],
        "llm_correct": llm_correct,
        "llm_fallback_count": llm_fallback,
        "fallback_rate": round(fallback_rate, 4),
        "complex_total": rule_report["complex_total"],
        "complex_downgraded": rule_report["complex_downgraded"],
        "complex_downgrade_rate": rule_report["complex_downgrade_rate"],
        "label_stats": rule_report["label_stats"],
        "detail": full_results,
    }


def print_report(report: dict) -> None:
    print("=" * 72)
    print(f"  Routing Evaluation Report  ({report['mode']})")
    print("=" * 72)
    print()
    print(f"  Total samples:          {report['total']}")
    print(f"  Correct routes:         {report['correct']}")
    print(f"  Accuracy:               {report['accuracy']:.2%}")
    print()
    print(f"  High-conf (rule >={settings.rule_route_threshold}):  {report['high_conf_count']} ({report['high_conf_rate']:.2%})")
    print(f"  LLM trigger (<{settings.rule_route_threshold}):      {report.get('llm_trigger_count', report.get('llm_calls', 0))} ({report['llm_trigger_rate']:.2%})")
    if "fallback_rate" in report:
        print(f"  LLM fallback rate:      {report['fallback_rate']:.2%}")
    print(f"  Complex downgrade rate: {report['complex_downgrade_rate']:.2%}")
    print()

    print("  Per-label breakdown:")
    print(f"  {'Label':<22} {'Total':>6} {'Correct':>8} {'Acc':>8} {'AvgConf':>8}")
    print("  " + "-" * 56)
    for label, stats in report["label_stats"].items():
        print(
            f"  {label:<22} {stats['total']:>6} {stats['correct']:>8} "
            f"{stats['accuracy']:>7.2%} {stats['avg_confidence']:>8.4f}"
        )
        if stats.get("wrong_routes"):
            for wrong_route, count in stats["wrong_routes"].items():
                print(f"    -> misrouted as {wrong_route}: {count}")
    print()

    # Print all misrouted queries
    misrouted = [d for d in report["detail"] if not d["correct"]]
    if misrouted:
        print(f"  Misrouted queries ({len(misrouted)}):")
        print()
        for i, m in enumerate(misrouted, 1):
            final = m.get("final_route", m["predicted"])
            conf = m.get("final_confidence", m["confidence"])
            src = m.get("final_source", "rule")
            print(f"  [{i}] label={m['label']} -> predicted={final} (conf={conf:.3f}, src={src})")
            print(f"      query: {m['query']}")
        print()

    # Print LLM-triggered queries (low confidence)
    low_conf = [d for d in report["detail"] if not d.get("high_conf", False)]
    if low_conf:
        print(f"  Low-confidence queries (LLM candidates, {len(low_conf)}):")
        print()
        for i, m in enumerate(low_conf[:10], 1):
            print(f"  [{i}] label={m['label']} rule->{m['predicted']} (conf={m['confidence']:.3f})")
            print(f"      query: {m['query']}")
        if len(low_conf) > 10:
            print(f"  ... and {len(low_conf) - 10} more")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate PaperMind routing accuracy")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline including LLM Router for low-confidence queries",
    )
    parser.add_argument(
        "--test-set",
        default="tests/routing/fixtures/routing_test_set.json",
        help="Path to test set JSON file",
    )
    args = parser.parse_args()

    test_set = load_test_set(args.test_set)
    print(f"Loaded {len(test_set)} test queries from {args.test_set}\n")

    if args.full:
        report = asyncio.run(full_pipeline_evaluate(test_set))
    else:
        report = rule_only_evaluate(test_set)

    print_report(report)

    # Save detailed results
    out_path = f"tests/routing_eval_{report['mode']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
