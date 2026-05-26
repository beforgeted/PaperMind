"""Benchmark routing latency: hybrid (rule bypass) vs all-LLM.

Uses the same 150-query set as ``evaluate_routing.py``.

Usage:
    # Full benchmark (requires DASHSCOPE_API_KEY, calls DashScope ~100–250 times):
    python tests/benchmark_routing_latency.py

    # Quick smoke (10 queries):
    python tests/benchmark_routing_latency.py --limit 10

    # Rule-only timing (no API calls):
    python tests/benchmark_routing_latency.py --rule-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.routing.llm_router import StructuredLLMRouter
from app.agents.routing.pre_router import EnhancedRuleRouter, PreRouteResult
from app.core.config import settings


def load_test_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Test set must be a JSON array")
    return data


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def _summarize_ms(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(samples_ms),
        "total_ms": round(sum(samples_ms), 2),
        "mean_ms": round(statistics.mean(samples_ms), 2),
        "median_ms": round(statistics.median(samples_ms), 2),
        "p95_ms": round(_percentile(samples_ms, 0.95), 2),
        "min_ms": round(min(samples_ms), 2),
        "max_ms": round(max(samples_ms), 2),
    }


@dataclass
class QueryTiming:
    query: str
    label: str
    pre_route: str
    pre_confidence: float
    final_route: str
    used_llm: bool
    pre_ms: float
    llm_ms: float
    total_ms: float


@dataclass
class ModeReport:
    mode: str
    total_queries: int
    llm_calls: int
    rule_bypass_count: int
    wall_seconds: float
    queries_per_second: float
    timings: list[QueryTiming] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        totals = [t.total_ms for t in self.timings]
        pre_only = [t.pre_ms for t in self.timings]
        llm_only = [t.llm_ms for t in self.timings if t.used_llm]
        return {
            "mode": self.mode,
            "total_queries": self.total_queries,
            "llm_calls": self.llm_calls,
            "rule_bypass_count": self.rule_bypass_count,
            "rule_bypass_rate": round(self.rule_bypass_count / self.total_queries, 4)
            if self.total_queries
            else 0.0,
            "wall_seconds": round(self.wall_seconds, 3),
            "queries_per_second": round(self.queries_per_second, 3),
            "total_latency": _summarize_ms(totals),
            "pre_route_latency": _summarize_ms(pre_only),
            "llm_latency": _summarize_ms(llm_only),
            "detail": [
                {
                    "query": t.query,
                    "label": t.label,
                    "pre_route": t.pre_route,
                    "pre_confidence": t.pre_confidence,
                    "final_route": t.final_route,
                    "used_llm": t.used_llm,
                    "pre_ms": round(t.pre_ms, 2),
                    "llm_ms": round(t.llm_ms, 2),
                    "total_ms": round(t.total_ms, 2),
                }
                for t in self.timings
            ],
        }


def _build_pre_context(pre: PreRouteResult) -> str:
    return (
        f"Most likely route: {pre.route} "
        f"(confidence: {pre.confidence:.2f}, reason: {pre.reason})"
    )


async def _route_one(
    query: str,
    label: str,
    pre_router: EnhancedRuleRouter,
    llm_router: StructuredLLMRouter,
    *,
    force_llm: bool,
) -> QueryTiming:
    t0 = time.perf_counter()
    pre = pre_router.route(query)
    t_pre = time.perf_counter()
    pre_ms = (t_pre - t0) * 1000

    used_llm = force_llm or pre.confidence < settings.rule_route_threshold
    llm_ms = 0.0
    final_route = pre.route

    if used_llm:
        t_llm0 = time.perf_counter()
        pre_ctx = _build_pre_context(pre) if pre.confidence < settings.rule_route_threshold else None
        if force_llm and pre_ctx is None:
            pre_ctx = _build_pre_context(pre)
        decision = await llm_router.route(query, pre_ctx)
        llm_ms = (time.perf_counter() - t_llm0) * 1000
        final_route = decision.route

    total_ms = (time.perf_counter() - t0) * 1000
    return QueryTiming(
        query=query,
        label=label,
        pre_route=pre.route,
        pre_confidence=pre.confidence,
        final_route=final_route,
        used_llm=used_llm,
        pre_ms=pre_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
    )


async def benchmark_mode(
    test_set: list[dict],
    *,
    mode: str,
    force_llm: bool,
    llm_router: StructuredLLMRouter,
) -> ModeReport:
    pre_router = EnhancedRuleRouter()
    timings: list[QueryTiming] = []
    llm_calls = 0
    rule_bypass = 0

    wall0 = time.perf_counter()
    for item in test_set:
        timing = await _route_one(
            item["query"],
            item.get("label", ""),
            pre_router,
            llm_router,
            force_llm=force_llm,
        )
        timings.append(timing)
        if timing.used_llm:
            llm_calls += 1
        else:
            rule_bypass += 1
    wall_seconds = time.perf_counter() - wall0

    n = len(test_set)
    return ModeReport(
        mode=mode,
        total_queries=n,
        llm_calls=llm_calls,
        rule_bypass_count=rule_bypass,
        wall_seconds=wall_seconds,
        queries_per_second=n / wall_seconds if wall_seconds > 0 else 0.0,
        timings=timings,
    )


def benchmark_rule_only(test_set: list[dict]) -> ModeReport:
    """Local rule router only — no network / LLM."""
    pre_router = EnhancedRuleRouter()
    timings: list[QueryTiming] = []
    high_conf = 0

    wall0 = time.perf_counter()
    for item in test_set:
        t0 = time.perf_counter()
        pre = pre_router.route(item["query"])
        total_ms = (time.perf_counter() - t0) * 1000
        if pre.confidence >= settings.rule_route_threshold:
            high_conf += 1
        timings.append(
            QueryTiming(
                query=item["query"],
                label=item.get("label", ""),
                pre_route=pre.route,
                pre_confidence=pre.confidence,
                final_route=pre.route,
                used_llm=False,
                pre_ms=total_ms,
                llm_ms=0.0,
                total_ms=total_ms,
            )
        )
    wall_seconds = time.perf_counter() - wall0
    n = len(test_set)
    return ModeReport(
        mode="rule_only",
        total_queries=n,
        llm_calls=0,
        rule_bypass_count=high_conf,
        wall_seconds=wall_seconds,
        queries_per_second=n / wall_seconds if wall_seconds > 0 else 0.0,
        timings=timings,
    )


def _compare(hybrid: ModeReport, all_llm: ModeReport) -> dict[str, Any]:
    h_mean = statistics.mean(t.total_ms for t in hybrid.timings) if hybrid.timings else 0.0
    a_mean = statistics.mean(t.total_ms for t in all_llm.timings) if all_llm.timings else 0.0
    h_wall = hybrid.wall_seconds
    a_wall = all_llm.wall_seconds
    return {
        "mean_total_ms_ratio_all_llm_over_hybrid": round(a_mean / h_mean, 3) if h_mean > 0 else None,
        "wall_seconds_saved": round(a_wall - h_wall, 3),
        "wall_speedup_x": round(a_wall / h_wall, 3) if h_wall > 0 else None,
        "llm_calls_saved": all_llm.llm_calls - hybrid.llm_calls,
        "estimated_llm_cost_ratio": round(hybrid.llm_calls / all_llm.llm_calls, 4)
        if all_llm.llm_calls
        else None,
    }


def print_report(
    hybrid: Optional[ModeReport],
    all_llm: Optional[ModeReport],
    rule_only: Optional[ModeReport],
    comparison: Optional[dict[str, Any]],
) -> None:
    threshold = settings.rule_route_threshold
    model = settings.llm_model

    print("=" * 72)
    print("  Routing Latency Benchmark")
    print("=" * 72)
    print(f"  LLM model: {model}")
    print(f"  Rule bypass threshold: {threshold}")
    print()

    def _print_mode(report: ModeReport) -> None:
        totals = _summarize_ms([t.total_ms for t in report.timings])
        llm_stats = _summarize_ms([t.llm_ms for t in report.timings if t.used_llm])
        print(f"  [{report.mode}]")
        print(f"    Queries:           {report.total_queries}")
        print(f"    LLM calls:         {report.llm_calls}")
        print(f"    Rule bypass:       {report.rule_bypass_count} ({report.rule_bypass_count / report.total_queries:.1%})")
        print(f"    Wall time:         {report.wall_seconds:.2f}s")
        print(f"    Throughput:        {report.queries_per_second:.2f} q/s")
        print(f"    Total/query mean:  {totals['mean_ms']:.2f} ms (p50={totals['median_ms']}, p95={totals['p95_ms']})")
        if report.llm_calls:
            print(f"    LLM call mean:     {llm_stats['mean_ms']:.2f} ms (p95={llm_stats['p95_ms']})")
        print()

    if rule_only is not None:
        _print_mode(rule_only)
    if hybrid is not None:
        _print_mode(hybrid)
    if all_llm is not None:
        _print_mode(all_llm)

    if comparison:
        print("  [hybrid vs all_llm]")
        print(f"    Mean latency ratio (all_llm / hybrid): {comparison['mean_total_ms_ratio_all_llm_over_hybrid']}x")
        print(f"    Wall speedup (hybrid faster):           {comparison['wall_speedup_x']}x")
        print(f"    Wall time saved (hybrid):               {comparison['wall_seconds_saved']:.2f}s")
        print(f"    LLM calls saved by hybrid:              {comparison['llm_calls_saved']}")
        print(f"    LLM call fraction (hybrid/all_llm):     {comparison['estimated_llm_cost_ratio']}")
        print()


async def run_benchmark(
    test_set: list[dict],
    *,
    warmup: bool,
) -> tuple[ModeReport, ModeReport]:
    llm_router = StructuredLLMRouter()

    if warmup and test_set:
        await llm_router.route(test_set[0]["query"], None)

    hybrid = await benchmark_mode(
        test_set,
        mode="hybrid_rule_bypass",
        force_llm=False,
        llm_router=llm_router,
    )
    all_llm = await benchmark_mode(
        test_set,
        mode="all_llm",
        force_llm=True,
        llm_router=llm_router,
    )
    return hybrid, all_llm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark routing latency: hybrid vs all-LLM on routing fixtures",
    )
    parser.add_argument(
        "--test-set",
        default="tests/routing/fixtures/routing_test_set.json",
        help="Path to test set JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run first N queries (for smoke tests)",
    )
    parser.add_argument(
        "--rule-only",
        action="store_true",
        help="Only benchmark local rule router (no DashScope calls)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip one LLM warmup call before timing",
    )
    parser.add_argument(
        "--output",
        default="tests/routing_benchmark_latency.json",
        help="Where to write JSON report",
    )
    args = parser.parse_args()

    test_set = load_test_set(args.test_set)
    if args.limit is not None:
        test_set = test_set[: args.limit]

    print(f"Loaded {len(test_set)} queries from {args.test_set}\n")

    if args.rule_only:
        rule_report = benchmark_rule_only(test_set)
        print_report(None, None, rule_report, None)
        out = {
            "test_set": args.test_set,
            "limit": args.limit,
            "rule_route_threshold": settings.rule_route_threshold,
            "llm_model": settings.llm_model,
            "rule_only": rule_report.to_dict(),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Report saved to {args.output}")
        return

    if not settings.dashscope_api_key:
        print("ERROR: DASHSCOPE_API_KEY is not set; cannot benchmark LLM routing.")
        print("Set the key in .env or run with --rule-only for local timing only.")
        sys.exit(1)

    hybrid, all_llm = asyncio.run(
        run_benchmark(test_set, warmup=not args.no_warmup),
    )
    rule_report = benchmark_rule_only(test_set)
    comparison = _compare(hybrid, all_llm)

    print_report(hybrid, all_llm, rule_report, comparison)

    out = {
        "test_set": args.test_set,
        "limit": args.limit,
        "rule_route_threshold": settings.rule_route_threshold,
        "llm_model": settings.llm_model,
        "comparison": comparison,
        "rule_only": rule_report.to_dict(),
        "hybrid_rule_bypass": hybrid.to_dict(),
        "all_llm": all_llm.to_dict(),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
