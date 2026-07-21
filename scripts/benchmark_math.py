"""Benchmark reproductible du calculateur mathématique sécurisé.

Le banc d'essai ne touche jamais à la base de mémoire. Il reconstruit chaque
résultat attendu par du code Python indépendant, puis compare la sortie du
moteur. Le nombre d'expressions peut donc monter sans créer autant de
souvenirs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.math_engine import MathEngine, MathEngineError  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("la valeur doit être supérieure à zéro")
    return parsed


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _case(index: int) -> tuple[str, int | float, str]:
    """Construit une expression et son oracle sans appeler le moteur."""

    a = 2 + (index * 17) % 997
    b = 2 + (index * 31) % 499
    c = 2 + (index * 13) % 97
    family = index % 10
    if family == 0:
        return f"{a} + {b} * {c}", a + b * c, "arithmetique"
    if family == 1:
        return f"gcd({a * c}, {b * c})", math.gcd(a * c, b * c), "entiers"
    if family == 2:
        n = 10 + index % 35
        k = 2 + index % 5
        return f"comb({n}, {k})", math.comb(n, k), "combinatoire"
    if family == 3:
        return f"sqrt({a * a})", float(a), "racines"
    if family == 4:
        return f"mean([{a}, {b}, {c}])", statistics.mean([a, b, c]), "statistiques"
    if family == 5:
        return "sin(0) + cos(0)", 1.0, "trigonometrie"
    if family == 6:
        return f"floor({a} / {b})", math.floor(a / b), "arrondis"
    if family == 7:
        value = a * a + b
        return f"isqrt({value})", math.isqrt(value), "entiers"
    if family == 8:
        return f"hypot({a}, {b})", math.hypot(a, b), "geometrie"
    exponent = 1 + index % 30
    return f"log2(2 ** {exponent})", float(exponent), "logarithmes"


def _numeric_value(calculation: dict[str, Any]) -> int | float:
    value = calculation.get("result", calculation.get("value"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"résultat numérique attendu, reçu {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError("résultat non fini")
    return value


def _equivalent(actual: int | float, expected: int | float) -> bool:
    if isinstance(actual, int) and isinstance(expected, int):
        return actual == expected
    return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)


def _verify_security(engine: MathEngine) -> dict[str, Any]:
    probes = [
        "__import__('os').getcwd()",
        "(1).__class__",
        "[x for x in [1, 2, 3]]",
        "open('secret.txt')",
        "lambda x: x",
    ]
    rejected = 0
    for expression in probes:
        try:
            engine.evaluate(expression)
        except (MathEngineError, ValueError, TypeError):
            rejected += 1
    return {
        "probes": len(probes),
        "rejected": rejected,
        "all_rejected": rejected == len(probes),
    }


def run(
    count: int,
    *,
    warmup: int,
    max_latency_samples: int = 50_000,
    measure_memory: bool = False,
) -> dict[str, Any]:
    engine = MathEngine()
    catalog = engine.catalog()
    for index in range(min(warmup, count)):
        expression, _, _ = _case(index)
        engine.evaluate(expression)

    sample_stride = max(1, count // max_latency_samples)
    # Les cas tournent sur dix familles. Un pas premier avec 10 évite que
    # l'échantillon de latence ne mesure qu'une seule famille.
    while sample_stride > 1 and math.gcd(sample_stride, 10) != 1:
        sample_stride += 1
    latencies_ms: list[float] = []
    family_counts: dict[str, int] = {}
    completed = 0
    mismatches = 0
    errors = 0
    first_failures: list[dict[str, Any]] = []

    if measure_memory:
        tracemalloc.start()
        tracemalloc.reset_peak()
    started = time.perf_counter()
    for index in range(count):
        expression, expected, family = _case(index)
        family_counts[family] = family_counts.get(family, 0) + 1
        call_started = time.perf_counter_ns()
        try:
            calculation = engine.evaluate(expression)
            actual = _numeric_value(calculation)
            if not _equivalent(actual, expected):
                mismatches += 1
                if len(first_failures) < 10:
                    first_failures.append(
                        {
                            "index": index,
                            "expression": expression,
                            "expected": expected,
                            "actual": actual,
                            "kind": "mismatch",
                        }
                    )
            else:
                completed += 1
        except Exception as error:  # le rapport doit conserver les erreurs inattendues
            errors += 1
            if len(first_failures) < 10:
                first_failures.append(
                    {
                        "index": index,
                        "expression": expression,
                        "expected": expected,
                        "error": f"{type(error).__name__}: {error}",
                        "kind": "error",
                    }
                )
        finally:
            if index % sample_stride == 0:
                latencies_ms.append((time.perf_counter_ns() - call_started) / 1_000_000)

    elapsed = time.perf_counter() - started
    traced_current = traced_peak = None
    if measure_memory:
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    security = _verify_security(engine)
    catalog_bytes = len(
        json.dumps(
            catalog,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "benchmark": "math-engine-v1",
        "registry_version": catalog.get("version"),
        "catalog_functions": catalog.get("count"),
        "catalog_serialized_bytes": catalog_bytes,
        "python_version": sys.version.split()[0],
        "expressions_requested": count,
        "completed_correctly": completed,
        "mismatches": mismatches,
        "errors": errors,
        "correct_percent": round(100.0 * completed / count, 6),
        "elapsed_seconds": round(elapsed, 6),
        "throughput_expressions_per_second": round(count / elapsed, 2),
        "latency_sample_count": len(latencies_ms),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 6),
            "p50": round(_percentile(latencies_ms, 0.50), 6),
            "p95": round(_percentile(latencies_ms, 0.95), 6),
            "p99": round(_percentile(latencies_ms, 0.99), 6),
            "max": round(max(latencies_ms, default=0.0), 6),
        },
        "families": family_counts,
        "security": security,
        "traced_memory": {
            "enabled": measure_memory,
            "current_bytes": traced_current,
            "peak_bytes": traced_peak,
            "warning": (
                "tracemalloc modifie le debit; comparer la vitesse avec une execution sans --measure-memory."
                if measure_memory
                else None
            ),
        },
        "first_failures": first_failures,
        "memory_writes": 0,
        "note": (
            "Les expressions sont évaluées puis oubliées. Le benchmark ne crée "
            "aucun événement dans la mémoire associative."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mesure le calculateur sans remplir la mémoire associative."
    )
    parser.add_argument("--count", type=_positive_int, default=100_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="mesure les allocations Python, au prix d'un benchmark plus lent",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmup < 0:
        parser.error("--warmup doit être positif ou nul")

    report = run(
        arguments.count,
        warmup=arguments.warmup,
        measure_memory=arguments.measure_memory,
    )
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    print(encoded)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if (
        report["mismatches"] == 0
        and report["errors"] == 0
        and report["security"]["all_rejected"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
