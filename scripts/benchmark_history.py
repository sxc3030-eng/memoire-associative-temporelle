"""Execute le laboratoire historique dans des bases temporaires isolees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.history_stress_lab import (  # noqa: E402
    HistoryStressConfig,
    run_history_stress,
)


def _bounded_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--count doit etre un entier") from error
    if not 2 <= parsed <= 10_000:
        raise argparse.ArgumentTypeError("--count doit etre compris entre 2 et 10000")
    return parsed


def _bounded_seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--seed doit etre un entier") from error
    if not 0 <= parsed <= 2**63 - 1:
        raise argparse.ArgumentTypeError(
            "--seed doit etre compris entre 0 et 2^63 - 1"
        )
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Genere un historique fictif, l'injecte dans deux bases temporaires "
            "et mesure rappel, provenance et deduplication."
        )
    )
    parser.add_argument("--count", type=_bounded_count, default=100)
    parser.add_argument("--seed", type=_bounded_seed, default=20_260_721)
    arguments = parser.parse_args()

    report = run_history_stress(
        HistoryStressConfig(
            event_count=arguments.count,
            seed=arguments.seed,
            max_queries=min(500, max(20, arguments.count)),
        )
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    pipeline = report["pipeline"]
    isolation = report["isolation"]
    return 0 if (
        pipeline["drained"]
        and pipeline["failed"] == 0
        and pipeline["deduplication_exact"]
        and pipeline["memory_event_count_exact"]
        and isolation["temporary_storage_removed_after_run"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
