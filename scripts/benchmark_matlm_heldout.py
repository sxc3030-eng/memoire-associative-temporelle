#!/usr/bin/env python3
"""Évalue hors ligne Granite nu et/ou Granite + LoRA sur le lot held-out."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.matlm_heldout_benchmark import (  # noqa: E402
    DEFAULT_LIMIT,
    MATLMHeldoutBenchmarkError,
    benchmark_plan,
    build_benchmark_arms,
    load_balanced_heldout,
    run_heldout_benchmark,
    write_atomic_report,
)


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "training-data" / "matlm-dev-v8.jsonl",
        help="JSONL held-out local.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "matlm-heldout-report.json",
        help="Rapport JSON atomique.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--compare",
        choices=("base", "adapter", "both"),
        help="Défaut: adapter si --adapter est fourni, sinon base.",
    )
    parser.add_argument("--adapter", type=Path, help="Dossier LoRA local.")
    parser.add_argument(
        "--base-model",
        required=True,
        help="Dossier Granite local ou identifiant déjà présent dans le cache local.",
    )
    parser.add_argument(
        "--load-mode", choices=("auto", "qlora-nf4", "bf16"), default="auto"
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument(
        "--replace-report",
        action="store_true",
        help="Remplace explicitement un rapport du même nom.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valide données, chemins et quotas sans charger le modèle.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    compare = arguments.compare or ("adapter" if arguments.adapter else "base")
    try:
        dataset = arguments.dataset.expanduser().resolve()
        report_path = arguments.report.expanduser().resolve()
        if dataset == report_path:
            raise MATLMHeldoutBenchmarkError(
                "le rapport ne peut pas remplacer le jeu held-out"
            )
        selection = load_balanced_heldout(dataset, limit=arguments.limit)
        arms = build_benchmark_arms(
            compare=compare,
            base_model=arguments.base_model,
            adapter_path=arguments.adapter,
            load_mode=arguments.load_mode,
            device_index=arguments.device_index,
            max_input_tokens=arguments.max_input_tokens,
            max_new_tokens=arguments.max_new_tokens,
            seed=arguments.seed,
        )
        if arguments.dry_run:
            sys.stdout.write(_json(benchmark_plan(selection, arms)) + "\n")
            return 0

        result = run_heldout_benchmark(selection, arms)
        output = write_atomic_report(
            report_path,
            result,
            replace=arguments.replace_report,
        )
        report_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        receipt = {
            "schema_version": result["schema_version"],
            "status": result["status"],
            "report": str(output),
            "report_sha256": report_sha256,
            "selection_sha256": result["dataset"]["selection_sha256"],
            "arms": [
                {
                    "name": arm["name"],
                    "status": arm["status"],
                    "attempted": arm["global_metrics"]["attempted"],
                    "contract_valid_rate": arm["global_metrics"]["rates"][
                        "contract_valid"
                    ],
                    "all_required_exact_rate": arm["global_metrics"]["rates"][
                        "all_required_exact"
                    ],
                }
                for arm in result["arms"]
            ],
        }
        sys.stdout.write(_json(receipt) + "\n")
        return 0
    except MATLMHeldoutBenchmarkError as error:
        message = " ".join(str(error).replace("\x00", " ").split())[:1_000]
        sys.stderr.write(f"Erreur benchmark MAT-LM: {message}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
