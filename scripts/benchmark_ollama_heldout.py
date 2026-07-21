#!/usr/bin/env python3
"""Compare un modele Ollama local au lot held-out MAT-LM, sans client HTTP."""

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
    load_balanced_heldout,
    write_atomic_report,
)
from memory_agent.ollama_cli_benchmark import (  # noqa: E402
    DEFAULT_MODEL,
    OllamaCLIBenchmarkError,
    OllamaCLIConfig,
    benchmark_plan,
    run_ollama_cli_benchmark,
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
        help="JSONL held-out local, identique a celui utilise par MAT-LM.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen-ollama-heldout.json",
        help="Rapport JSON atomique sans donnees brutes.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Tag deja installe dans Ollama; aucun telechargement n'est tente.",
    )
    parser.add_argument(
        "--ollama-executable",
        default="ollama",
        help="Nom ou chemin de l'executable local.",
    )
    parser.add_argument("--case-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--preflight-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--total-timeout-seconds", type=float, default=14_400.0)
    parser.add_argument("--max-prompt-bytes", type=int, default=1_000_000)
    parser.add_argument("--max-stdout-bytes", type=int, default=262_144)
    parser.add_argument("--max-stderr-bytes", type=int, default=65_536)
    parser.add_argument(
        "--replace-report",
        action="store_true",
        help="Remplace explicitement un rapport du meme nom.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valide le jeu et les limites sans chercher Ollama ni lancer de modele.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        dataset = arguments.dataset.expanduser().resolve()
        report_path = arguments.report.expanduser().resolve()
        if dataset == report_path:
            raise OllamaCLIBenchmarkError(
                "le rapport ne peut pas remplacer le jeu held-out"
            )
        selection = load_balanced_heldout(dataset, limit=arguments.limit)
        config = OllamaCLIConfig(
            model=arguments.model,
            executable=arguments.ollama_executable,
            case_timeout_seconds=arguments.case_timeout_seconds,
            preflight_timeout_seconds=arguments.preflight_timeout_seconds,
            stop_timeout_seconds=arguments.stop_timeout_seconds,
            total_timeout_seconds=arguments.total_timeout_seconds,
            max_prompt_bytes=arguments.max_prompt_bytes,
            max_stdout_bytes=arguments.max_stdout_bytes,
            max_stderr_bytes=arguments.max_stderr_bytes,
        )
        if arguments.dry_run:
            sys.stdout.write(_json(benchmark_plan(selection, config)) + "\n")
            return 0

        result = run_ollama_cli_benchmark(selection, config)
        output = write_atomic_report(
            report_path,
            result,
            replace=arguments.replace_report,
        )
        receipt = {
            "schema_version": result["schema_version"],
            "status": result["status"],
            "report": str(output),
            "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "selection_sha256": result["dataset"]["selection_sha256"],
            "model": result["model"]["tag"],
            "attempted": result["global_metrics"]["attempted"],
            "contract_valid_rate": result["global_metrics"]["rates"][
                "contract_valid"
            ],
            "content_core_exact_rate": result["global_metrics"]["rates"][
                "content_core_exact"
            ],
        }
        sys.stdout.write(_json(receipt) + "\n")
        return 0
    except (OllamaCLIBenchmarkError, MATLMHeldoutBenchmarkError) as error:
        message = " ".join(str(error).replace("\x00", " ").split())[:1_000]
        sys.stderr.write(f"Erreur benchmark Ollama CLI: {message}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
