"""Génère le curriculum JSONL d'un modèle dédié à la mémoire externe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.memory_native_curriculum import (  # noqa: E402
    audit_curriculum_isolation,
    build_memory_native_curriculum,
    build_synthetic_memory_curriculum,
    encode_training_jsonl,
    write_training_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument("--count", type=int, default=2_250)
    parser.add_argument(
        "--science-demo-dataset",
        type=Path,
        help=(
            "Démonstration seulement : produit des exemples depuis ce corpus, impropres "
            "à toute évaluation sur les mêmes faits."
        ),
    )
    parser.add_argument(
        "--forbidden-corpus",
        type=Path,
        default=PROJECT_ROOT / "examples" / "science-biographies-v1.json",
        help="Corpus interdit utilisé uniquement pour l'audit de collision après génération.",
    )
    parser.add_argument("--output", type=Path, help="Fichier JSONL; stdout par défaut.")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Manifeste JSON facultatif, sans recopier les exemples.",
    )
    parser.add_argument(
        "--maximum-per-task",
        type=int,
        help="Borne déterministe facultative par type d'exercice.",
    )
    arguments = parser.parse_args()

    if arguments.science_demo_dataset is None:
        if arguments.maximum_per_task is not None:
            parser.error("--maximum-per-task exige --science-demo-dataset")
        result = build_synthetic_memory_curriculum(seed=arguments.seed, count=arguments.count)
        isolation_audit = audit_curriculum_isolation(result, arguments.forbidden_corpus)
        if not isolation_audit["passed"]:
            raise RuntimeError("collision entre le curriculum synthétique et le corpus interdit")
    else:
        result = build_memory_native_curriculum(
            arguments.science_demo_dataset,
            maximum_per_task=arguments.maximum_per_task,
        )
        isolation_audit = None
    if arguments.output is None:
        sys.stdout.write(encode_training_jsonl(result["examples"]))
    else:
        write_training_jsonl(arguments.output, result["examples"])

    if arguments.manifest_output is not None:
        manifest = {key: value for key, value in result.items() if key != "examples"}
        if isolation_audit is not None:
            manifest["isolation_audit"] = isolation_audit
        arguments.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.manifest_output.write_text(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if arguments.output is not None:
        sys.stdout.write(
            json.dumps(
                {
                    "output": str(arguments.output),
                    "example_count": result["example_count"],
                    "task_counts": result["task_counts"],
                    "curriculum_sha256": result.get(
                        "generator_sha256", result.get("facts_sha256")
                    ),
                    "synthetic": result["synthetic"],
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
