#!/usr/bin/env python3
"""Mesure hors ligne les longueurs réelles des exemples MAT-LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


MAX_DATASET_BYTES = 256 * 1024 * 1024
MAX_EXAMPLES = 100_000


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"jeu JSONL introuvable: {resolved}")
    if resolved.stat().st_size > MAX_DATASET_BYTES:
        raise ValueError(f"jeu JSONL supérieur à {MAX_DATASET_BYTES} octets")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(rows) >= MAX_EXAMPLES:
                raise ValueError(f"jeu JSONL supérieur à {MAX_EXAMPLES} exemples")
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
                raise ValueError(f"ligne {line_number}: messages absent ou invalide")
            rows.append(value)
    if not rows:
        raise ValueError("jeu JSONL vide")
    return rows


def _measure(tokenizer: Any, path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    lengths: list[tuple[int, int, int]] = []
    by_task: dict[str, list[tuple[int, int, int]]] = {}
    for row in rows:
        messages = row["messages"]
        if len(messages) < 3:
            raise ValueError("chaque exemple doit contenir system, user et assistant")
        full = len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        prefix = len(
            tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        target = len(
            tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
        )
        current = (full, prefix, target)
        lengths.append(current)
        by_task.setdefault(str(row.get("task", "unknown")), []).append(current)
    totals = [value[0] for value in lengths]
    return {
        "path": str(path.expanduser().resolve()),
        "example_count": len(rows),
        "full_tokens": {
            "maximum": max(totals),
            "p95": _percentile(totals, 0.95),
            "over_1024": sum(value > 1_024 for value in totals),
            "over_1536": sum(value > 1_536 for value in totals),
            "over_2048": sum(value > 2_048 for value in totals),
        },
        "prefix_tokens_maximum": max(value[1] for value in lengths),
        "target_tokens_maximum": max(value[2] for value in lengths),
        "task_maximums": {
            task: {
                "full_tokens": max(value[0] for value in values),
                "target_tokens": max(value[2] for value in values),
            }
            for task, values in sorted(by_task.items())
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("datasets", nargs="+", type=Path)
    arguments = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        arguments.model.expanduser().resolve(),
        local_files_only=True,
        trust_remote_code=False,
    )
    result = {
        "schema_version": "matlm-token-length-audit-v1",
        "network": "offline",
        "datasets": [_measure(tokenizer, path) for path in arguments.datasets],
    }
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
