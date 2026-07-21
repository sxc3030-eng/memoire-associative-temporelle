"""Importe localement un dump YAGO 4.5 dans un SQLite de reference."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.yago_import import (  # noqa: E402
    ImportLimits,
    YAGOImportError,
    import_yago,
    iter_memory_batches,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("la valeur doit etre positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="fichier .ttl/.nt/.ntx ou ZIP YAGO tiny local")
    parser.add_argument("--db", type=Path, help="SQLite de reference (requis sauf --dry-run)")
    parser.add_argument("--release", default="4.5.0.2", help="version inscrite dans la provenance")
    parser.add_argument("--dry-run", action="store_true", help="analyse sans aucune ecriture")
    parser.add_argument("--strict", action="store_true", help="arrete au premier element non pris en charge")
    parser.add_argument("--max-triples", type=_positive_int, default=10_000_000)
    parser.add_argument("--max-statements", type=_positive_int, default=12_000_000)
    parser.add_argument("--max-prefixes", type=_positive_int, default=4_096)
    parser.add_argument("--batch-size", type=_positive_int, default=2_000)
    parser.add_argument("--max-archive-mib", type=_positive_int, default=512)
    parser.add_argument("--max-member-mib", type=_positive_int, default=4 * 1024)
    parser.add_argument("--max-total-mib", type=_positive_int, default=8 * 1024)
    parser.add_argument("--max-ratio", type=float, default=250.0)
    parser.add_argument("--report", type=Path, help="copie JSON du rapport")
    parser.add_argument(
        "--memory-jsonl",
        type=Path,
        help="export optionnel des lots compatibles avec une memoire de reference",
    )
    args = parser.parse_args(argv)

    if not args.dry_run and args.db is None:
        parser.error("--db est requis sauf avec --dry-run")
    if args.memory_jsonl is not None and (args.dry_run or args.db is None):
        parser.error("--memory-jsonl exige un import reel avec --db")

    limits = replace(
        ImportLimits(),
        max_archive_bytes=args.max_archive_mib * 1024 * 1024,
        max_member_bytes=args.max_member_mib * 1024 * 1024,
        max_total_uncompressed_bytes=args.max_total_mib * 1024 * 1024,
        max_compression_ratio=args.max_ratio,
        max_triples=args.max_triples,
        max_statements=args.max_statements,
        max_prefixes_per_member=args.max_prefixes,
        batch_size=args.batch_size,
    )
    try:
        report = import_yago(
            args.source,
            args.db,
            release=args.release,
            dry_run=args.dry_run,
            strict=args.strict,
            limits=limits,
        )
        if args.memory_jsonl is not None:
            args.memory_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.memory_jsonl.open("w", encoding="utf-8", newline="\n") as output:
                for batch in iter_memory_batches(args.db, batch_size=args.batch_size):
                    for record in batch:
                        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            report["memory_jsonl"] = str(args.memory_jsonl.resolve())
    except (OSError, ValueError, YAGOImportError) as error:
        parser.exit(2, f"Import YAGO refuse: {error}\n")

    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(encoded)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
