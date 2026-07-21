"""Mesure reproductible du pipeline sans modifier la memoire principale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent import MemoryEngine, MemoryPipeline  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def run_benchmark(count: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="memory-pipeline-benchmark-") as directory:
        root = Path(directory)
        memory_path = root / "memory.sqlite3"
        queue_path = root / "injection.sqlite3"

        seed = MemoryEngine(memory_path)
        try:
            seed.observe(
                "Le repere du lecteur est une etoile bleue",
                idempotency_key="benchmark:reader-seed",
            )
        finally:
            seed.close()

        pipeline = MemoryPipeline(
            memory_path,
            queue_path,
            batch_size=8,
            poll_interval=0.005,
        )
        read_latencies: list[float] = []
        read_errors: list[str] = []
        stop_reader = threading.Event()

        def read_continuously() -> None:
            while not stop_reader.is_set():
                started = time.perf_counter()
                try:
                    pipeline.recall("etoile bleue", top_k=3)
                except Exception as error:  # pragma: no cover - reported in output
                    if len(read_errors) < 5:
                        read_errors.append(f"{type(error).__name__}: {error}")
                if len(read_latencies) < 10_000:
                    read_latencies.append((time.perf_counter() - started) * 1_000)
                stop_reader.wait(0.002)

        reader = threading.Thread(target=read_continuously, daemon=True)
        enqueue_latencies: list[float] = []
        submitted: list[tuple[str, str]] = []
        try:
            reader.start()
            started_all = time.perf_counter()
            try:
                for index in range(count):
                    if index and index % 5 == 0:
                        text, key = submitted[-1]
                    else:
                        text = (
                            f"Observation experimentale {index}: groupe {index % 17}, "
                            f"valeur {index * 13 % 997}."
                        )
                        key = f"benchmark:{index}"
                        submitted.append((text, key))
                    started = time.perf_counter()
                    pipeline.enqueue(
                        text,
                        episode_id=f"benchmark-episode-{key.removeprefix('benchmark:')}",
                        context={"category": "benchmark"},
                        source={"type": "observed", "origin": "pipeline_benchmark"},
                        idempotency_key=key,
                    )
                    enqueue_latencies.append((time.perf_counter() - started) * 1_000)

                enqueue_finished = time.perf_counter()
                drained = pipeline.wait_until_idle(timeout=max(30.0, count * 0.2))
                finished_all = time.perf_counter()
            finally:
                stop_reader.set()
                reader.join(timeout=2)

            stats = pipeline.stats()
            queue = stats["queue"]
            memory = stats["memory"]
            unique = int(queue["total"])
            elapsed = finished_all - started_all
            return {
                "submitted": count,
                "unique_after_deduplication": unique,
                "completed": int(queue["completed"]),
                "failed": int(queue["failed"]),
                "deduplicated": int(queue["deduplicated_requests"]),
                "deduplication_percent": round(
                    100.0 * int(queue["deduplicated_requests"]) / count, 2
                ),
                "drained": drained,
                "enqueue_seconds": round(enqueue_finished - started_all, 6),
                "total_seconds": round(elapsed, 6),
                "consolidated_per_second": (
                    round(int(queue["completed"]) / elapsed, 2) if elapsed else None
                ),
                "enqueue_latency_ms": {
                    "mean": round(statistics.fmean(enqueue_latencies), 4),
                    "p50": round(percentile(enqueue_latencies, 0.50), 4),
                    "p95": round(percentile(enqueue_latencies, 0.95), 4),
                },
                "recall_during_writes_ms": {
                    "samples": len(read_latencies),
                    "sample_limit": 10_000,
                    "interval_ms": 2,
                    "p50": round(percentile(read_latencies, 0.50), 4),
                    "p95": round(percentile(read_latencies, 0.95), 4),
                    "p99": round(percentile(read_latencies, 0.99), 4),
                    "errors": read_errors,
                },
                "seed_events": 1,
                "benchmark_events": max(0, int(memory["events"]) - 1),
                "memory_events_total": int(memory["events"]),
                "memory_size_bytes_including_wal_shm": memory.get("database_size_bytes"),
                "queue_size_bytes_including_wal_shm": queue.get("database_size_bytes"),
                "reader_writer_separated": bool(stats["reader_writer_separated"]),
            }
        finally:
            stop_reader.set()
            if reader.is_alive():
                reader.join(timeout=2)
            pipeline.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Mesure le pipeline de memoire local")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.count <= 10_000:
        parser.error("--count doit etre compris entre 1 et 10000")
    print(json.dumps(run_benchmark(args.count), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
