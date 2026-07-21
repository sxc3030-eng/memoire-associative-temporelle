from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent import (  # noqa: E402
    BackgroundConsolidator,
    DurableInjectionQueue,
    IdempotencyConflictError,
    MemoryEngine,
    MemoryIdempotencyConflictError,
    MemoryPipeline,
)


class DurableInjectionQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.queue_path = self.root / "injection.sqlite3"
        self.queue = DurableInjectionQueue(self.queue_path)

    def tearDown(self) -> None:
        self.queue.close()
        self.temporary_directory.cleanup()

    def test_enqueue_is_durable_idempotent_and_detects_conflicts(self) -> None:
        first = self.queue.enqueue(
            "Mars est une planete tellurique",
            idempotency_key="nasa:mars:v1",
            context={"collection": "planetes"},
        )
        duplicate = self.queue.enqueue(
            "Mars est une planete tellurique",
            idempotency_key="nasa:mars:v1",
            context={"collection": "planetes"},
        )

        self.assertTrue(first["created"])
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(first["job_id"], duplicate["job_id"])
        self.assertEqual(self.queue.stats()["pending"], 1)
        self.assertEqual(self.queue.stats()["enqueue_requests"], 2)
        self.assertEqual(self.queue.stats()["deduplicated_requests"], 1)
        with self.assertRaises(IdempotencyConflictError):
            self.queue.enqueue(
                "Mars est une geante gazeuse",
                idempotency_key="nasa:mars:v1",
                context={"collection": "planetes"},
            )
        self.assertEqual(self.queue.stats()["enqueue_requests"], 3)
        self.assertEqual(self.queue.stats()["deduplicated_requests"], 1)

        self.queue.close()
        reopened = DurableInjectionQueue(self.queue_path)
        try:
            persisted = reopened.get(first["job_id"])
            self.assertIsNotNone(persisted)
            self.assertEqual(persisted["state"], "pending")
            self.assertEqual(persisted["context"], {"collection": "planetes"})
            self.assertEqual(reopened.stats()["enqueue_requests"], 3)
            self.assertEqual(reopened.stats()["deduplicated_requests"], 1)
        finally:
            reopened.close()
        self.queue = DurableInjectionQueue(self.queue_path)

    def test_claim_is_bounded_and_failures_have_a_finite_budget(self) -> None:
        for number in range(3):
            self.queue.enqueue(
                f"observation {number}",
                idempotency_key=f"source:{number}",
                max_attempts=2,
            )
        engine = MemoryEngine(self.root / "memory.sqlite3")
        original_observe = engine.observe

        def always_fails(*args, **kwargs):
            raise RuntimeError("indisponible")

        engine.observe = always_fails  # type: ignore[method-assign]
        worker = BackgroundConsolidator(
            self.queue,
            engine,
            batch_size=2,
            retry_base_delay=0,
            retry_max_delay=0,
        )
        try:
            self.assertEqual(worker.run_once(), 1)
            first_pass = self.queue.stats()
            self.assertEqual(first_pass["pending"], 3)
            self.assertEqual(first_pass["attempts"], 1)

            self.assertEqual(worker.run_once(), 1)
            second_pass = self.queue.stats()
            self.assertEqual(second_pass["failed"], 1)
            self.assertEqual(second_pass["pending"], 2)
            self.assertEqual(second_pass["attempts"], 2)
            for _ in range(4):
                self.assertEqual(worker.run_once(), 1)
            terminal = self.queue.stats()
            self.assertEqual(terminal["failed"], 3)
            self.assertEqual(terminal["unfinished"], 0)
        finally:
            engine.observe = original_observe  # type: ignore[method-assign]
            worker.close()
            engine.close()

    def test_startup_requeues_interrupted_processing(self) -> None:
        queued = self.queue.enqueue(
            "souvenir interrompu",
            idempotency_key="interruption:1",
            max_attempts=2,
        )
        claimed = self.queue.claim(batch_size=1, worker_id="ancien-worker")
        self.assertEqual(claimed[0]["state"], "processing")

        self.queue.close()
        reopened = DurableInjectionQueue(self.queue_path)
        try:
            recovery = reopened.recover_processing(stale_after_seconds=0)
            self.assertEqual(recovery, {"recovered": 1, "failed": 0})
            self.assertEqual(reopened.get(queued["job_id"])["state"], "pending")
        finally:
            reopened.close()
        self.queue = DurableInjectionQueue(self.queue_path)

    def test_exhausted_interrupted_job_becomes_terminal(self) -> None:
        queued = self.queue.enqueue(
            "dernier essai",
            idempotency_key="interruption:terminal",
            max_attempts=1,
        )
        self.queue.claim(batch_size=1, worker_id="ancien-worker")

        recovery = self.queue.recover_processing(stale_after_seconds=0)

        self.assertEqual(recovery, {"recovered": 1, "failed": 0})
        recovered = self.queue.get(queued["job_id"])
        self.assertEqual(recovered["state"], "pending")
        self.assertEqual(recovered["attempts"], 0)

    def test_retry_after_observe_does_not_duplicate_memory(self) -> None:
        queued = self.queue.enqueue(
            "alpha beta gamma",
            idempotency_key="crash-window:1",
            max_attempts=3,
        )
        engine = MemoryEngine(self.root / "memory.sqlite3")
        worker = BackgroundConsolidator(
            self.queue,
            engine,
            batch_size=1,
            retry_base_delay=0,
            retry_max_delay=0,
            worker_id="nouveau-worker",
        )
        try:
            claimed = self.queue.claim(batch_size=1, worker_id="ancien-worker")[0]
            engine.observe(
                claimed["text"],
                idempotency_key=claimed["idempotency_key"],
                source="observed",
            )
            # Simule un arret apres l'ecriture memoire mais avant l'acquittement.
            self.queue.recover_processing(stale_after_seconds=0)

            self.assertEqual(worker.run_once(), 1)
            self.assertEqual(self.queue.get(queued["job_id"])["state"], "completed")
            self.assertEqual(engine.stats()["events"], 1)
            self.assertTrue(self.queue.get(queued["job_id"])["result"]["duplicate"])
        finally:
            worker.close()
            engine.close()

    def test_pipeline_reuses_source_key_already_present_in_memory(self) -> None:
        queued = self.queue.enqueue(
            "Jupiter est une geante gazeuse",
            idempotency_key="catalogue:jupiter:v1",
        )
        engine = MemoryEngine(self.root / "memory.sqlite3")
        engine.observe(
            "Jupiter est une geante gazeuse",
            idempotency_key="catalogue:jupiter:v1",
            source="observed",
        )
        worker = BackgroundConsolidator(self.queue, engine, batch_size=1)
        try:
            self.assertEqual(worker.run_once(), 1)
            result = self.queue.get(queued["job_id"])["result"]
            self.assertTrue(result["duplicate"])
            self.assertEqual(engine.stats()["events"], 1)
        finally:
            worker.close()
            engine.close()

    def test_stats_report_queue_lag(self) -> None:
        self.queue.enqueue("en attente", idempotency_key="lag:1")

        stats = self.queue.stats()

        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["ready_pending"], 1)
        self.assertGreaterEqual(stats["lag_seconds"], 0)
        self.assertIsNotNone(stats["oldest_unfinished_at"])

    def test_live_lease_is_not_recovered(self) -> None:
        queued = self.queue.enqueue("travail actif", idempotency_key="lease:1")
        self.queue.claim(batch_size=1, worker_id="worker-actif")

        self.assertEqual(
            self.queue.recover_processing(stale_after_seconds=60),
            {"recovered": 0, "failed": 0},
        )
        self.assertEqual(self.queue.get(queued["job_id"])["state"], "processing")
        self.assertEqual(
            self.queue.recover_processing(stale_after_seconds=0),
            {"recovered": 1, "failed": 0},
        )

    def test_transient_failure_preserves_temporal_order(self) -> None:
        self.queue.enqueue(
            "alpha", idempotency_key="order:a", episode_id="ordered-episode"
        )
        self.queue.enqueue(
            "beta", idempotency_key="order:b", episode_id="ordered-episode"
        )
        engine = MemoryEngine(self.root / "ordered.sqlite3")
        original_observe = engine.observe
        failed_once = False

        def fail_alpha_once(text, *args, **kwargs):
            nonlocal failed_once
            if text == "alpha" and not failed_once:
                failed_once = True
                raise RuntimeError("transient")
            return original_observe(text, *args, **kwargs)

        engine.observe = fail_alpha_once  # type: ignore[method-assign]
        worker = BackgroundConsolidator(
            self.queue,
            engine,
            batch_size=2,
            retry_base_delay=0,
            retry_max_delay=0,
        )
        try:
            self.assertEqual(worker.run_once(), 1)
            self.assertEqual(engine.stats()["events"], 0)
            self.assertEqual(worker.run_once(), 2)
            prediction = engine.predict("alpha", top_k=1)
            self.assertEqual(prediction[0]["concept"], "beta")
        finally:
            engine.observe = original_observe  # type: ignore[method-assign]
            worker.close()
            engine.close()


class MemoryPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_background_pipeline_uses_distinct_reader_and_writer(self) -> None:
        pipeline = MemoryPipeline(
            self.root / "memory.sqlite3",
            self.root / "queue.sqlite3",
            batch_size=2,
            poll_interval=0.01,
            retry_base_delay=0,
            retry_max_delay=0,
        )
        try:
            queued = pipeline.enqueue(
                "Neptune est une geante de glace",
                idempotency_key="nasa:neptune:v1",
                episode_id="planetes",
                source="observed",
            )

            self.assertTrue(pipeline.wait_until_idle(timeout=5))
            self.assertEqual(pipeline.queue.get(queued["job_id"])["state"], "completed")
            self.assertTrue(pipeline.recall("Neptune glace"))
            stats = pipeline.stats()
            self.assertTrue(stats["reader_writer_separated"])
            self.assertEqual(stats["memory"]["events"], 1)
            self.assertIsNone(stats["worker"]["fatal_error"])
        finally:
            pipeline.close()
            pipeline.close()

        with self.assertRaises(RuntimeError):
            pipeline.enqueue("ferme", idempotency_key="closed:1")

    def test_terminal_ticket_is_redelivered_after_forget(self) -> None:
        pipeline = MemoryPipeline(
            self.root / "memory.sqlite3",
            self.root / "queue.sqlite3",
            poll_interval=0.01,
        )
        try:
            first = pipeline.enqueue(
                "Vega est une etoile brillante",
                idempotency_key="catalogue:vega:v1",
                source="observed",
            )
            self.assertTrue(pipeline.wait_until_idle(timeout=5))
            first_event = pipeline.queue.get(first["job_id"])["result"]["event_id"]
            self.assertTrue(pipeline.reader_engine.forget(first_event)["forgotten"])

            retried = pipeline.enqueue(
                "Vega est une etoile brillante",
                idempotency_key="catalogue:vega:v1",
                source="observed",
                retry_terminal=True,
            )
            self.assertTrue(retried["retried"])
            self.assertTrue(pipeline.wait_until_idle(timeout=5))
            new_event = pipeline.queue.get(first["job_id"])["result"]["event_id"]
            self.assertNotEqual(first_event, new_event)
            self.assertTrue(pipeline.reader_engine.event_exists(new_event))
        finally:
            pipeline.close()

    def test_queue_refuses_another_memory_identity(self) -> None:
        queue_path = self.root / "bound-queue.sqlite3"
        first = MemoryPipeline(
            self.root / "first-memory.sqlite3", queue_path, autostart=False
        )
        try:
            first.enqueue("memoire liee", idempotency_key="bound:1")
        finally:
            first.close()

        with self.assertRaisesRegex(RuntimeError, "autre base memoire"):
            MemoryPipeline(
                self.root / "second-memory.sqlite3", queue_path, autostart=False
            )

    def test_test_run_enqueue_is_atomic_on_mid_batch_failure(self) -> None:
        pipeline = MemoryPipeline(
            self.root / "atomic-memory.sqlite3",
            self.root / "atomic-queue.sqlite3",
            autostart=False,
        )
        try:
            pipeline.queue._connection.execute(  # noqa: SLF001 - fault injection
                """
                CREATE TRIGGER reject_third_test_ticket
                BEFORE INSERT ON injection_jobs
                WHEN NEW.idempotency_key LIKE 'pipeline-test:%:3'
                BEGIN
                    SELECT RAISE(ABORT, 'injected batch failure');
                END
                """
            )
            with self.assertRaises(sqlite3.IntegrityError):
                pipeline.enqueue_test_run("abcdef123456", count=5)

            self.assertIsNone(pipeline.queue.get_test_run("abcdef123456"))
            queue_stats = pipeline.queue.stats()
            self.assertEqual(queue_stats["total"], 0)
            self.assertEqual(queue_stats["enqueue_requests"], 0)
        finally:
            pipeline.close()

    def test_terminal_test_run_is_resumed_and_cleaned_after_restart(self) -> None:
        memory_path = self.root / "restart-memory.sqlite3"
        queue_path = self.root / "restart-queue.sqlite3"
        first = MemoryPipeline(
            memory_path,
            queue_path,
            batch_size=1,
            poll_interval=0.01,
            autostart=False,
        )
        run_id = "abc123def456"
        try:
            first.worker.maintenance_callback = None
            first.enqueue_test_run(run_id, count=3)
            first.start()
            self.assertTrue(first.wait_until_idle(timeout=5))
            self.assertEqual(first.writer_engine.stats()["events"], 3)
            self.assertEqual(first.queue.get_test_run(run_id)["state"], "active")
        finally:
            first.close()

        resumed = MemoryPipeline(
            memory_path,
            queue_path,
            batch_size=1,
            poll_interval=0.01,
        )
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                run = resumed.queue.get_test_run(run_id)
                if run is not None and run["state"] == "cleaned":
                    break
                time.sleep(0.01)
            else:
                self.fail("Le nettoyage durable n'a pas repris au redemarrage")
            self.assertEqual(run["successful_count"], 3)
            self.assertEqual(run["forgotten_count"], 3)
            self.assertEqual(resumed.writer_engine.stats()["events"], 0)
            self.assertEqual(resumed.queue.stats()["total"], 0)
        finally:
            resumed.close()

    def test_partial_test_failure_waits_for_all_jobs_then_cleans_every_ticket(self) -> None:
        pipeline = MemoryPipeline(
            self.root / "partial-memory.sqlite3",
            self.root / "partial-queue.sqlite3",
            batch_size=1,
            poll_interval=0.01,
            retry_base_delay=0,
            retry_max_delay=0,
            autostart=False,
        )
        original_observe = pipeline.writer_engine.observe

        def fail_second(text: str, **kwargs):
            if str(kwargs.get("episode_id", "")).endswith("-2"):
                raise MemoryIdempotencyConflictError("echec terminal injecte")
            return original_observe(text, **kwargs)

        pipeline.writer_engine.observe = fail_second  # type: ignore[method-assign]
        try:
            run_id = "123456abcdef"
            pipeline.enqueue_test_run(run_id, count=5)
            pipeline.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                run = pipeline.queue.get_test_run(run_id)
                if run is not None and run["state"] == "cleaned":
                    break
                time.sleep(0.01)
            else:
                self.fail("Le run partiellement echoue n'a pas ete nettoye")
            self.assertEqual(run["successful_count"], 4)
            self.assertEqual(run["failed_count"], 1)
            self.assertEqual(run["forgotten_count"], 4)
            self.assertEqual(pipeline.writer_engine.stats()["events"], 0)
            self.assertEqual(pipeline.queue.stats()["total"], 0)
        finally:
            pipeline.writer_engine.observe = original_observe  # type: ignore[method-assign]
            pipeline.close()

    def test_cleanup_finds_event_committed_before_failed_ticket_ack(self) -> None:
        pipeline = MemoryPipeline(
            self.root / "crash-clean-memory.sqlite3",
            self.root / "crash-clean-queue.sqlite3",
            autostart=False,
        )
        try:
            run_id = "fedcba654321"
            created = pipeline.enqueue_test_run(run_id, count=1)
            job = pipeline.queue.claim(batch_size=1, worker_id="fault-worker")[0]
            observed = pipeline.writer_engine.observe(
                job["text"],
                episode_id=job["episode_id"],
                context=job["context"],
                source=job["source"],
                idempotency_key=job["idempotency_key"],
            )
            pipeline.queue.fail(
                job["job_id"],
                worker_id="fault-worker",
                error="queue acknowledgement lost",
                permanent=True,
            )
            self.assertTrue(pipeline.writer_engine.event_exists(observed["event_id"]))

            cleaned = pipeline.cleanup_terminal_test_runs()
            self.assertEqual(len(cleaned), 1)
            self.assertFalse(pipeline.writer_engine.event_exists(observed["event_id"]))
            run = pipeline.queue.get_test_run(run_id)
            self.assertEqual(run["state"], "cleaned")
            self.assertEqual(run["successful_count"], 0)
            self.assertEqual(run["failed_count"], 1)
            self.assertEqual(run["forgotten_count"], 1)
            self.assertIsNotNone(created["jobs"][0]["job_id"])
        finally:
            pipeline.close()


if __name__ == "__main__":
    unittest.main()
