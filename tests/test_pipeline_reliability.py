from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent import (  # noqa: E402
    DurableInjectionQueue,
    IdempotencyConflictError,
    MemoryPipeline,
    QueueStateError,
)
from memory_agent.server import (  # noqa: E402
    MAX_CONVERSATION_EPISODE_EVENTS,
    MemoryHTTPServer,
)


def wait_for_test_run(
    pipeline: MemoryPipeline,
    run_id: str,
    *,
    state: str = "cleaned",
    timeout: float = 5.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summary = pipeline.queue.get_test_run(run_id)
        if summary is not None and summary["state"] == state:
            return summary
        time.sleep(0.01)
    raise AssertionError(f"Le run {run_id} n'a pas atteint l'etat {state}")


class DurableTestRunReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_enqueue_test_run_rolls_back_every_row_on_mid_batch_collision(self) -> None:
        queue = DurableInjectionQueue(self.root / "queue.sqlite3")
        run_id = "abc123abc123"
        collision_key = f"pipeline-test:{run_id}:2"
        try:
            existing = queue.enqueue(
                "ticket preexistant",
                idempotency_key=collision_key,
                source="observed",
            )

            with self.assertRaises((sqlite3.IntegrityError, IdempotencyConflictError)):
                queue.enqueue_test_run(run_id, count=3)

            self.assertIsNone(queue.get_test_run(run_id))
            self.assertIsNone(
                queue.get_by_idempotency_key(f"pipeline-test:{run_id}:1")
            )
            self.assertIsNone(
                queue.get_by_idempotency_key(f"pipeline-test:{run_id}:3")
            )
            self.assertEqual(
                queue.get_by_idempotency_key(collision_key)["job_id"],
                existing["job_id"],
            )
            stats = queue.stats()
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["enqueue_requests"], 1)
        finally:
            queue.close()

    def test_server_side_maintenance_cleans_and_keeps_exact_summary_after_purge(
        self,
    ) -> None:
        pipeline = MemoryPipeline(
            self.root / "memory.sqlite3",
            self.root / "queue.sqlite3",
            batch_size=3,
            poll_interval=0.01,
            retry_base_delay=0,
            retry_max_delay=0,
        )
        run_id = "def456def456"
        try:
            created = pipeline.enqueue_test_run(run_id, count=3)
            job_ids = [job["job_id"] for job in created["jobs"]]

            cleaned = wait_for_test_run(pipeline, run_id)

            self.assertEqual(cleaned["state"], "cleaned")
            self.assertEqual(cleaned["expected_count"], 3)
            self.assertEqual(cleaned["successful_count"], 3)
            self.assertEqual(cleaned["failed_count"], 0)
            self.assertEqual(cleaned["forgotten_count"], 3)
            self.assertEqual(cleaned["terminal_count"], 3)
            self.assertEqual(cleaned["job_ids"], job_ids)
            self.assertTrue(
                all(job["state"] == "purged" for job in cleaned["jobs"])
            )
            self.assertEqual(pipeline.queue.get_many(job_ids), [])
            self.assertEqual(pipeline.queue.stats()["total"], 0)
            self.assertEqual(pipeline.writer_engine.stats()["events"], 0)
        finally:
            pipeline.close()

    def test_cleanup_finds_committed_event_even_when_its_ticket_is_failed(self) -> None:
        pipeline = MemoryPipeline(
            self.root / "memory.sqlite3",
            self.root / "queue.sqlite3",
            autostart=False,
        )
        run_id = "fed987fed987"
        try:
            created = pipeline.enqueue_test_run(run_id, count=1)
            job = created["jobs"][0]
            claimed = pipeline.queue.claim(
                batch_size=1, worker_id="writer-before-crash"
            )[0]
            observed = pipeline.writer_engine.observe(
                claimed["text"],
                episode_id=claimed["episode_id"],
                context=claimed["context"],
                source=claimed["source"],
                idempotency_key=claimed["idempotency_key"],
            )
            pipeline.queue.fail(
                job["job_id"],
                worker_id="writer-before-crash",
                error="acquittement perdu apres commit",
                permanent=True,
            )
            self.assertTrue(
                pipeline.writer_engine.event_exists(observed["event_id"])
            )

            cleaned = pipeline.cleanup_test_run(run_id)

            self.assertEqual(cleaned["state"], "cleaned")
            self.assertEqual(cleaned["successful_count"], 0)
            self.assertEqual(cleaned["failed_count"], 1)
            self.assertEqual(cleaned["forgotten_count"], 1)
            self.assertIsNone(
                pipeline.writer_engine.event_id_for_idempotency_key(
                    claimed["idempotency_key"]
                )
            )
            self.assertIsNone(pipeline.queue.get(job["job_id"]))
        finally:
            pipeline.close()

    @staticmethod
    def _terminal_test_run(
        queue: DurableInjectionQueue, run_id: str
    ) -> dict:
        created = queue.enqueue_test_run(run_id, count=1)
        job = created["jobs"][0]
        claimed = queue.claim(batch_size=1, worker_id=f"worker-{run_id}")[0]
        queue.complete(
            claimed["job_id"],
            worker_id=f"worker-{run_id}",
            result={},
        )
        return job

    def test_cleanup_owner_lease_recovers_cleaning_and_cleanup_failed_runs(self) -> None:
        queue = DurableInjectionQueue(self.root / "queue.sqlite3")
        try:
            cleaning_run = "111aaa111aaa"
            self._terminal_test_run(queue, cleaning_run)
            first_claim = queue.claim_test_run_for_cleanup(
                cleaning_run,
                owner="old-owner",
                stale_after_seconds=60,
                retry_after_seconds=0,
            )
            self.assertEqual(first_claim["cleanup_owner"], "old-owner")
            self.assertIsNone(
                queue.claim_test_run_for_cleanup(
                    cleaning_run,
                    owner="new-owner",
                    stale_after_seconds=60,
                    retry_after_seconds=0,
                )
            )
            with self.assertRaises(QueueStateError):
                queue.finish_test_run_cleanup(
                    cleaning_run, forgotten_count=0, owner="new-owner"
                )

            recovered = queue.claim_test_run_for_cleanup(
                cleaning_run,
                owner="new-owner",
                stale_after_seconds=0,
                retry_after_seconds=0,
            )
            self.assertEqual(recovered["cleanup_owner"], "new-owner")
            self.assertEqual(
                queue.finish_test_run_cleanup(
                    cleaning_run, forgotten_count=0, owner="new-owner"
                )["state"],
                "cleaned",
            )

            failed_run = "222bbb222bbb"
            self._terminal_test_run(queue, failed_run)
            queue.claim_test_run_for_cleanup(
                failed_run,
                owner="failing-owner",
                stale_after_seconds=60,
                retry_after_seconds=0,
            )
            queue.mark_test_run_cleanup_failed(
                failed_run,
                RuntimeError("panne simulee"),
                owner="failing-owner",
            )
            failed = queue.get_test_run(failed_run)
            self.assertEqual(failed["state"], "cleanup_failed")
            self.assertIsNone(failed["cleanup_owner"])
            self.assertIsNone(
                queue.claim_test_run_for_cleanup(
                    failed_run,
                    owner="retry-owner",
                    stale_after_seconds=60,
                    retry_after_seconds=60,
                )
            )
            retried = queue.claim_test_run_for_cleanup(
                failed_run,
                owner="retry-owner",
                stale_after_seconds=60,
                retry_after_seconds=0,
            )
            self.assertEqual(retried["state"], "cleaning")
            self.assertEqual(retried["cleanup_owner"], "retry-owner")
            self.assertEqual(
                queue.finish_test_run_cleanup(
                    failed_run, forgotten_count=0, owner="retry-owner"
                )["state"],
                "cleaned",
            )
        finally:
            queue.close()


class ChatPipelineReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.pipeline = MemoryPipeline(
            self.root / "memory.sqlite3",
            self.root / "queue.sqlite3",
            batch_size=1,
            poll_interval=0.01,
            retry_base_delay=0,
            retry_max_delay=0,
            autostart=False,
        )
        self.server = MemoryHTTPServer(
            ("127.0.0.1", 0),
            self.pipeline.reader_engine,
            PROJECT_ROOT / "web",
            pipeline=self.pipeline,
        )
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=3)
        self.pipeline.close()
        self.temporary_directory.cleanup()

    def post_json(self, path: str, payload: dict) -> tuple[int, dict]:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_same_chat_request_survives_rotation_and_redelivers_failed_and_forgotten(
        self,
    ) -> None:
        body = {
            "message": "Souviens-toi que Vega reste brillante",
            "request_id": "stable-chat-retry",
        }
        first_status, first = self.post_json("/api/chat", body)
        self.assertEqual(first_status, 202)
        job_id = first["data"]["job_id"]
        episode_id = first["data"]["episode_id"]

        claimed = self.pipeline.queue.claim(
            batch_size=1, worker_id="terminal-failure"
        )[0]
        self.pipeline.queue.fail(
            claimed["job_id"],
            worker_id="terminal-failure",
            error="echec terminal simule",
            permanent=True,
        )

        self.server.conversation_episode_events = MAX_CONVERSATION_EPISODE_EVENTS
        self.server.conversation_episode_id = "episode-after-failure"
        retry_status, retry = self.post_json("/api/chat", body)
        self.assertEqual(retry_status, 202)
        self.assertEqual(retry["data"]["job_id"], job_id)
        self.assertEqual(retry["data"]["episode_id"], episode_id)
        self.assertTrue(retry["data"]["retried"])
        self.assertEqual(retry["data"]["state"], "pending")
        self.assertEqual(
            self.server.conversation_episode_events,
            MAX_CONVERSATION_EPISODE_EVENTS,
        )
        self.assertEqual(
            self.server.conversation_episode_id, "episode-after-failure"
        )

        self.assertEqual(self.pipeline.worker.run_once(), 1)
        completed = self.pipeline.queue.get(job_id)
        old_event_id = completed["result"]["event_id"]
        self.assertTrue(
            self.pipeline.writer_engine.forget(old_event_id)["forgotten"]
        )

        self.server.conversation_episode_events = MAX_CONVERSATION_EPISODE_EVENTS
        self.server.conversation_episode_id = "episode-after-forget"
        forgotten_status, forgotten_retry = self.post_json("/api/chat", body)
        self.assertEqual(forgotten_status, 202)
        self.assertEqual(forgotten_retry["data"]["job_id"], job_id)
        self.assertEqual(forgotten_retry["data"]["episode_id"], episode_id)
        self.assertTrue(forgotten_retry["data"]["retried"])
        self.assertEqual(forgotten_retry["data"]["state"], "pending")
        self.assertEqual(
            self.server.conversation_episode_id, "episode-after-forget"
        )

        self.assertEqual(self.pipeline.worker.run_once(), 1)
        replacement = self.pipeline.queue.get(job_id)
        self.assertEqual(replacement["state"], "completed")
        self.assertNotEqual(replacement["result"]["event_id"], old_event_id)
        self.assertEqual(replacement["result"]["episode_id"], episode_id)

    def test_chat_forget_uses_full_writer_and_never_normal_reader(self) -> None:
        status, queued = self.post_json(
            "/api/chat",
            {
                "message": "Souviens-toi que Altair est proche",
                "request_id": "full-writer-forget",
            },
        )
        self.assertEqual(status, 202)
        self.assertEqual(self.pipeline.worker.run_once(), 1)
        job = self.pipeline.queue.get(queued["data"]["job_id"])
        event_id = job["result"]["event_id"]

        writer_mode = int(
            self.pipeline.writer_engine._connection.execute(  # noqa: SLF001
                "PRAGMA synchronous"
            ).fetchone()[0]
        )
        reader_mode = int(
            self.pipeline.reader_engine._connection.execute(  # noqa: SLF001
                "PRAGMA synchronous"
            ).fetchone()[0]
        )
        self.assertEqual(writer_mode, 2)
        self.assertEqual(reader_mode, 1)

        writer_forget = self.pipeline.writer_engine.forget
        reader_forget = self.pipeline.reader_engine.forget
        writer_calls: list[str] = []

        def recorded_writer_forget(value: str) -> dict:
            writer_calls.append(value)
            return writer_forget(value)

        def forbidden_reader_forget(_value: str) -> dict:
            raise AssertionError("Le lecteur NORMAL ne doit jamais oublier")

        self.pipeline.writer_engine.forget = recorded_writer_forget  # type: ignore[method-assign]
        self.pipeline.reader_engine.forget = forbidden_reader_forget  # type: ignore[method-assign]
        try:
            forget_status, forgotten = self.post_json(
                "/api/chat", {"message": f"oublie {event_id}"}
            )
        finally:
            self.pipeline.writer_engine.forget = writer_forget  # type: ignore[method-assign]
            self.pipeline.reader_engine.forget = reader_forget  # type: ignore[method-assign]

        self.assertEqual(forget_status, 200)
        self.assertTrue(forgotten["data"]["forgotten"])
        self.assertEqual(writer_calls, [event_id])
        self.assertFalse(self.pipeline.writer_engine.event_exists(event_id))


if __name__ == "__main__":
    unittest.main()
