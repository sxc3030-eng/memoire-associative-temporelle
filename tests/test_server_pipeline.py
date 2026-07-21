from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent import MemoryIdempotencyConflictError, MemoryPipeline
from memory_agent.server import MemoryHTTPServer


class AsyncPipelineServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.pipeline = MemoryPipeline(
            root / "memory.sqlite3",
            root / "injection.sqlite3",
            batch_size=1,
            poll_interval=0.01,
        )
        self.server = MemoryHTTPServer(
            ("127.0.0.1", 0),
            self.pipeline.reader_engine,
            PROJECT_ROOT / "web",
            pipeline=self.pipeline,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.pipeline.close()
        self.temporary_directory.cleanup()

    def get_json(self, path: str) -> tuple[int, dict]:
        with urlopen(self.base_url + path, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def post_json(self, path: str, payload: dict) -> tuple[int, dict]:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def wait_for_job(self, job_id: str, *, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, payload = self.get_json(f"/api/pipeline/jobs/{job_id}")
            if payload["job"]["state"] in {"completed", "failed"}:
                return payload["job"]
            time.sleep(0.01)
        self.fail(f"Le travail {job_id} n'a pas termine")

    def wait_for_test_run(
        self, run_id: str, *, state: str = "cleaned", timeout: float = 5.0
    ) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, payload = self.get_json(f"/api/pipeline/test/runs/{run_id}")
            test_run = payload["test_run"]
            if test_run["state"] == state:
                return test_run
            time.sleep(0.01)
        self.fail(f"Le run {run_id} n'a pas atteint l'etat {state}")

    def test_chat_is_queued_then_becomes_recallable_without_self_learning(self) -> None:
        status, queued = self.post_json(
            "/api/chat",
            {"message": "Souviens-toi que Nova adore les mangues"},
        )

        self.assertEqual(status, 202)
        self.assertTrue(queued["data"]["queued"])
        job_id = queued["data"]["job_id"]
        completed = self.wait_for_job(job_id)
        self.assertEqual(completed["state"], "completed")
        self.assertNotIn("text", completed)
        self.assertNotIn("context", completed)
        self.assertNotIn("source", completed)

        _, recalled = self.post_json(
            "/api/chat",
            {"message": "Que sais-tu de Nova ?"},
        )
        self.assertEqual(recalled["intent"], "recall")
        self.assertTrue(recalled["data"])

        _, pipeline_status = self.get_json("/api/pipeline")
        metrics = pipeline_status["pipeline"]
        self.assertTrue(metrics["reader_writer_separated"])
        self.assertEqual(metrics["received_submissions"], 1)
        # La reponse de rappel n'est jamais remise dans l'apprentissage.
        self.assertEqual(metrics["total_unique_jobs"], 1)

    def test_reader_remains_available_while_writer_is_blocked(self) -> None:
        _, seed = self.post_json(
            "/api/chat",
            {
                "message": "Souviens-toi que le repere stable est turquoise",
                "request_id": "wal-seed",
            },
        )
        self.wait_for_job(seed["data"]["job_id"])

        original_refresh = self.pipeline.writer_engine._refresh_aggregates
        writer_started = threading.Event()
        release_writer = threading.Event()

        def blocked_refresh():
            writer_started.set()
            if not release_writer.wait(3):
                raise TimeoutError("test writer release timeout")
            return original_refresh()

        self.pipeline.writer_engine._refresh_aggregates = blocked_refresh  # type: ignore[method-assign]
        try:
            status, queued = self.post_json(
                "/api/chat",
                {
                    "message": "Souviens-toi que le lecteur reste disponible",
                    "request_id": "wal-blocked-write",
                },
            )
            self.assertEqual(status, 202)
            self.assertTrue(writer_started.wait(1))

            started = time.monotonic()
            recall_status, recalled = self.post_json(
                "/api/chat", {"message": "Que sais-tu du repere turquoise ?"}
            )
            elapsed = time.monotonic() - started

            self.assertEqual(recall_status, 200)
            self.assertTrue(recalled["data"])
            self.assertLess(elapsed, 0.75)
        finally:
            release_writer.set()
            self.pipeline.writer_engine._refresh_aggregates = original_refresh  # type: ignore[method-assign]

        completed = self.wait_for_job(queued["data"]["job_id"])
        self.assertEqual(completed["state"], "completed")

    def test_json_import_is_validated_then_queued(self) -> None:
        document = {"profil": {"nom": "Lina", "ville": "Quebec"}}
        preview_status, preview = self.post_json(
            "/api/import",
            {"mode": "preview", "data": document, "filename": "profil.json"},
        )
        self.assertEqual(preview_status, 200)

        commit_status, committed = self.post_json(
            "/api/import",
            {
                "mode": "commit",
                "data": document,
                "filename": "profil.json",
                "import_id": preview["import_id"],
            },
        )
        self.assertEqual(commit_status, 202)
        self.assertTrue(committed["queued"])
        self.assertEqual(committed["queued_count"], 2)
        for job_id in committed["job_ids"]:
            self.assertEqual(self.wait_for_job(job_id)["state"], "completed")

        _, recalled = self.post_json(
            "/api/chat", {"message": "Que sais-tu de Lina ?"}
        )
        self.assertTrue(recalled["data"])

    def test_visible_pipeline_test_does_not_turn_generated_data_into_proof(self) -> None:
        maintenance = self.pipeline.worker.maintenance_callback
        self.pipeline.worker.maintenance_callback = None
        status, test_run = self.post_json("/api/pipeline/test", {"count": 5})

        self.assertEqual(status, 202)
        self.assertEqual(test_run["queued_count"], 5)
        for job_id in test_run["job_ids"]:
            self.assertEqual(self.wait_for_job(job_id)["state"], "completed")

        _, stats = self.get_json("/api/stats")
        self.assertEqual(stats["stats"]["events"], 5)
        self.assertEqual(stats["stats"]["sources"]["generated"], 5)
        self.assertEqual(stats["stats"]["trusted_events"], 0)
        self.assertEqual(stats["stats"]["continuations"], 0)

        self.pipeline.worker.maintenance_callback = maintenance
        self.pipeline.cleanup_terminal_test_runs()

        cleanup_status, cleanup = self.post_json(
            "/api/pipeline/test/cleanup",
            {"run_id": test_run["run_id"], "job_ids": test_run["job_ids"]},
        )
        self.assertEqual(cleanup_status, 200)
        self.assertEqual(cleanup["forgotten"], 5)
        self.assertEqual(cleanup["failed_count"], 0)
        self.assertEqual(self.get_json("/api/stats")[1]["stats"]["events"], 0)
        self.assertEqual(self.get_json("/api/pipeline")[1]["pipeline"]["total_unique_jobs"], 0)

    def test_test_run_is_cleaned_server_side_without_a_browser_callback(self) -> None:
        original_reader_forget = self.pipeline.reader_engine.forget_by_idempotency_key
        self.pipeline.reader_engine.forget_by_idempotency_key = (  # type: ignore[method-assign]
            lambda _key: (_ for _ in ()).throw(
                AssertionError("Le lecteur NORMAL ne doit pas nettoyer un test")
            )
        )
        try:
            status, created = self.post_json("/api/pipeline/test", {"count": 4})
            cleaned = self.wait_for_test_run(created["run_id"])
        finally:
            self.pipeline.reader_engine.forget_by_idempotency_key = original_reader_forget  # type: ignore[method-assign]

        self.assertEqual(status, 202)
        self.assertEqual(cleaned["successful_count"], 4)
        self.assertEqual(cleaned["failed_count"], 0)
        self.assertEqual(cleaned["forgotten_count"], 4)
        self.assertEqual(self.get_json("/api/stats")[1]["stats"]["events"], 0)
        self.assertEqual(
            self.get_json("/api/pipeline")[1]["pipeline"]["total_unique_jobs"], 0
        )

    def test_chat_retry_reuses_episode_and_redelivers_after_durable_forget(self) -> None:
        body = {
            "message": "Souviens-toi que Polaris indique le nord",
            "request_id": "retry-rotation-forget",
        }
        first_status, first = self.post_json("/api/chat", body)
        self.assertEqual(first_status, 202)
        job_id = first["data"]["job_id"]
        episode_id = first["data"]["episode_id"]
        completed = self.wait_for_job(job_id)
        event_id = completed["result"]["event_id"]

        # Simule une rotation entre l'acquittement perdu et le retry.
        self.server.conversation_episode_events = 32
        self.server.conversation_episode_id = "episode-apres-rotation"

        original_reader_forget = self.pipeline.reader_engine.forget
        self.pipeline.reader_engine.forget = (  # type: ignore[method-assign]
            lambda _event_id: (_ for _ in ()).throw(
                AssertionError("Le lecteur NORMAL ne doit pas supprimer")
            )
        )
        try:
            forget_status, forgotten = self.post_json(
                "/api/chat", {"message": f"oublie l'événement {event_id}"}
            )
        finally:
            self.pipeline.reader_engine.forget = original_reader_forget  # type: ignore[method-assign]
        self.assertEqual(forget_status, 200)
        self.assertTrue(forgotten["data"]["forgotten"])

        retry_status, retry = self.post_json("/api/chat", body)
        self.assertEqual(retry_status, 202)
        self.assertEqual(retry["data"]["job_id"], job_id)
        self.assertEqual(retry["data"]["episode_id"], episode_id)
        self.assertTrue(retry["data"]["retried"])
        replacement = self.wait_for_job(job_id)
        self.assertNotEqual(replacement["result"]["event_id"], event_id)

    def test_chat_retry_redelivers_a_failed_ticket(self) -> None:
        original_observe = self.pipeline.writer_engine.observe

        def reject_once(text: str, **kwargs):
            raise MemoryIdempotencyConflictError("echec terminal injecte")

        self.pipeline.writer_engine.observe = reject_once  # type: ignore[method-assign]
        body = {
            "message": "Souviens-toi que Sirius est brillante",
            "request_id": "retry-failed-ticket",
        }
        try:
            first_status, first = self.post_json("/api/chat", body)
            self.assertEqual(first_status, 202)
            job_id = first["data"]["job_id"]
            episode_id = first["data"]["episode_id"]
            self.assertEqual(self.wait_for_job(job_id)["state"], "failed")
        finally:
            self.pipeline.writer_engine.observe = original_observe  # type: ignore[method-assign]

        retry_status, retry = self.post_json("/api/chat", body)
        self.assertEqual(retry_status, 202)
        self.assertEqual(retry["data"]["job_id"], job_id)
        self.assertEqual(retry["data"]["episode_id"], episode_id)
        self.assertTrue(retry["data"]["retried"])
        self.assertEqual(self.wait_for_job(job_id)["state"], "completed")

    def test_async_import_does_not_duplicate_a_pre_pipeline_import(self) -> None:
        document = {"catalogue": {"objet": "Orion", "couleur": "bleu"}}
        preview = self.pipeline.reader_engine.preview_json_import(
            document, filename="catalogue.json"
        )
        self.pipeline.writer_engine.import_json(
            document,
            import_id=preview["import_id"],
            filename="catalogue.json",
        )
        self.assertEqual(self.pipeline.reader_engine.stats()["events"], 2)

        status, committed = self.post_json(
            "/api/import",
            {
                "mode": "commit",
                "data": document,
                "filename": "catalogue.json",
                "import_id": preview["import_id"],
            },
        )
        self.assertEqual(status, 202)
        for job_id in committed["job_ids"]:
            job = self.wait_for_job(job_id)
            self.assertTrue(job["result"]["duplicate"])
        self.assertEqual(self.pipeline.reader_engine.stats()["events"], 2)

    def test_chat_request_id_makes_a_lost_ack_safe_to_retry(self) -> None:
        body = {
            "message": "Souviens-toi que Altair brille en ete",
            "request_id": "requete-stable-1",
        }
        first_status, first = self.post_json("/api/chat", body)
        second_status, second = self.post_json("/api/chat", body)

        self.assertEqual(first_status, 202)
        self.assertEqual(second_status, 202)
        self.assertEqual(first["data"]["job_id"], second["data"]["job_id"])
        self.assertEqual(self.wait_for_job(first["data"]["job_id"])["state"], "completed")
        self.assertEqual(self.pipeline.reader_engine.stats()["events"], 1)

    def test_batch_job_status_is_precise_and_does_not_expose_payload(self) -> None:
        _, first = self.post_json(
            "/api/chat",
            {"message": "Souviens-toi que Deneb est lointaine", "request_id": "status-1"},
        )
        _, second = self.post_json(
            "/api/chat",
            {"message": "Souviens-toi que Rigel est bleue", "request_id": "status-2"},
        )
        ids = [first["data"]["job_id"], second["data"]["job_id"]]
        for job_id in ids:
            self.wait_for_job(job_id)

        status, payload = self.post_json(
            "/api/pipeline/jobs/status", {"job_ids": ids}
        )
        self.assertEqual(status, 200)
        self.assertEqual({job["job_id"] for job in payload["jobs"]}, set(ids))
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("Deneb est lointaine", encoded)
        self.assertNotIn("Rigel est bleue", encoded)
        self.assertTrue(all(job["state"] == "completed" for job in payload["jobs"]))

    def test_same_json_under_another_filename_remains_a_duplicate(self) -> None:
        document = {"objet": {"nom": "Lyra"}}
        _, first_preview = self.post_json(
            "/api/import",
            {"mode": "preview", "data": document, "filename": "a.json"},
        )
        _, first_commit = self.post_json(
            "/api/import",
            {
                "mode": "commit",
                "data": document,
                "filename": "a.json",
                "import_id": first_preview["import_id"],
            },
        )
        for job_id in first_commit["job_ids"]:
            self.wait_for_job(job_id)

        _, second_preview = self.post_json(
            "/api/import",
            {"mode": "preview", "data": document, "filename": "b.json"},
        )
        status, second_commit = self.post_json(
            "/api/import",
            {
                "mode": "commit",
                "data": document,
                "filename": "b.json",
                "import_id": second_preview["import_id"],
            },
        )
        self.assertEqual(status, 202)
        self.assertEqual(second_commit["queued_count"], 0)
        self.assertEqual(self.pipeline.reader_engine.stats()["events"], 1)


if __name__ == "__main__":
    unittest.main()
