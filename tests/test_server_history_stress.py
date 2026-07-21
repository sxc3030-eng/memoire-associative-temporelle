from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent import MemoryPipeline  # noqa: E402
from memory_agent.history_stress_lab import HistoryStressConfig  # noqa: E402
from memory_agent.server import (  # noqa: E402
    MAX_HISTORY_STRESS_BODY_BYTES,
    MemoryHTTPServer,
)


class HistoryStressServerTests(unittest.TestCase):
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

    def request_json(
        self, path: str, *, method: str = "GET", payload: dict | None = None
    ) -> tuple[int, dict]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method,
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            try:
                return error.code, json.loads(error.read().decode("utf-8"))
            finally:
                error.close()

    def main_memory_counters(self) -> tuple[int, int, int, int]:
        memory = self.pipeline.reader_engine.stats()
        queue = self.pipeline.queue.stats()
        return (
            int(memory["events"]),
            int(memory["episodes"]),
            int(queue["total"]),
            int(queue["unfinished"]),
        )

    def test_catalog_and_run_use_contract_without_mutating_main_memory(self) -> None:
        before = self.main_memory_counters()
        fake_catalog = {
            "schema_version": "history-stress-test-v1",
            "config_bounds": {"event_count": {"minimum": 2, "maximum": 10_000}},
        }
        fake_report = {
            "schema_version": "history-stress-test-v1",
            "scenario": {"source_events": 25},
            "retrieval": {"top1_percent": 80.0},
            "isolation": {"temporary_storage_removed_after_run": True},
        }

        with (
            patch(
                "memory_agent.server.history_stress_catalog",
                return_value=fake_catalog,
            ) as catalog_mock,
            patch(
                "memory_agent.server.run_history_stress",
                return_value=fake_report,
            ) as run_mock,
        ):
            catalog_status, catalog = self.request_json(
                "/api/stress/history/catalog"
            )
            run_status, run = self.request_json(
                "/api/stress/history/run",
                method="POST",
                payload={"event_count": 25, "seed": 20_260_721},
            )

        expected_http_catalog = {
            **fake_catalog,
            "config_bounds": {
                "event_count": {"minimum": 5, "maximum": 100}
            },
            "library_config_bounds": fake_catalog["config_bounds"],
            "interface": "http_local",
        }
        self.assertEqual(catalog_status, 200)
        self.assertEqual(
            catalog,
            {"ok": True, "catalog": expected_http_catalog},
        )
        catalog_mock.assert_called_once_with()
        self.assertEqual(run_status, 200)
        self.assertEqual(run, {"ok": True, "report": fake_report})
        run_mock.assert_called_once()
        config = run_mock.call_args.args[0]
        self.assertIsInstance(config, HistoryStressConfig)
        self.assertEqual(config.event_count, 25)
        self.assertEqual(config.seed, 20_260_721)
        self.assertEqual(self.main_memory_counters(), before)

    def test_run_rejects_invalid_or_expansive_parameters_before_execution(self) -> None:
        invalid_payloads = (
            {},
            {"event_count": 4, "seed": 1},
            {"event_count": 101, "seed": 1},
            {"event_count": True, "seed": 1},
            {"event_count": 5.0, "seed": 1},
            {"event_count": 5},
            {"event_count": 5, "seed": True},
            {"event_count": 5, "seed": -1},
            {"event_count": 5, "seed": 2**63},
            {"event_count": 5, "seed": 1, "episode_size": 64},
        )
        before = self.main_memory_counters()

        with patch("memory_agent.server.run_history_stress") as run_mock:
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    status, response = self.request_json(
                        "/api/stress/history/run",
                        method="POST",
                        payload=payload,
                    )
                    self.assertEqual(status, 400)
                    self.assertFalse(response["ok"])

        run_mock.assert_not_called()
        self.assertEqual(self.main_memory_counters(), before)

    def test_run_body_is_bounded(self) -> None:
        before = self.main_memory_counters()
        payload = {
            "event_count": 5,
            "seed": 1,
            "padding": "x" * MAX_HISTORY_STRESS_BODY_BYTES,
        }

        with patch("memory_agent.server.run_history_stress") as run_mock:
            status, response = self.request_json(
                "/api/stress/history/run", method="POST", payload=payload
            )

        self.assertEqual(status, 413)
        self.assertFalse(response["ok"])
        run_mock.assert_not_called()
        self.assertEqual(self.main_memory_counters(), before)

    def test_only_one_history_run_can_execute_at_a_time(self) -> None:
        started = threading.Event()
        release = threading.Event()
        first_response: list[tuple[int, dict]] = []
        before = self.main_memory_counters()

        def blocked_run(config: HistoryStressConfig) -> dict:
            started.set()
            if not release.wait(timeout=3):
                raise TimeoutError("test release timeout")
            return {"config": {"event_count": config.event_count, "seed": config.seed}}

        def send_first_request() -> None:
            first_response.append(
                self.request_json(
                    "/api/stress/history/run",
                    method="POST",
                    payload={"event_count": 5, "seed": 1},
                )
            )

        with patch(
            "memory_agent.server.run_history_stress", side_effect=blocked_run
        ) as run_mock:
            first_thread = threading.Thread(target=send_first_request, daemon=True)
            first_thread.start()
            self.assertTrue(started.wait(timeout=2))
            health_status, health = self.request_json("/api/health")
            second_status, second = self.request_json(
                "/api/stress/history/run",
                method="POST",
                payload={"event_count": 5, "seed": 2},
            )
            release.set()
            first_thread.join(timeout=3)

        self.assertFalse(first_thread.is_alive())
        self.assertEqual(health_status, 200)
        self.assertTrue(health["read_available"])
        self.assertEqual(second_status, 409)
        self.assertFalse(second["ok"])
        self.assertEqual(first_response[0][0], 200)
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(self.main_memory_counters(), before)

    def test_internal_error_is_sanitized_and_releases_run_lock(self) -> None:
        before = self.main_memory_counters()
        with (
            patch(
                "memory_agent.server.run_history_stress",
                side_effect=[
                    RuntimeError("chemin-secret-et-details"),
                    {"finished": True},
                ],
            ) as run_mock,
            patch("memory_agent.server.LOGGER.exception") as log_mock,
        ):
            failed_status, failed = self.request_json(
                "/api/stress/history/run",
                method="POST",
                payload={"event_count": 5, "seed": 1},
            )
            retry_status, retry = self.request_json(
                "/api/stress/history/run",
                method="POST",
                payload={"event_count": 5, "seed": 1},
            )

        self.assertEqual(failed_status, 500)
        self.assertFalse(failed["ok"])
        self.assertNotIn("chemin-secret", failed["error"])
        self.assertEqual(retry_status, 200)
        self.assertEqual(retry["report"], {"finished": True})
        self.assertEqual(run_mock.call_count, 2)
        log_mock.assert_called_once()
        self.assertEqual(self.main_memory_counters(), before)


if __name__ == "__main__":
    unittest.main()
