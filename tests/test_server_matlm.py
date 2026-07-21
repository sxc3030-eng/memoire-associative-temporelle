from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent import MemoryEngine
from memory_agent.native_llm_contract import build_capsule, validate_capsule
from memory_agent.server import (
    MATLMUnavailableError,
    MATLMWorker,
    MATLMWorkerConfig,
    MATLMWorkerError,
    MemoryHTTPServer,
)


def _capsule(request_id: str) -> dict:
    return build_capsule(
        request_id=request_id,
        question="Que dit la preuve ?",
        evidence=[
            {
                "evidence_id": "test:evidence:1",
                "text": "La preuve dit alpha.",
                "space": "private",
                "status": "confirmed",
                "confidence": 1.0,
                "temporal_context": None,
                "tags": ["test"],
            }
        ],
    )


class MATLMWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.model = self.root / "model"
        self.adapter = self.root / "adapter"
        self.model.mkdir()
        self.adapter.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _worker(self, script_text: str) -> MATLMWorker:
        script = self.root / "fake_ask_matlm.py"
        script.write_text(script_text, encoding="utf-8")
        return MATLMWorker(
            MATLMWorkerConfig(
                enabled=True,
                python_path=Path(sys.executable),
                base_model_path=self.model,
                adapter_path=self.adapter,
                script_path=script,
                request_timeout_seconds=5.0,
                stop_timeout_seconds=2.0,
            )
        )

    def test_disabled_or_unconfigured_worker_never_starts(self) -> None:
        disabled = MATLMWorker()
        self.assertEqual(disabled.status()["state"], "disabled")
        with self.assertRaises(MATLMUnavailableError):
            disabled.start()

        missing = MATLMWorker(
            MATLMWorkerConfig(
                enabled=True,
                python_path=self.root / "missing-python.exe",
                base_model_path=self.root / "missing-model",
                adapter_path=self.root / "missing-adapter",
            )
        )
        with self.assertRaises(MATLMUnavailableError):
            missing.start()
        self.assertFalse(missing.status()["running"])

    def test_one_persistent_offline_process_answers_twice_then_stops(self) -> None:
        worker = self._worker(
            """
import json
import os
import sys

count = 0
print(json.dumps({
    "schema_version": "matlm-interactive-control-v1",
    "event": "ready",
    "ok": True,
}, separators=(",", ":")), flush=True)
for line in sys.stdin:
    capsule = json.loads(line)
    count += 1
    evidence_ids = [row["evidence_id"] for row in capsule["evidence"][:1]]
    answer = {
        "schema_version": "memory-native-answer-v1",
        "request_id": capsule["request_id"],
        "answer": f"reply-{count}-offline-{os.environ.get('HF_HUB_OFFLINE')}-{os.environ.get('TRANSFORMERS_OFFLINE')}",
        "confidence": 0.8,
        "evidence_ids": evidence_ids,
        "calculations": [],
        "abstention": {"abstained": False, "reason": "none", "missing_information": []},
    }
    print(json.dumps(answer, separators=(",", ":")), flush=True)
""".lstrip()
        )
        try:
            started = worker.start()
            self.assertTrue(started["running"])
            deadline = time.monotonic() + 2.0
            while worker.status()["state"] != "ready" and time.monotonic() < deadline:
                time.sleep(0.01)
            ready = worker.status()
            self.assertEqual(ready["state"], "ready")
            self.assertTrue(ready["model_loaded"])
            self.assertEqual(ready["completed_requests"], 0)
            self.assertEqual(ready["runtime"]["max_input_tokens"], 4_096)
            self.assertEqual(ready["runtime"]["max_new_tokens"], 768)
            first = worker.ask(_capsule("worker-test-1"))
            second = worker.ask(_capsule("worker-test-2"))

            self.assertEqual(first["answer"], "reply-1-offline-1-1")
            self.assertEqual(second["answer"], "reply-2-offline-1-1")
            self.assertEqual(worker.status()["completed_requests"], 2)
            self.assertEqual(worker.status()["state"], "ready")
        finally:
            stopped = worker.stop()
        self.assertFalse(stopped["running"])
        self.assertEqual(stopped["state"], "stopped")

    def test_ready_frame_during_first_request_is_not_confused_with_answer(self) -> None:
        worker = self._worker(
            """
import json
import sys
import time

time.sleep(0.1)
print(json.dumps({
    "schema_version": "matlm-interactive-control-v1",
    "event": "ready",
    "ok": True,
}, separators=(",", ":")), flush=True)
for line in sys.stdin:
    capsule = json.loads(line)
    answer = {
        "schema_version": "memory-native-answer-v1",
        "request_id": capsule["request_id"],
        "answer": "reponse-apres-ready",
        "confidence": 0.8,
        "evidence_ids": [capsule["evidence"][0]["evidence_id"]],
        "calculations": [],
        "abstention": {"abstained": False, "reason": "none", "missing_information": []},
    }
    print(json.dumps(answer, separators=(",", ":")), flush=True)
""".lstrip()
        )
        try:
            worker.start()
            answer = worker.ask(_capsule("worker-early-question"))
            self.assertEqual(answer["answer"], "reponse-apres-ready")
            status = worker.status()
            self.assertEqual(status["state"], "ready")
            self.assertTrue(status["model_loaded"])
            self.assertEqual(status["completed_requests"], 1)
        finally:
            worker.stop()

    def test_invalid_child_output_is_not_returned_and_stops_worker(self) -> None:
        worker = self._worker(
            """
import sys
for line in sys.stdin:
    print("{}", flush=True)
""".lstrip()
        )
        worker.start()
        with self.assertRaises(MATLMWorkerError):
            worker.ask(_capsule("worker-invalid-1"))
        status = worker.status()
        self.assertFalse(status["running"])
        self.assertEqual(status["state"], "error")


class _FakeMATLMWorker:
    def __init__(self) -> None:
        self.running = False
        self.capsules: list[dict] = []

    def status(self) -> dict:
        return {
            "enabled": True,
            "configured": True,
            "state": "ready" if self.running else "stopped",
            "running": self.running,
            "model_loaded": self.running,
            "offline_only": True,
            "persistent_process": True,
            "one_model_at_a_time": True,
            "automatic_learning": False,
            "completed_requests": len(self.capsules),
            "configuration_issues": [],
            "last_error": None,
            "limits": {"request_timeout_seconds": 5},
        }

    def start(self) -> dict:
        self.running = True
        return self.status()

    def stop(self) -> dict:
        self.running = False
        return self.status()

    def ask(self, capsule: dict) -> dict:
        if not self.running:
            raise MATLMUnavailableError("worker arrete")
        trusted = validate_capsule(capsule)
        self.capsules.append(trusted)
        evidence_ids = [row["evidence_id"] for row in trusted["evidence"][:1]]
        if not evidence_ids:
            return {
                "schema_version": "memory-native-answer-v1",
                "request_id": trusted["request_id"],
                "answer": "",
                "confidence": 0.0,
                "evidence_ids": [],
                "calculations": [],
                "abstention": {
                    "abstained": True,
                    "reason": "insufficient_evidence",
                    "missing_information": ["preuve manquante"],
                },
            }
        return {
            "schema_version": "memory-native-answer-v1",
            "request_id": trusted["request_id"],
            "answer": "Rio aime courir dans le parc.",
            "confidence": 0.9,
            "evidence_ids": evidence_ids,
            "calculations": [],
            "abstention": {"abstained": False, "reason": "none", "missing_information": []},
        }


class MATLMServerRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "memory.sqlite3"
        self.engine = MemoryEngine(database_path)
        self.engine.observe("Rio aime courir dans le parc", source="user_confirmed")
        self.worker = _FakeMATLMWorker()
        self.server = MemoryHTTPServer(
            ("127.0.0.1", 0),
            self.engine,
            PROJECT_ROOT / "web",
            matlm_worker=self.worker,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.engine.close()
        self.temporary_directory.cleanup()

    def _get(self, path: str) -> dict:
        with urlopen(self.base_url + path, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            try:
                return error.code, json.loads(error.read().decode("utf-8"))
            finally:
                error.close()

    def test_status_start_ask_stop_and_no_automatic_learning(self) -> None:
        before = self.engine.stats()["events"]
        with urlopen(self.base_url + "/", timeout=5) as response:
            interface = response.read().decode("utf-8")
        self.assertIn('id="matlm-messages"', interface)
        self.assertIn('id="matlm-new-chat"', interface)
        self.assertFalse(self._get("/api/matlm/status")["matlm"]["running"])

        start_status, started = self._post("/api/matlm/start", {})
        ask_status, answer = self._post(
            "/api/matlm/ask",
            {"question": "Que sais-tu de Rio ?", "request_id": "http-matlm-1"},
        )
        stop_status, stopped = self._post("/api/matlm/stop", {})

        self.assertEqual(start_status, 200)
        self.assertTrue(started["matlm"]["running"])
        self.assertEqual(ask_status, 200)
        self.assertEqual(answer["intent"], "matlm")
        self.assertEqual(answer["reply"], "Rio aime courir dans le parc.")
        self.assertFalse(answer["details"]["automatic_learning"])
        self.assertTrue(answer["answer"]["evidence_ids"])
        self.assertEqual(len(answer["citations"]), 1)
        self.assertIn("Rio", answer["citations"][0]["text"])
        self.assertEqual(stop_status, 200)
        self.assertFalse(stopped["matlm"]["running"])
        self.assertEqual(self.engine.stats()["events"], before)
        self.assertEqual(len(self.worker.capsules), 1)

    def test_question_validation_happens_before_worker(self) -> None:
        self._post("/api/matlm/start", {})
        status, payload = self._post(
            "/api/matlm/ask",
            {"question": "Rio", "unexpected": True},
        )
        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertEqual(self.worker.capsules, [])


if __name__ == "__main__":
    unittest.main()
