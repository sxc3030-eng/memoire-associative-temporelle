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

from memory_agent import MemoryPipeline  # noqa: E402
from memory_agent.math_engine import MathEngine  # noqa: E402
from memory_agent.server import (  # noqa: E402
    MAX_MATH_EXPRESSION_CHARS,
    MathEngineError,
    MemoryHTTPServer,
)


class FakeMathError(MathEngineError):
    def __init__(self, code: str, message: str):
        Exception.__init__(self, message)
        self.code = code
        self.message = message


class FakeMathEngine:
    def evaluate(self, expression: str) -> dict:
        if expression == "2 + 3 * 4":
            return {
                "expression": expression,
                "result": 14,
                "algorithm": "arithmetic",
                "algorithm_version": "math-test-v1",
            }
        raise FakeMathError("unsupported_expression", "Expression non prise en charge")

    def catalog(self) -> dict:
        return {
            "version": "math-test-v1",
            "algorithms": ["arithmetic", "power"],
        }

    def catalog_entries(self) -> list[dict]:
        return [
            {
                "name": "arithmetic",
                "category": "arithmetic",
                "description": "Respecte la priorite des operations.",
                "signature": "expression arithmetique",
                "rules": ["multiplication avant addition"],
                "exact": True,
                "exactness": "true",
                "learning_level": 4,
                "maturity": "operational",
                "returns": "entier ou decimal",
                "examples": [{"expression": "2+2", "result": 4}],
                "result": 999,
            },
            {
                "name": "power",
                "description": "Calcule une puissance entiere.",
                "syntax": "base ** exposant",
            },
        ]


class MathServerTests(unittest.TestCase):
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
            math_engine=FakeMathEngine(),
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
            return error.code, json.loads(error.read().decode("utf-8"))

    def wait_for_jobs(self, job_ids: list[str], *, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            jobs = self.pipeline.queue.get_many(job_ids)
            if len(jobs) == len(job_ids) and all(
                job["state"] in {"completed", "failed"} for job in jobs
            ):
                self.assertTrue(all(job["state"] == "completed" for job in jobs))
                return
            time.sleep(0.01)
        self.fail("Les imports mathematiques n'ont pas ete consolides")

    def test_catalog_and_calculation_never_write_memory(self) -> None:
        before_events = self.pipeline.reader_engine.stats()["events"]
        before_jobs = self.pipeline.queue.stats()["total"]

        catalog_status, catalog = self.request_json("/api/math/catalog")
        calculate_status, calculation = self.request_json(
            "/api/calculate",
            method="POST",
            payload={"expression": "2 + 3 * 4"},
        )
        chat_status, chat = self.request_json(
            "/api/chat",
            method="POST",
            payload={"message": "Calcule 2 + 3 * 4"},
        )

        self.assertEqual(catalog_status, 200)
        self.assertEqual(catalog["count"], 2)
        self.assertEqual(catalog["catalog"]["version"], "math-test-v1")
        self.assertEqual(calculate_status, 200)
        self.assertEqual(calculation["intent"], "calculate")
        self.assertEqual(calculation["result"]["result"], 14)
        self.assertEqual(chat_status, 200)
        self.assertEqual(chat["intent"], "calculate")
        self.assertEqual(chat["data"]["result"], 14)
        self.assertEqual(self.pipeline.reader_engine.stats()["events"], before_events)
        self.assertEqual(self.pipeline.queue.stats()["total"], before_jobs)

    def test_real_math_engine_contract_is_exposed_without_memory_write(self) -> None:
        self.server.math_engine = MathEngine()
        before_events = self.pipeline.reader_engine.stats()["events"]
        before_jobs = self.pipeline.queue.stats()["total"]

        catalog_status, catalog = self.request_json("/api/math/catalog")
        calculate_status, calculation = self.request_json(
            "/api/calculate",
            method="POST",
            payload={"expression": "frac(1, 3) + frac(1, 6)"},
        )
        chat_status, chat = self.request_json(
            "/api/chat",
            method="POST",
            payload={"message": "Calcule frac(1, 3) + frac(1, 6)"},
        )

        self.assertEqual(catalog_status, 200)
        self.assertEqual(catalog["count"], 41)
        self.assertEqual(catalog["catalog"]["version"], "math-core-v1")
        self.assertEqual(calculate_status, 200)
        self.assertEqual(calculation["result"]["display"], "1/2")
        self.assertTrue(calculation["result"]["exact"])
        self.assertEqual(chat_status, 200)
        self.assertEqual(chat["reply"], "Résultat : 1/2")
        self.assertEqual(self.pipeline.reader_engine.stats()["events"], before_events)
        self.assertEqual(self.pipeline.queue.stats()["total"], before_jobs)

    def test_calculation_validation_and_domain_errors_are_bounded(self) -> None:
        missing_status, _ = self.request_json(
            "/api/calculate", method="POST", payload={}
        )
        invalid_status, invalid = self.request_json(
            "/api/calculate",
            method="POST",
            payload={"expression": "fonction_interdite(2)"},
        )
        oversized_status, _ = self.request_json(
            "/api/calculate",
            method="POST",
            payload={"expression": "1" * (MAX_MATH_EXPRESSION_CHARS + 1)},
        )

        self.assertEqual(missing_status, 400)
        self.assertEqual(invalid_status, 422)
        self.assertEqual(invalid["code"], "unsupported_expression")
        self.assertEqual(oversized_status, 413)
        self.assertEqual(self.pipeline.reader_engine.stats()["events"], 0)

    def test_explicit_catalog_import_is_idempotent_and_excludes_results(self) -> None:
        first_status, first = self.request_json(
            "/api/math/catalog/import", method="POST", payload={}
        )

        self.assertEqual(first_status, 202)
        self.assertEqual(first["catalog_version"], "math-test-v1")
        self.assertEqual(first["total_count"], 2)
        self.assertEqual(first["queued_count"], 2)
        self.assertEqual(first["duplicate_count"], 0)
        self.wait_for_jobs(first["job_ids"])

        jobs = self.pipeline.queue.get_many(first["job_ids"])
        self.assertEqual(
            {job["idempotency_key"] for job in jobs},
            {
                "math-catalog:math-test-v1:arithmetic",
                "math-catalog:math-test-v1:power",
            },
        )
        self.assertTrue(
            all(job["source"]["type"] == "executed" for job in jobs)
        )
        self.assertTrue(
            all(job["source"]["origin"] == "math_engine_catalog" for job in jobs)
        )
        encoded_text = " ".join(job["text"] for job in jobs)
        self.assertNotIn("999", encoded_text)
        self.assertNotIn("2+2", encoded_text)
        self.assertIn("signature: expression arithmetique", encoded_text)
        self.assertIn("exact: true", encoded_text)
        self.assertIn("learning_level: 4", encoded_text)
        self.assertIn("maturity: operational", encoded_text)

        second_status, second = self.request_json(
            "/api/math/catalog/import", method="POST", payload={}
        )
        self.assertEqual(second_status, 202)
        self.assertEqual(second["queued_count"], 0)
        self.assertEqual(second["duplicate_count"], 2)
        self.assertEqual(second["deduplicated_count"], 2)
        self.assertEqual(second["job_ids"], first["job_ids"])
        self.assertEqual(self.pipeline.reader_engine.stats()["events"], 2)


if __name__ == "__main__":
    unittest.main()
