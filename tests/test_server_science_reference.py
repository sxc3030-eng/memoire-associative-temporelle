from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.memory import MemoryEngine  # noqa: E402
from memory_agent.science_curriculum import import_science_reference  # noqa: E402
from memory_agent.server import (  # noqa: E402
    DEFAULT_SCIENCE_DATASET,
    MemoryHTTPServer,
    build_parser,
)


DATASET = PROJECT_ROOT / "examples" / "science-biographies-v1.json"
EVALUATION_KEYS = {
    "evaluation_questions",
    "expected_answer_fragments",
    "forbidden_answer_fragments",
    "supporting_claim_ids",
    "answer_status",
    "evaluation_kind",
}


def _stored_reference_rows(database: Path) -> list[tuple[str, ...]]:
    connection = sqlite3.connect(database)
    try:
        return connection.execute(
            """
            SELECT episode_id, text, source_json, context_json, idempotency_key
            FROM events
            ORDER BY episode_id
            """
        ).fetchall()
    finally:
        connection.close()


class ScienceReferenceServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.personal_path = root / "personal.sqlite3"
        self.reference_path = root / "science-reference.sqlite3"
        self.personal = MemoryEngine(self.personal_path)
        self.reference = MemoryEngine(self.reference_path)
        self.server = MemoryHTTPServer(
            ("127.0.0.1", 0),
            self.personal,
            PROJECT_ROOT / "web",
            reference_engine=self.reference,
            science_dataset_path=DATASET,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.personal.close()
        self.temporary_directory.cleanup()

    def get_json(self, path: str) -> dict:
        with urlopen(self.base_url + path, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_json(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_status_public_questions_and_chat_use_separate_reference(self) -> None:
        status = self.get_json("/api/science/reference")["reference"]
        questions = self.get_json("/api/science/questions")
        answer = self.post_json(
            "/api/chat",
            {"message": "Que sais-tu au sujet de Photo 51 ?"},
        )

        self.assertTrue(status["enabled"])
        self.assertTrue(status["ready"])
        self.assertEqual(status["policy"], "reference")
        self.assertEqual(status["storage"], "separate_sqlite")
        self.assertEqual(status["claims"], 31)
        self.assertEqual(status["dossiers"], 5)
        self.assertEqual(questions["count"], 9)
        self.assertTrue(questions["questions"])
        self.assertTrue(
            all(set(question) == {"id", "question"} for question in questions["questions"])
        )
        self.assertEqual(answer["intent"], "recall")
        self.assertTrue(
            any(item.get("space") == "science-reference" for item in answer["data"])
        )
        self.assertEqual(self.personal.stats()["events"], 0)
        self.assertEqual(self.reference.stats()["events"], 5)

        repeated = self.post_json("/api/science/reference/import", {})
        imported = repeated["reference"]["import"]
        self.assertEqual(imported["created_dossiers"], 0)
        self.assertEqual(imported["duplicate_dossiers"], 5)

    def test_persistent_rows_ignore_the_entire_evaluation_partition(self) -> None:
        root = Path(self.temporary_directory.name)
        original_dataset = root / "original.json"
        changed_dataset = root / "changed-evaluation.json"
        original_document = json.loads(DATASET.read_text(encoding="utf-8"))
        original_dataset.write_text(
            json.dumps(original_document, ensure_ascii=False), encoding="utf-8"
        )
        changed_document = json.loads(json.dumps(original_document, ensure_ascii=False))
        sentinel = "EVALUATION-SENTINEL-NEVER-STORE-9F5E"
        for index, question in enumerate(changed_document["evaluation_questions"]):
            question["question"] = f"{sentinel}-QUESTION-{index}"
            question["expected_answer_fragments"] = (
                []
                if question["answer_status"] == "unanswerable"
                else [f"{sentinel}-EXPECTED-{index}"]
            )
            question["forbidden_answer_fragments"] = [
                f"{sentinel}-FORBIDDEN-{index}"
            ]
        changed_dataset.write_text(
            json.dumps(changed_document, ensure_ascii=False), encoding="utf-8"
        )

        first_path = root / "first-reference.sqlite3"
        second_path = root / "second-reference.sqlite3"
        first_engine = MemoryEngine(first_path)
        second_engine = MemoryEngine(second_path)
        try:
            first = import_science_reference(original_dataset, first_engine)
            second = import_science_reference(changed_dataset, second_engine)
        finally:
            first_engine.close()
            second_engine.close()

        first_rows = _stored_reference_rows(first_path)
        second_rows = _stored_reference_rows(second_path)
        encoded = json.dumps(second_rows, ensure_ascii=False, sort_keys=True)
        self.assertEqual(first["reference_sha256"], second["reference_sha256"])
        self.assertEqual(first_rows, second_rows)
        self.assertNotIn(sentinel, encoded)
        for key in EVALUATION_KEYS:
            self.assertNotIn(f'"{key}"', encoded)

        reopened = MemoryEngine(first_path)
        try:
            recalled = reopened.recall("penicilline", top_k=5)
            self.assertTrue(recalled)
            self.assertEqual(reopened.stats()["events"], 5)
        finally:
            reopened.close()


class ScienceReferenceStartupTests(unittest.TestCase):
    def test_default_cli_enables_project_dataset_and_sibling_database(self) -> None:
        arguments = build_parser().parse_args([])

        self.assertFalse(arguments.no_science_reference)
        self.assertEqual(arguments.science_dataset, DEFAULT_SCIENCE_DATASET)
        self.assertIsNone(arguments.science_reference_db)


if __name__ == "__main__":
    unittest.main()
