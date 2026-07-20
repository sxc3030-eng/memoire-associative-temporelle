from __future__ import annotations

import json
from http.client import HTTPConnection
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent import MemoryEngine
from memory_agent.json_import import JSONImportError
from memory_agent.server import MemoryHTTPServer


class JSONImportEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "memory.sqlite3"
        self.engine = MemoryEngine(database_path)

    def tearDown(self) -> None:
        self.engine.close()
        self.temporary_directory.cleanup()

    def test_preview_categorises_without_writing(self) -> None:
        before = self.engine.stats()
        preview = self.engine.preview_json_import(
            {
                "profiles": [{"name": "Alice", "age": 31}],
                "city": "Montreal",
                "budget": 125.5,
            },
            filename=r"C:\fake\people.json",
        )
        after = self.engine.stats()

        self.assertEqual(before["events"], after["events"])
        self.assertEqual(preview["filename"], "people.json")
        self.assertEqual(preview["summary"]["memories"], 4)
        self.assertEqual(preview["summary"]["root_type"], "object")
        categories = {item["path"]: item["category"] for item in preview["items"]}
        self.assertEqual(categories["$.profiles[0].name"], "profiles")
        self.assertEqual(categories["$.profiles[0].age"], "profiles")
        self.assertEqual(categories["$.city"], "localisation")
        self.assertEqual(categories["$.budget"], "finance")
        self.assertTrue(preview["import_id"].startswith("json-v1-"))

    def test_commit_requires_matching_preview_and_is_idempotent(self) -> None:
        data = {"identity": {"name": "Milo"}, "likes": ["poisson", "soleil"]}
        preview = self.engine.preview_json_import(data, filename="agent.json")

        with self.assertRaisesRegex(JSONImportError, "import_id"):
            self.engine.import_json(data, import_id="", filename="agent.json")
        with self.assertRaisesRegex(JSONImportError, "ne correspond pas"):
            self.engine.import_json(data, import_id="json-v1-deadbeef", filename="agent.json")
        with self.assertRaisesRegex(JSONImportError, "ne correspond pas"):
            self.engine.import_json(
                {"identity": {"name": "Mila"}},
                import_id=preview["import_id"],
                filename="agent.json",
            )
        with self.assertRaisesRegex(JSONImportError, "ne correspond pas"):
            self.engine.import_json(
                data,
                import_id=preview["import_id"],
                filename="renamed.json",
            )

        first = self.engine.import_json(
            data,
            import_id=preview["import_id"],
            filename="agent.json",
        )
        second = self.engine.import_json(
            data,
            import_id=preview["import_id"],
            filename="agent.json",
        )

        self.assertEqual(first["created"], 3)
        self.assertEqual(first["duplicates"], 0)
        self.assertEqual(second["created"], 0)
        self.assertEqual(second["duplicates"], 3)
        self.assertEqual(self.engine.stats()["events"], 3)
        self.assertTrue(second["resume_safe"])

    def test_imported_memory_explains_path_category_and_source(self) -> None:
        data = {"profile": {"favorite_color": "bleu cobalt", "active": True}, "score": 8}
        preview = self.engine.preview_json_import(data, filename="profile.json")
        committed = self.engine.import_json(
            data,
            import_id=preview["import_id"],
            filename="profile.json",
        )

        memories = self.engine.recall("cobalt", top_k=5)
        event = next(
            item
            for item in memories[0]["events"]
            if item["context"]["json_path"] == "$.profile.favorite_color"
        )
        metadata = event["context"]
        self.assertEqual(event["source"], "user_confirmed")
        self.assertEqual(metadata["origin"], "json_import")
        self.assertEqual(metadata["category"], "profile")
        self.assertEqual(metadata["json_path"], "$.profile.favorite_color")
        self.assertEqual(metadata["filename"], "profile.json")
        self.assertEqual(metadata["digest"], preview["digest"])
        self.assertIn("favorite_color", event["text"])
        self.assertEqual(len({item["episode_id"] for item in committed["results"]}), 3)

    def test_empty_values_are_ignored_and_symbol_only_leaf_is_repairable(self) -> None:
        data = {"a": "ok", "missing": None, "blank": "  ", "💥": "🔥"}
        preview = self.engine.preview_json_import(data, filename="symbols.json")

        self.assertEqual(preview["summary"]["memories"], 2)
        symbol = next(item for item in preview["items"] if item["path"] == '$["💥"]')
        self.assertIn("Valeur JSON", symbol["text"])

        first = self.engine.import_json(
            data,
            import_id=preview["import_id"],
            filename="symbols.json",
        )
        second = self.engine.import_json(
            data,
            import_id=preview["import_id"],
            filename="symbols.json",
        )
        self.assertEqual(first["created"], 2)
        self.assertEqual(second["duplicates"], 2)
        self.assertEqual(self.engine.stats()["events"], 2)

    def test_generic_root_wrapper_does_not_become_the_category(self) -> None:
        preview = self.engine.preview_json_import(
            {"data": [{"name": "Alice", "city": "Montreal", "first-name": "Al"}]}
        )
        categories = {item["path"]: item["category"] for item in preview["items"]}
        self.assertEqual(categories["$.data[0].name"], "identite")
        self.assertEqual(categories["$.data[0].city"], "localisation")
        self.assertIn('$.data[0]["first-name"]', categories)

    def test_depth_node_memory_and_file_limits(self) -> None:
        too_deep: object = "leaf"
        for _ in range(33):
            too_deep = [too_deep]
        with self.assertRaisesRegex(JSONImportError, "profondeur"):
            self.engine.preview_json_import(too_deep)

        with self.assertRaisesRegex(JSONImportError, "noeuds"):
            self.engine.preview_json_import([[] for _ in range(10_000)])

        with self.assertRaisesRegex(JSONImportError, "souvenirs"):
            self.engine.preview_json_import({f"k{index}": index for index in range(201)})

        token_heavy = {
            "items": [" ".join(f"mot{index}" for index in range(101)) for _ in range(100)]
        }
        with self.assertRaisesRegex(JSONImportError, "concepts textuels"):
            self.engine.preview_json_import(token_heavy)

        with self.assertRaisesRegex(JSONImportError, "caracteres par souvenir"):
            self.engine.preview_json_import({"value": "x" * 4_000})

        with self.assertRaisesRegex(JSONImportError, "octets"):
            self.engine.preview_json_import({"value": "x" * (1024 * 1024)})


class JSONImportServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "memory.sqlite3"
        self.engine = MemoryEngine(database_path)
        self.server = MemoryHTTPServer(("127.0.0.1", 0), self.engine, PROJECT_ROOT / "web")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.engine.close()
        self.temporary_directory.cleanup()

    def post_import(self, payload: dict) -> tuple[int, dict]:
        request = Request(
            self.base_url + "/api/import",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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

    def test_preview_commit_and_duplicate_cycle(self) -> None:
        data = {"contacts": [{"name": "Rio", "city": "Quebec"}]}
        status, preview = self.post_import(
            {"mode": "preview", "filename": "contacts.json", "data": data}
        )
        self.assertEqual(status, 200)
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["mode"], "preview")
        self.assertEqual(self.engine.stats()["events"], 0)

        status, committed = self.post_import(
            {
                "mode": "commit",
                "filename": "contacts.json",
                "data": data,
                "import_id": preview["import_id"],
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(committed["created"], 2)
        self.assertEqual(committed["duplicates"], 0)

        status, duplicate = self.post_import(
            {
                "mode": "commit",
                "filename": "contacts.json",
                "data": data,
                "import_id": preview["import_id"],
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(duplicate["created"], 0)
        self.assertEqual(duplicate["duplicates"], 2)

    def test_commit_rejects_missing_wrong_or_stale_import_id(self) -> None:
        data = {"name": "Milo"}
        _, preview = self.post_import({"mode": "preview", "data": data})

        status, missing = self.post_import({"mode": "commit", "data": data})
        self.assertEqual(status, 400)
        self.assertFalse(missing["ok"])

        status, wrong = self.post_import(
            {"mode": "commit", "data": data, "import_id": "json-v1-bad"}
        )
        self.assertEqual(status, 400)
        self.assertFalse(wrong["ok"])

        status, stale = self.post_import(
            {
                "mode": "commit",
                "data": {"name": "Mila"},
                "import_id": preview["import_id"],
            }
        )
        self.assertEqual(status, 400)
        self.assertFalse(stale["ok"])
        self.assertEqual(self.engine.stats()["events"], 0)

    def test_raw_content_preserves_integer_larger_than_javascript_safe_range(self) -> None:
        number = "900719925474099312345678901234567890"
        content = '{"account":{"exact_id":' + number + "}}"
        status, preview = self.post_import(
            {"mode": "preview", "filename": "large.json", "content": content}
        )
        self.assertEqual(status, 200)

        status, committed = self.post_import(
            {
                "mode": "commit",
                "filename": "large.json",
                "content": content,
                "import_id": preview["import_id"],
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(committed["created"], 1)
        memories = self.engine.recall(number, top_k=3)
        texts = [event["text"] for memory in memories for event in memory["events"]]
        self.assertTrue(any(number in text for text in texts))

    def test_api_rejects_non_loopback_host_even_when_origin_matches(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            evil_host = f"memory.example:{self.server.server_port}"
            connection.request(
                "GET",
                "/api/memories",
                headers={"Host": evil_host, "Origin": f"http://{evil_host}"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()
        self.assertEqual(response.status, 403)
        self.assertFalse(payload["ok"])

    def test_raw_content_rejects_duplicate_keys(self) -> None:
        content = '{"profile":{"name":"Alice","name":"Bob"}}'
        status, payload = self.post_import(
            {"mode": "preview", "filename": "duplicate.json", "content": content}
        )
        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("dupliquee", payload["error"])
        self.assertEqual(self.engine.stats()["events"], 0)


if __name__ == "__main__":
    unittest.main()
