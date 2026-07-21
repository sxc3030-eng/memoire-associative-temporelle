from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent import MemoryEngine
from memory_agent.server import MemoryHTTPServer


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "memory.sqlite3"
        self.engine = MemoryEngine(database_path)
        self.server = MemoryHTTPServer(
            ("127.0.0.1", 0),
            self.engine,
            PROJECT_ROOT / "web",
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

    def get_json(self, path: str) -> dict:
        with urlopen(self.base_url + path, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def chat(self, message: str) -> dict:
        request = Request(
            self.base_url + "/api/chat",
            data=json.dumps({"message": message}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_health_interface_and_conversation_cycle(self) -> None:
        health = self.get_json("/api/health")
        with urlopen(self.base_url + "/", timeout=5) as response:
            interface = response.read().decode("utf-8")

        remembered = self.chat("Souviens-toi que Rio aime courir dans le parc")
        recalled = self.chat("De quoi te souviens-tu au sujet de Rio ?")
        predicted = self.chat("Qu'est-ce qui vient après Rio aime ?")
        memories = self.get_json("/api/memories?limit=10")
        stats = self.get_json("/api/stats")

        self.assertTrue(health["ok"])
        self.assertIn("Mémoire vivante", interface)
        self.assertIn('data-view="focus"', interface)
        self.assertIn("/assets/matlm-icon-192.png", interface)
        self.assertIn("/assets/matlm-dashboard-hero.webp", interface)
        self.assertEqual(remembered["intent"], "observe")
        self.assertEqual(recalled["intent"], "recall")
        self.assertEqual(predicted["intent"], "predict")
        self.assertTrue(memories["memories"])
        self.assertEqual(stats["stats"]["events"], 1)

        event_id = remembered["data"]["event_id"]
        forgotten = self.chat(f"Oublie {event_id}")
        self.assertTrue(forgotten["data"]["forgotten"])
        self.assertEqual(self.get_json("/api/stats")["stats"]["events"], 0)

    def test_conversation_learns_between_messages(self) -> None:
        first = self.chat("Souviens-toi que alpha")
        second = self.chat("Souviens-toi que beta")
        prediction = self.chat("Qu'est-ce qui vient après alpha ?")
        knowledge = self.chat("Que sais-tu de beta ?")

        self.assertEqual(first["data"]["episode_id"], second["data"]["episode_id"])
        self.assertEqual(prediction["intent"], "predict")
        self.assertEqual(prediction["data"][0]["concept"], "beta")
        self.assertEqual(knowledge["intent"], "recall")
        self.assertTrue(knowledge["data"])

    def test_brand_assets_are_served_and_cacheable(self) -> None:
        assets = (
            "/assets/matlm-favicon.ico",
            "/assets/matlm-icon-64.png",
            "/assets/matlm-icon-192.png",
            "/assets/matlm-apple-touch-icon.png",
            "/assets/matlm-dashboard-hero.webp",
        )
        for path in assets:
            with self.subTest(path=path):
                with urlopen(self.base_url + path, timeout=5) as response:
                    body = response.read()
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.headers.get_content_type().startswith("image/"))
                    self.assertIn("max-age=3600", response.headers.get("Cache-Control", ""))
                    self.assertGreater(len(body), 1000)

        request = Request(
            self.base_url + "/assets/matlm-dashboard-hero.webp",
            method="HEAD",
        )
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), "image/webp")
            self.assertEqual(response.read(), b"")

    def test_server_refuses_non_loopback_host(self) -> None:
        with self.assertRaisesRegex(ValueError, "uniquement"):
            MemoryHTTPServer(("0.0.0.0", 0), self.engine, PROJECT_ROOT / "web")


if __name__ == "__main__":
    unittest.main()
