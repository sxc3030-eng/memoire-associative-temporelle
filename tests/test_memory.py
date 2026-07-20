from __future__ import annotations

import tempfile
from pathlib import Path
import sqlite3
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent import MemoryEngine


class MemoryEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "memory.sqlite3"
        self.engine = MemoryEngine(self.database_path)

    def tearDown(self) -> None:
        self.engine.close()
        self.temporary_directory.cleanup()

    def test_observe_recall_predict_and_explain(self) -> None:
        first = self.engine.observe("4 8 3 2 10", episode_id="episode-a")
        self.engine.observe("4 8 3 2 10", episode_id="episode-b")
        self.engine.observe("4 8 3 7", episode_id="episode-c")

        memories = self.engine.recall("8 10", top_k=5)
        predictions = self.engine.predict("4 8 3", top_k=5)

        self.assertTrue(memories)
        self.assertIn("episode-a", {item["episode_id"] for item in memories})
        self.assertEqual(predictions[0]["concept"], "2")
        self.assertEqual(predictions[0]["suffix_used"], ["4", "8", "3"])
        self.assertGreaterEqual(predictions[0]["support_count"], 2)
        self.assertTrue(predictions[0]["evidence"])
        self.assertEqual(
            predictions[0]["ranking_share_kind"],
            "relative_score_not_probability",
        )
        self.assertTrue(first["event_id"])

    def test_idempotence_does_not_double_support(self) -> None:
        first = self.engine.observe(
            "alpha beta gamma",
            episode_id="episode-a",
            idempotency_key="source:event-1",
        )
        duplicate = self.engine.observe(
            "this text is ignored",
            episode_id="episode-other",
            idempotency_key="source:event-1",
        )

        prediction = self.engine.predict("alpha beta", top_k=5)
        self.assertFalse(first["duplicate"])
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(first["event_id"], duplicate["event_id"])
        self.assertEqual(prediction[0]["concept"], "gamma")
        self.assertEqual(prediction[0]["support_count"], 1)
        self.assertEqual(self.engine.stats()["events"], 1)

    def test_generated_text_is_recallable_but_does_not_reinforce(self) -> None:
        generated = self.engine.observe(
            "licorne argentée demain",
            episode_id="generated-episode",
            source="generated",
        )

        self.assertTrue(self.engine.recall("licorne"))
        self.assertEqual(self.engine.predict("licorne argentée"), [])
        self.assertEqual(self.engine.stats()["trusted_events"], 0)
        self.assertTrue(generated["event_id"])

    def test_forget_rebuilds_evidence(self) -> None:
        remembered = self.engine.observe(
            "chat aime poisson",
            episode_id="episode-a",
        )
        self.assertEqual(self.engine.predict("chat aime")[0]["concept"], "poisson")

        result = self.engine.forget(remembered["event_id"])

        self.assertTrue(result["forgotten"])
        self.assertEqual(self.engine.recall("poisson"), [])
        self.assertEqual(self.engine.predict("chat aime"), [])
        self.assertEqual(self.engine.stats()["events"], 0)

    def test_forget_middle_event_does_not_invent_shortcut(self) -> None:
        first = self.engine.observe("alpha", episode_id="chain")
        middle = self.engine.observe("beta", episode_id="chain")
        third = self.engine.observe("gamma", episode_id="chain")
        self.assertEqual(self.engine.predict("alpha")[0]["concept"], "beta")
        self.assertEqual(self.engine.predict("beta")[0]["concept"], "gamma")

        result = self.engine.forget(middle["event_id"])

        self.assertTrue(result["episode_split"])
        self.assertEqual(self.engine.predict("alpha"), [])
        self.assertEqual(self.engine.predict("beta"), [])
        alpha_episode = self.engine.recall("alpha")[0]["episode_id"]
        gamma_episode = self.engine.recall("gamma")[0]["episode_id"]
        self.assertNotEqual(alpha_episode, gamma_episode)
        self.assertTrue(first["event_id"])
        self.assertTrue(third["event_id"])

    def test_incompatible_schema_is_refused(self) -> None:
        incompatible_path = Path(self.temporary_directory.name) / "future.sqlite3"
        connection = sqlite3.connect(incompatible_path)
        try:
            connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES('schema_version', '999')"
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            MemoryEngine(incompatible_path)

    def test_persistence_after_restart(self) -> None:
        remembered = self.engine.observe(
            "Mila préfère le thé vert",
            episode_id="preferences",
        )
        self.engine.close()

        reopened = MemoryEngine(self.database_path)
        try:
            memories = reopened.recall("Mila thé")
            self.assertTrue(memories)
            self.assertEqual(memories[0]["events"][0]["text"], "Mila préfère le thé vert")
            self.assertEqual(reopened.stats()["events"], 1)
            self.assertTrue(remembered["event_id"])
        finally:
            reopened.close()
        self.engine = MemoryEngine(self.database_path)


if __name__ == "__main__":
    unittest.main()
