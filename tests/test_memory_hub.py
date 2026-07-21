from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.memory import MemoryEngine
from memory_agent.memory_hub import (
    GeneratedObservationError,
    MemoryAccessError,
    MemoryHub,
    SpacePolicy,
)


class MemoryHubTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.engines = {
            "alice-private": MemoryEngine(root / "alice.sqlite3"),
            "team": MemoryEngine(root / "team.sqlite3"),
            "reference": MemoryEngine(root / "reference.sqlite3"),
        }
        self.engines["reference"].observe(
            "Mercure est la planète la plus proche du Soleil",
            episode_id="reference-planets",
            source="observed",
        )
        self.hub = MemoryHub(
            self.engines,
            {
                "alice-private": SpacePolicy.private("alice"),
                "team": SpacePolicy.shared(
                    readers={"alice", "bob"}, writers={"alice"}
                ),
                "reference": SpacePolicy.reference(),
            },
        )

    def tearDown(self) -> None:
        for engine in self.engines.values():
            engine.close()
        self.temp.cleanup()

    def test_private_space_is_strictly_isolated(self) -> None:
        self.hub.observe("alice", "alice-private", "code privé améthyste")

        with self.assertRaises(MemoryAccessError):
            self.hub.recall_capsule(
                "bob", "améthyste", space_names=["alice-private"]
            )
        capsule = self.hub.recall_capsule("bob", "améthyste")
        self.assertEqual(capsule["items"], [])
        self.assertNotIn("améthyste", self.hub.capsule_json(capsule))

    def test_shared_and_reference_read_access(self) -> None:
        self.hub.observe("alice", "team", "projet commun constellation")

        shared = self.hub.recall_capsule("bob", "constellation")
        reference = self.hub.recall_capsule("bob", "Mercure")

        self.assertEqual(shared["items"][0]["space"], "team")
        self.assertEqual(reference["items"][0]["space_policy"], "reference")

    def test_capsule_respects_character_budget(self) -> None:
        for number in range(4):
            self.hub.observe(
                "alice",
                "team",
                f"budget mémoire {number} " + "x" * 700,
                episode_id=f"budget-{number}",
            )

        capsule = self.hub.recall_capsule(
            "bob", "budget mémoire", character_budget=700, top_k=4
        )
        encoded = self.hub.capsule_json(capsule)

        self.assertLessEqual(len(encoded), 700)
        self.assertEqual(capsule["budget"]["characters_used"], len(encoded))
        self.assertTrue(capsule["budget"]["truncated"])

    def test_deduplication_and_output_are_deterministic(self) -> None:
        for space in ("alice-private", "team"):
            self.hub.observe(
                "alice",
                space,
                "souvenir identique boréal",
                episode_id="same",
                idempotency_key="same-source",
            )

        first = self.hub.recall_capsule("alice", "boréal")
        second = self.hub.recall_capsule("alice", "boréal")

        self.assertEqual(self.hub.capsule_json(first), self.hub.capsule_json(second))
        self.assertEqual(len(first["items"]), 1)
        self.assertEqual(first["retrieval"]["duplicates_removed"], 1)
        self.assertEqual(
            [origin["space"] for origin in first["items"][0]["origins"]],
            ["alice-private", "team"],
        )

    def test_capsule_preserves_provenance_and_explanation(self) -> None:
        self.hub.observe("alice", "team", "preuve turquoise vérifiable")

        item = self.hub.recall_capsule("bob", "turquoise")["items"][0]

        self.assertEqual(item["events"][0]["source"], "observed")
        self.assertTrue(item["evidence"])
        self.assertEqual(item["explanation"]["algorithm_version"], "recall-v1")
        self.assertEqual(item["events"][0]["context"]["_memory_hub_agent_id"], "alice")

    def test_unauthorized_and_generated_writes_are_rejected(self) -> None:
        with self.assertRaises(MemoryAccessError):
            self.hub.observe("bob", "team", "écriture interdite")
        with self.assertRaises(MemoryAccessError):
            self.hub.observe("alice", "reference", "altération référence")
        with self.assertRaises(GeneratedObservationError):
            self.hub.observe(
                "alice", "team", "réponse du modèle", source="generated"
            )

        self.assertEqual(self.engines["team"].stats()["events"], 0)


if __name__ == "__main__":
    unittest.main()
