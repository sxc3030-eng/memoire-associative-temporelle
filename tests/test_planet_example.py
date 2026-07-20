from __future__ import annotations

import json
from pathlib import Path
import unittest

from memory_agent.memory import MemoryEngine


PLANET_DATASET = Path(__file__).resolve().parents[1] / "examples" / "planetes-nasa.json"


class PlanetExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = MemoryEngine(":memory:")
        data = json.loads(PLANET_DATASET.read_text(encoding="utf-8"))
        preview = self.engine.preview_json_import(data, filename=PLANET_DATASET.name)
        self.engine.import_json(
            data,
            import_id=preview["import_id"],
            filename=PLANET_DATASET.name,
        )

    def tearDown(self) -> None:
        self.engine.close()

    def test_natural_questions_rank_the_expected_memory_first(self) -> None:
        questions = {
            "Quelle est la plus grande planete ?": "Jupiter",
            "Quelle est la plus petite planète ?": "Mercure",
            "Quelle planète est la plus éloignée du Soleil ?": "Neptune",
            "Quel est le diametre de Saturne ?": "Saturne",
            "D'ou viennent les donnees sur les planetes ?": "NASA Science",
        }

        for question, expected in questions.items():
            with self.subTest(question=question):
                results = self.engine.recall(question, top_k=5)
                self.assertTrue(results)
                self.assertIn(expected, results[0]["text"])


if __name__ == "__main__":
    unittest.main()
