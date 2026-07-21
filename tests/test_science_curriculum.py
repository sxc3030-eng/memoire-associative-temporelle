from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.science_curriculum import (  # noqa: E402
    ScienceCurriculumError,
    build_science_capsules,
    load_science_dataset,
)
from memory_agent.memory import MemoryEngine  # noqa: E402


DATASET = PROJECT_ROOT / "examples" / "science-biographies-v1.json"
EVALUATION_KEYS = {
    "evaluation_questions",
    "expected_answer_fragments",
    "forbidden_answer_fragments",
    "supporting_claim_ids",
    "answer_status",
    "evaluation_kind",
}


class ScienceCurriculumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dataset = Path(self.temp.name) / "science.json"
        self.dataset.write_text(DATASET.read_text(encoding="utf-8"), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_validates_imports_claims_and_reports_support_coverage(self) -> None:
        document = json.loads(self.dataset.read_text(encoding="utf-8"))
        result = build_science_capsules(self.dataset)

        self.assertEqual(result["schema_version"], "science-curriculum-output-v1")
        self.assertEqual(
            result["diagnostics"]["claims_imported"], len(document["claims"])
        )
        self.assertEqual(
            result["diagnostics"]["questions_recalled"],
            len(document["evaluation_questions"]),
        )
        self.assertEqual(result["diagnostics"]["supporting_claim_coverage"], 1.0)
        self.assertTrue(
            all(row["coverage"] == 1.0 for row in result["diagnostics"]["questions"])
        )
        self.assertTrue(result["diagnostics"]["temporary_reference_removed"])

    def test_capsules_do_not_leak_evaluation_fields(self) -> None:
        result = build_science_capsules(self.dataset)
        capsules = result["capsules"]
        encoded = json.dumps(capsules, ensure_ascii=False, sort_keys=True)

        for key in EVALUATION_KEYS:
            self.assertNotIn(f'"{key}"', encoded)
        self.assertEqual(
            set(capsules),
            {
                row["id"]
                for row in json.loads(self.dataset.read_text(encoding="utf-8"))[
                    "evaluation_questions"
                ]
            },
        )
        self.assertTrue(any(capsule["items"] for capsule in capsules.values()))
        for capsule in capsules.values():
            compact = json.dumps(
                capsule, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            self.assertLessEqual(len(compact), capsule["budget"]["character_limit"])
            self.assertEqual(len(compact), capsule["budget"]["characters_used"])

    def test_evaluation_values_never_enter_observations_or_capsules(self) -> None:
        document = json.loads(self.dataset.read_text(encoding="utf-8"))
        question = document["evaluation_questions"][0]
        sentinels = {
            "leak-query-sentinel",
            "leak-expected-sentinel",
            "leak-forbidden-sentinel",
            "leak-evaluation-kind-sentinel",
        }
        question["question"] += " leak-query-sentinel"
        question["expected_answer_fragments"] = ["leak-expected-sentinel"]
        question["forbidden_answer_fragments"] = ["leak-forbidden-sentinel"]
        question["evaluation_kind"] = "leak-evaluation-kind-sentinel"
        self.dataset.write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8"
        )

        observations: list[dict[str, object]] = []
        original_observe = MemoryEngine.observe

        def recording_observe(
            engine: MemoryEngine, text: str, *args: object, **kwargs: object
        ) -> dict[str, object]:
            observations.append({"text": text, "args": args, "kwargs": kwargs})
            return original_observe(engine, text, *args, **kwargs)

        with patch.object(MemoryEngine, "observe", new=recording_observe):
            result = build_science_capsules(self.dataset)

        stored = json.dumps(observations, ensure_ascii=False, sort_keys=True)
        capsules = json.dumps(result["capsules"], ensure_ascii=False, sort_keys=True)
        for sentinel in sentinels:
            self.assertNotIn(sentinel, stored)
            self.assertNotIn(sentinel, capsules)

    def test_output_is_deterministic_across_temporary_memories(self) -> None:
        first = build_science_capsules(self.dataset)
        second = build_science_capsules(self.dataset)

        self.assertEqual(first, second)

    def test_unknown_supporting_claim_is_rejected_before_import(self) -> None:
        document = json.loads(self.dataset.read_text(encoding="utf-8"))
        document["evaluation_questions"][0]["supporting_claim_ids"] = ["claim-absent"]
        self.dataset.write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8"
        )

        with self.assertRaisesRegex(ScienceCurriculumError, "affirmation inconnue"):
            load_science_dataset(self.dataset)


if __name__ == "__main__":
    unittest.main()
