from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.memory_native_curriculum import (  # noqa: E402
    ABSTENTION_MARKER,
    SYNTHETIC_TASK_ORDER,
    TASK_ORDER,
    audit_curriculum_isolation,
    build_memory_native_curriculum,
    build_synthetic_memory_curriculum,
    encode_training_jsonl,
    load_facts_only_corpus,
    write_training_jsonl,
)
from memory_agent.matlm_bridge import strict_chat_messages  # noqa: E402
from memory_agent.native_llm_contract import (  # noqa: E402
    ANSWER_SCHEMA_VERSION,
    CAPSULE_SCHEMA_VERSION,
    validate_answer,
    validate_capsule,
)
from memory_agent.matlm_training import load_training_jsonl  # noqa: E402


DATASET = PROJECT_ROOT / "examples" / "science-biographies-v1.json"
BENCHMARK_KEYS = {
    "evaluation_questions",
    "expected_answer_fragments",
    "forbidden_answer_fragments",
    "supporting_claim_ids",
    "answer_status",
    "evaluation_kind",
}


class MemoryNativeCurriculumTests(unittest.TestCase):
    def test_builds_all_tasks_with_grounding_and_explicit_calculation(self) -> None:
        source = json.loads(DATASET.read_text(encoding="utf-8"))
        known_claims = {row["id"] for row in source["claims"]}
        known_sources = {row["id"] for row in source["sources"]}
        result = build_memory_native_curriculum(DATASET)

        self.assertEqual(tuple(result["task_counts"]), TASK_ORDER)
        self.assertEqual(result["example_count"], len(result["examples"]))
        self.assertTrue(all(result["task_counts"][task] > 0 for task in TASK_ORDER))

        for example in result["examples"]:
            capsule_claims = {
                fact["claim_id"] for fact in example["memory_capsule"]["facts"]
            }
            capsule_sources = {
                row["id"] for row in example["memory_capsule"]["sources"]
            }
            self.assertLessEqual(capsule_claims, known_claims)
            self.assertLessEqual(capsule_sources, known_sources)
            self.assertEqual(
                example["provenance"]["facts_sha256"], result["facts_sha256"]
            )

        direct = next(
            row for row in result["examples"] if row["task"] == TASK_ORDER[0]
        )
        for identifier in (
            direct["target"]["citations"]["claim_ids"]
            + direct["target"]["citations"]["source_ids"]
        ):
            self.assertIn(f"[{identifier}]", direct["target"]["answer"])

        age = next(row for row in result["examples"] if row["task"] == TASK_ORDER[2])
        self.assertEqual(age["messages"][2]["tool_calls"][0]["function"]["name"], "calendar_age")
        self.assertEqual(age["messages"][3]["role"], "tool")
        self.assertEqual(age["target"]["calculation"]["operation"], "calendar_age")

        abstention = next(
            row for row in result["examples"] if row["task"] == TASK_ORDER[5]
        )
        self.assertTrue(abstention["target"]["answer"].startswith(ABSTENTION_MARKER))
        self.assertEqual(
            abstention["target"]["citations"], {"claim_ids": [], "source_ids": []}
        )
        self.assertEqual(abstention["target"]["grounding"], "unsupported")

    def test_benchmark_payload_is_ignored_and_cannot_change_training(self) -> None:
        baseline = build_memory_native_curriculum(DATASET)
        document = json.loads(DATASET.read_text(encoding="utf-8"))
        document["evaluation_questions"] = [
            {
                "id": "LEAK-ID-SENTINEL",
                "question": "LEAK-QUESTION-SENTINEL",
                "expected_answer_fragments": ["LEAK-EXPECTED-SENTINEL"],
                "forbidden_answer_fragments": ["LEAK-FORBIDDEN-SENTINEL"],
                "supporting_claim_ids": ["LEAK-SUPPORT-SENTINEL"],
                "answer_status": "LEAK-STATUS-SENTINEL",
                "evaluation_kind": "LEAK-KIND-SENTINEL",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            modified = Path(directory) / "modified.json"
            modified.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            candidate = build_memory_native_curriculum(modified)

        self.assertEqual(candidate, baseline)
        encoded = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("LEAK-", encoded)
        for key in BENCHMARK_KEYS:
            self.assertNotIn(f'"{key}"', encoded)

    def test_facts_only_file_without_benchmark_section_is_supported(self) -> None:
        document = json.loads(DATASET.read_text(encoding="utf-8"))
        document.pop("evaluation_questions")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facts-only.json"
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            projection = load_facts_only_corpus(path)
            result = build_memory_native_curriculum(path, maximum_per_task=1)

        self.assertEqual(len(projection["claims"]), len(document["claims"]))
        self.assertEqual(result["example_count"], len(TASK_ORDER))
        self.assertEqual(set(result["task_counts"].values()), {1})

    def test_age_calculations_cover_exact_and_imprecise_dates(self) -> None:
        result = build_memory_native_curriculum(DATASET)
        ages = {
            row["example_id"]: row["target"]["calculation"]
            for row in result["examples"]
            if row["task"] == TASK_ORDER[2]
        }
        marie = ages["date-age--claim-marie-born--claim-marie-polonium"]
        fleming = ages["date-age--claim-fleming-born--claim-fleming-discovery"]

        self.assertEqual(marie["result_kind"], "exact")
        self.assertEqual(marie["years"], 30)
        self.assertEqual(fleming["result_kind"], "range")
        self.assertEqual((fleming["minimum_years"], fleming["maximum_years"]), (46, 47))

    def test_role_and_causal_targets_preserve_documented_limits(self) -> None:
        result = build_memory_native_curriculum(DATASET)
        roles = [row for row in result["examples"] if row["task"] == TASK_ORDER[3]]
        causal = [row for row in result["examples"] if row["task"] == TASK_ORDER[4]]

        self.assertGreaterEqual(len(roles), 2)
        self.assertTrue(all(len(row["target"]["citations"]["claim_ids"]) >= 2 for row in roles))
        self.assertTrue(causal)
        self.assertTrue(all(row["target"]["grounding"] == "supported_with_causal_limit" for row in causal))
        self.assertTrue(all("Non." in row["target"]["answer"] for row in causal))

    def test_jsonl_is_canonical_deterministic_and_round_trips(self) -> None:
        result = build_memory_native_curriculum(DATASET, maximum_per_task=2)
        first = encode_training_jsonl(result["examples"])
        second = encode_training_jsonl(result["examples"])
        self.assertEqual(first, second)
        rows = [json.loads(line) for line in first.splitlines()]
        self.assertEqual(rows, result["examples"])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "curriculum.jsonl"
            write_training_jsonl(output, result["examples"])
            self.assertEqual(output.read_text(encoding="utf-8"), first)


class SyntheticMemoryNativeCurriculumTests(unittest.TestCase):
    def test_exact_balanced_count_and_all_nine_training_families(self) -> None:
        result = build_synthetic_memory_curriculum(seed=731, count=2_007)

        self.assertTrue(result["synthetic"])
        self.assertEqual(result["example_count"], 2_007)
        self.assertEqual(len(result["examples"]), 2_007)
        self.assertEqual(tuple(result["task_counts"]), SYNTHETIC_TASK_ORDER)
        self.assertLessEqual(
            max(result["task_counts"].values()) - min(result["task_counts"].values()),
            1,
        )
        self.assertTrue(all(count > 200 for count in result["task_counts"].values()))

    def test_same_seed_is_identical_and_another_seed_changes_every_world(self) -> None:
        first = build_synthetic_memory_curriculum(seed=17, count=90)
        repeated = build_synthetic_memory_curriculum(seed=17, count=90)
        other = build_synthetic_memory_curriculum(seed=18, count=90)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first["generator_sha256"], other["generator_sha256"])
        self.assertTrue(
            set(row["example_id"] for row in first["examples"]).isdisjoint(
                row["example_id"] for row in other["examples"]
            )
        )

    def test_zero_science_identity_claim_and_source_collision(self) -> None:
        result = build_synthetic_memory_curriculum(seed=20_260_721, count=2_025)
        audit = audit_curriculum_isolation(result, DATASET)
        science = json.loads(DATASET.read_text(encoding="utf-8"))
        encoded = encode_training_jsonl(result["examples"]).casefold()

        self.assertTrue(audit["passed"], audit["collisions"])
        self.assertEqual(audit["collision_count"], 0)
        for entity in science["entities"]:
            self.assertNotIn(entity["id"].casefold(), encoded)
            self.assertNotIn(entity["name"].casefold(), encoded)
        for claim in science["claims"]:
            self.assertNotIn(claim["id"].casefold(), encoded)
            self.assertNotIn(claim["statement_fr"].casefold(), encoded)
        for source in science["sources"]:
            for field in ("id", "title", "url", "publisher"):
                self.assertNotIn(source[field].casefold(), encoded)

    def test_behavior_contracts_are_encoded_without_external_facts(self) -> None:
        result = build_synthetic_memory_curriculum(seed=99, count=18)
        by_task = {
            task: next(row for row in result["examples"] if row["task"] == task)
            for task in SYNTHETIC_TASK_ORDER
        }

        direct = by_task["direct_recall_with_citations"]
        self.assertEqual(direct["memory_capsule"]["schema_version"], CAPSULE_SCHEMA_VERSION)
        self.assertEqual(direct["target"]["schema_version"], ANSWER_SCHEMA_VERSION)
        self.assertTrue(direct["target"]["evidence_ids"])

        age = by_task["date_age_arithmetic"]
        self.assertEqual(len(age["target"]["calculations"]), 1)
        self.assertIn("calendar_age", age["target"]["calculations"][0]["expression"])
        self.assertTrue(age["target"]["calculations"][0]["evidence_ids"])

        distractor = by_task["distractor_rejection"]
        self.assertGreater(len(distractor["memory_capsule"]["evidence"]), 1)
        self.assertEqual(len(distractor["target"]["evidence_ids"]), 1)

        abstention = by_task["unsupported_abstention"]
        self.assertTrue(abstention["target"]["answer"].startswith(ABSTENTION_MARKER))
        self.assertEqual(abstention["target"]["evidence_ids"], [])
        self.assertTrue(abstention["target"]["abstention"]["abstained"])

        contradiction = by_task["contradiction_resolution"]
        self.assertEqual(len(contradiction["target"]["evidence_ids"]), 2)

        provenance = by_task["provenance_selection"]
        self.assertIn("non vérifié", provenance["target"]["answer"])

    def test_every_synthetic_capsule_and_last_answer_validate_against_real_contract(self) -> None:
        result = build_synthetic_memory_curriculum(seed=4_242, count=2_007)

        for example in result["examples"]:
            capsule = validate_capsule(example["memory_capsule"])
            for evidence in capsule["evidence"]:
                self.assertRegex(evidence["evidence_id"], r"^ev:[0-9a-f]{20}$")
            parsed_last_answer = json.loads(example["messages"][-1]["content"])
            validated = validate_answer(parsed_last_answer, capsule)
            self.assertEqual(validated, example["target"])
            self.assertEqual(example["messages"][:-1], strict_chat_messages(capsule))
        encoded = encode_training_jsonl(result["examples"])
        self.assertNotIn("syn-claim-", encoded)
        self.assertNotIn("syn-source-", encoded)
        self.assertNotIn("facts-only-capsule-v1", encoded)

    def test_generated_jsonl_is_accepted_by_matlm_training_loader(self) -> None:
        result = build_synthetic_memory_curriculum(seed=8_181, count=90)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic-train.jsonl"
            write_training_jsonl(output, result["examples"])
            loaded = load_training_jsonl(output)

        self.assertEqual(loaded.summary.example_count, 90)
        self.assertEqual(loaded.summary.task_counts, result["task_counts"])
        self.assertEqual(loaded.summary.provenance_sha256, (result["generator_sha256"],))

    def test_science_derived_builder_is_explicitly_not_evaluation_safe(self) -> None:
        result = build_memory_native_curriculum(DATASET, maximum_per_task=1)

        self.assertFalse(result["synthetic"])
        self.assertEqual(
            result["evaluation_eligibility"], "invalid_against_the_source_corpus"
        )
        self.assertIn("ne doivent jamais entraîner", result["warning"])


if __name__ == "__main__":
    unittest.main()
