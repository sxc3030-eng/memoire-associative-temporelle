from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.memory_native_curriculum import (  # noqa: E402
    build_synthetic_memory_curriculum,
)


TRAIN_PATH = PROJECT_ROOT / "training-data" / "matlm-train-v8.jsonl"
EVAL_PATH = PROJECT_ROOT / "training-data" / "matlm-dev-v8.jsonl"
TRAIN_MANIFEST = TRAIN_PATH.with_suffix(".manifest.json")
EVAL_MANIFEST = EVAL_PATH.with_suffix(".manifest.json")


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identifier_values(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "id" or key.endswith("_id"):
                if isinstance(child, str) and child:
                    found.add(child)
            elif key.endswith("_ids") and isinstance(child, list):
                found.update(item for item in child if isinstance(item, str) and item)
            found.update(_identifier_values(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_identifier_values(child))
    return found


@dataclass(frozen=True)
class SplitInventory:
    rows: int
    example_ids: frozenset[str]
    all_ids: frozenset[str]
    synthetic_worlds: frozenset[str]
    evidence_ids: frozenset[str]
    generator_sha256: frozenset[str]
    target_answer_sha256: frozenset[str]
    assistant_target_sha256: frozenset[str]
    target_object_sha256: frozenset[str]
    evidence_text_sha256: frozenset[str]


def _inventory(rows: list[dict]) -> SplitInventory:
    example_ids: set[str] = set()
    all_ids: set[str] = set()
    worlds: set[str] = set()
    evidence_ids: set[str] = set()
    generators: set[str] = set()
    answer_hashes: set[str] = set()
    assistant_hashes: set[str] = set()
    target_hashes: set[str] = set()
    evidence_text_hashes: set[str] = set()

    for row in rows:
        example_ids.add(row["example_id"])
        all_ids.update(_identifier_values(row))
        provenance = row["provenance"]
        worlds.add(provenance["synthetic_world"])
        generators.add(provenance["generator_sha256"])
        evidence_ids.update(provenance["evidence_ids"])

        target = row["target"]
        answer_hashes.add(_text_sha256(target["answer"]))
        target_hashes.add(_json_sha256(target))
        for message in row["messages"]:
            if message.get("role") == "assistant" and message.get("content"):
                assistant_hashes.add(_text_sha256(message["content"]))
        for evidence in row["memory_capsule"]["evidence"]:
            evidence_ids.add(evidence["evidence_id"])
            evidence_text_hashes.add(_text_sha256(evidence["text"]))

    return SplitInventory(
        rows=len(rows),
        example_ids=frozenset(example_ids),
        all_ids=frozenset(all_ids),
        synthetic_worlds=frozenset(worlds),
        evidence_ids=frozenset(evidence_ids),
        generator_sha256=frozenset(generators),
        target_answer_sha256=frozenset(answer_hashes),
        assistant_target_sha256=frozenset(assistant_hashes),
        target_object_sha256=frozenset(target_hashes),
        evidence_text_sha256=frozenset(evidence_text_hashes),
    )


def _overlaps(train: SplitInventory, evaluation: SplitInventory) -> dict[str, set[str]]:
    fields = (
        "example_ids",
        "all_ids",
        "synthetic_worlds",
        "evidence_ids",
        "generator_sha256",
        "target_answer_sha256",
        "assistant_target_sha256",
        "target_object_sha256",
        "evidence_text_sha256",
    )
    return {
        field: set(getattr(train, field) & getattr(evaluation, field))
        for field in fields
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class MATLMSplitIsolationTests(unittest.TestCase):
    def assert_no_overlap(self, train: SplitInventory, evaluation: SplitInventory) -> None:
        collisions = {key: value for key, value in _overlaps(train, evaluation).items() if value}
        self.assertEqual(collisions, {})

    def assert_unique_target_answers(self, split: SplitInventory) -> None:
        self.assertEqual(len(split.target_answer_sha256), split.rows)

    def test_actual_generated_artifacts_have_zero_train_eval_overlap(self) -> None:
        required = (TRAIN_PATH, EVAL_PATH, TRAIN_MANIFEST, EVAL_MANIFEST)
        if not all(path.is_file() for path in required):
            self.skipTest("artefacts MAT-LM locaux absents")

        train_rows = _read_jsonl(TRAIN_PATH)
        evaluation_rows = _read_jsonl(EVAL_PATH)
        train_manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
        evaluation_manifest = json.loads(EVAL_MANIFEST.read_text(encoding="utf-8"))
        train = _inventory(train_rows)
        evaluation = _inventory(evaluation_rows)

        self.assertEqual(train.rows, train_manifest["example_count"])
        self.assertEqual(evaluation.rows, evaluation_manifest["example_count"])
        self.assertEqual(len(train.example_ids), train.rows)
        self.assertEqual(len(evaluation.example_ids), evaluation.rows)
        self.assertEqual(len(train.synthetic_worlds), train.rows)
        self.assertEqual(len(evaluation.synthetic_worlds), evaluation.rows)
        self.assertEqual(
            train.generator_sha256, frozenset({train_manifest["generator_sha256"]})
        )
        self.assertEqual(
            evaluation.generator_sha256,
            frozenset({evaluation_manifest["generator_sha256"]}),
        )
        self.assertTrue(train_manifest["isolation_audit"]["passed"])
        self.assertTrue(evaluation_manifest["isolation_audit"]["passed"])
        self.assert_unique_target_answers(train)
        self.assert_unique_target_answers(evaluation)
        self.assert_no_overlap(train, evaluation)

    def test_independent_seeds_are_disjoint_by_contract(self) -> None:
        train = build_synthetic_memory_curriculum(seed=20_260_721, count=90)
        evaluation = build_synthetic_memory_curriculum(seed=20_260_722, count=90)

        train_inventory = _inventory(train["examples"])
        evaluation_inventory = _inventory(evaluation["examples"])
        self.assert_unique_target_answers(train_inventory)
        self.assert_unique_target_answers(evaluation_inventory)
        self.assert_no_overlap(train_inventory, evaluation_inventory)


if __name__ == "__main__":
    unittest.main()
