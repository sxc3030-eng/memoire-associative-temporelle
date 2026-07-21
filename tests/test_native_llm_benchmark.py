from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.native_llm_benchmark import (
    ComparisonCase,
    ComparisonMode,
    SequentialComparisonHarness,
    default_comparison_plan,
)
from memory_agent.native_llm_contract import ANSWER_SCHEMA_VERSION, build_capsule


def benchmark_capsule() -> dict[str, object]:
    return build_capsule(
        request_id="bench-001",
        question="Quelle année est mémorisée ?",
        evidence=[
            {
                "evidence_id": "ref:event:1898",
                "text": "L'événement de référence date de 1898.",
                "space": "reference",
                "status": "verified",
                "confidence": 1.0,
                "temporal_context": "1898",
                "tags": ["histoire"],
            }
        ],
        max_evidence_ids=2,
    )


class FakeSession:
    def __init__(self, provider: "TrackingProvider", model_id: str) -> None:
        self.provider = provider
        self.model_id = model_id

    def generate_json(self, capsule, *, output_schema, mode):
        self.provider.calls.append((mode, self.model_id, capsule))
        self.provider.schema_versions.append(output_schema["properties"]["schema_version"]["const"])
        if mode is self.provider.invalid_mode:
            return {"not": "the contract"}
        evidence_ids = [] if mode is ComparisonMode.BASELINE else ["ref:event:1898"]
        return {
            "schema_version": ANSWER_SCHEMA_VERSION,
            "request_id": capsule["request_id"],
            "answer": "1898",
            "confidence": 0.9,
            "evidence_ids": evidence_ids,
            "calculations": [],
            "abstention": {
                "abstained": False,
                "reason": "none",
                "missing_information": [],
            },
        }


class FakeContext:
    def __init__(self, provider: "TrackingProvider", model_id: str) -> None:
        self.provider = provider
        self.model_id = model_id

    def __enter__(self):
        self.provider.active += 1
        self.provider.maximum_active = max(self.provider.maximum_active, self.provider.active)
        self.provider.opened.append(self.model_id)
        return FakeSession(self.provider, self.model_id)

    def __exit__(self, error_type, error, traceback):
        self.provider.closed.append(self.model_id)
        self.provider.active -= 1
        return False


class TrackingProvider:
    def __init__(self, invalid_mode=None) -> None:
        self.invalid_mode = invalid_mode
        self.active = 0
        self.maximum_active = 0
        self.opened = []
        self.closed = []
        self.calls = []
        self.schema_versions = []

    def open(self, model_id):
        return FakeContext(self, model_id)


class NativeLLMBenchmarkTests(unittest.TestCase):
    def test_three_arms_run_in_order_with_only_one_open_model(self) -> None:
        provider = TrackingProvider()
        harness = SequentialComparisonHarness(provider)
        case = ComparisonCase("science-001", benchmark_capsule(), "science-held-out")
        plan = default_comparison_plan(
            general_model_id="small-general:1b",
            specialized_model_id="memory-native:1b",
        )

        results = harness.run(case, plan)

        self.assertEqual([result.mode for result in results], list(ComparisonMode))
        self.assertEqual([result.status for result in results], ["ok", "ok", "ok"])
        self.assertEqual(provider.maximum_active, 1)
        self.assertEqual(provider.active, 0)
        self.assertEqual(provider.opened, provider.closed)
        self.assertEqual(provider.opened, ["small-general:1b", "small-general:1b", "memory-native:1b"])
        self.assertEqual(provider.calls[0][2]["evidence"], [])
        self.assertEqual(len(provider.calls[1][2]["evidence"]), 1)
        self.assertEqual(len(provider.calls[2][2]["evidence"]), 1)

    def test_invalid_arm_is_recorded_and_does_not_stop_next_model(self) -> None:
        provider = TrackingProvider(invalid_mode=ComparisonMode.MEMORY)
        harness = SequentialComparisonHarness(provider)
        results = harness.run(
            ComparisonCase("science-002", benchmark_capsule()),
            default_comparison_plan(
                general_model_id="small-general:1b",
                specialized_model_id="memory-native:1b",
            ),
        )

        self.assertEqual(
            [result.status for result in results],
            ["ok", "invalid_output", "ok"],
        )
        self.assertEqual(provider.active, 0)
        self.assertEqual(len(provider.calls), 3)

    def test_plan_requires_same_general_model_for_fair_comparison(self) -> None:
        provider = TrackingProvider()
        harness = SequentialComparisonHarness(provider)
        plan = list(
            default_comparison_plan(
                general_model_id="small-general:1b",
                specialized_model_id="memory-native:1b",
            )
        )
        plan[1] = type(plan[1])(ComparisonMode.MEMORY, "different-general:1b")

        with self.assertRaisesRegex(ValueError, "même modèle général"):
            harness.run(ComparisonCase("science-003", benchmark_capsule()), plan)


if __name__ == "__main__":
    unittest.main()
