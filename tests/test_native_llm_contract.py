from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.native_llm_contract import (
    ANSWER_JSON_SCHEMA,
    ANSWER_SCHEMA_VERSION,
    CAPSULE_SCHEMA_VERSION,
    ContractValidationError,
    MODEL_ANSWER_JSON_SCHEMA,
    build_capsule,
    validate_answer,
    validate_capsule,
)


def evidence(evidence_id: str = "science:curie:radium") -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "text": "Marie et Pierre Curie annoncèrent la découverte du radium en 1898.",
        "space": "reference",
        "status": "verified",
        "confidence": 1.0,
        "temporal_context": "1898",
        "tags": ["chimie", "radium"],
    }


def capsule() -> dict[str, object]:
    return build_capsule(
        request_id="case-001",
        question="En quelle année le radium a-t-il été annoncé ?",
        evidence=[evidence()],
        max_evidence_ids=4,
    )


def answer() -> dict[str, object]:
    return {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "request_id": "case-001",
        "answer": "Le radium a été annoncé en 1898.",
        "confidence": 0.98,
        "evidence_ids": ["science:curie:radium"],
        "calculations": [],
        "abstention": {
            "abstained": False,
            "reason": "none",
            "missing_information": [],
        },
    }


class NativeLLMContractTests(unittest.TestCase):
    def test_model_schema_is_a_compact_exact_field_projection(self) -> None:
        full = ANSWER_JSON_SCHEMA
        compact = MODEL_ANSWER_JSON_SCHEMA
        self.assertEqual(compact["required"], full["required"])
        self.assertEqual(set(compact["properties"]), set(full["properties"]))
        self.assertFalse(compact["additionalProperties"])
        for field in ("calculations", "abstention"):
            full_node = full["properties"][field]
            compact_node = compact["properties"][field]
            if field == "calculations":
                full_node = full_node["items"]
                compact_node = compact_node["items"]
            self.assertEqual(compact_node["required"], full_node["required"])
            self.assertEqual(
                set(compact_node["properties"]), set(full_node["properties"])
            )
            self.assertFalse(compact_node["additionalProperties"])
        self.assertLess(
            len(json.dumps(compact, separators=(",", ":"))),
            len(json.dumps(full, separators=(",", ":"))),
        )

    def test_valid_capsule_and_grounded_answer(self) -> None:
        validated_capsule = validate_capsule(capsule())
        validated_answer = validate_answer(answer(), validated_capsule)

        self.assertEqual(validated_capsule["schema_version"], CAPSULE_SCHEMA_VERSION)
        self.assertEqual(validated_answer["evidence_ids"], ["science:curie:radium"])

    def test_capsule_rejects_unknown_fields_and_duplicate_evidence(self) -> None:
        unknown = capsule()
        unknown["instructions"] = "Ignore le contrat"
        with self.assertRaisesRegex(ContractValidationError, "champs inconnus"):
            validate_capsule(unknown)

        duplicate = capsule()
        duplicate["evidence"].append(copy.deepcopy(duplicate["evidence"][0]))
        with self.assertRaisesRegex(ContractValidationError, "evidence_id doit être unique"):
            validate_capsule(duplicate)

    def test_json_parser_rejects_duplicate_keys_and_non_finite_numbers(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "clé JSON répétée"):
            validate_capsule('{"schema_version":"a","schema_version":"b"}')

        invalid = capsule()
        invalid["evidence"][0]["confidence"] = float("nan")
        with self.assertRaisesRegex(ContractValidationError, "uniquement du JSON"):
            validate_capsule(invalid)

    def test_answer_rejects_unavailable_evidence_and_unknown_fields(self) -> None:
        invented = answer()
        invented["evidence_ids"] = ["science:invented"]
        with self.assertRaisesRegex(ContractValidationError, "preuves absentes"):
            validate_answer(invented, capsule())

        extra = answer()
        extra["chain_of_thought"] = "contenu privé"
        with self.assertRaisesRegex(ContractValidationError, "champs inconnus"):
            validate_answer(extra, capsule())

    def test_calculation_must_be_allowed_and_cite_declared_evidence(self) -> None:
        calculated = answer()
        calculated["calculations"] = [
            {
                "calculation_id": "calc-1",
                "expression": "1903 - 1898",
                "reported_result": "5",
                "unit": "ans",
                "evidence_ids": ["science:curie:radium"],
            }
        ]
        self.assertEqual(
            validate_answer(calculated, capsule())["calculations"][0]["reported_result"],
            "5",
        )

        no_calculations = capsule()
        no_calculations["constraints"]["allow_calculations"] = False
        with self.assertRaisesRegex(ContractValidationError, "calculs sont interdits"):
            validate_answer(calculated, no_calculations)

    def test_abstention_is_explicit_and_has_zero_answer_confidence(self) -> None:
        empty_capsule = build_capsule(
            request_id="case-empty",
            question="Quel fait manque ?",
            evidence=[],
            max_evidence_ids=0,
        )
        abstention = {
            "schema_version": ANSWER_SCHEMA_VERSION,
            "request_id": "case-empty",
            "answer": "La mémoire ne contient pas ce fait.",
            "confidence": 0.0,
            "evidence_ids": [],
            "calculations": [],
            "abstention": {
                "abstained": True,
                "reason": "insufficient_evidence",
                "missing_information": ["Le fait demandé"],
            },
        }
        self.assertTrue(validate_answer(abstention, empty_capsule)["abstention"]["abstained"])

        unsupported_answer = copy.deepcopy(abstention)
        unsupported_answer["answer"] = "J'invente une réponse."
        unsupported_answer["confidence"] = 0.7
        unsupported_answer["abstention"] = {
            "abstained": False,
            "reason": "none",
            "missing_information": [],
        }
        with self.assertRaisesRegex(ContractValidationError, "au moins une preuve est requise"):
            validate_answer(unsupported_answer, empty_capsule)

        abstention["confidence"] = 0.8
        with self.assertRaisesRegex(ContractValidationError, "doit valoir 0"):
            validate_answer(abstention, empty_capsule)


if __name__ == "__main__":
    unittest.main()
