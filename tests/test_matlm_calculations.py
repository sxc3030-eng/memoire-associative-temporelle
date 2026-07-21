from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.matlm_calculations import (  # noqa: E402
    MATLMCalculationError,
    reexecute_matlm_calculations,
)
from memory_agent.native_llm_contract import (  # noqa: E402
    ANSWER_SCHEMA_VERSION,
    ContractValidationError,
    build_capsule,
)


REQUEST_ID = "calendar-case-01"
EVIDENCE_IDS = ["ev:birth", "ev:event"]


def capsule(*, max_answer_characters: int = 4_000, max_calculations: int = 4):
    return build_capsule(
        request_id=REQUEST_ID,
        question="Quel âge civil la mémoire permet-elle de calculer ?",
        evidence=[
            {
                "evidence_id": "ev:birth",
                "text": "La personne est née le 2000-06-15.",
                "space": "reference",
                "status": "verified",
                "confidence": 1.0,
                "temporal_context": "2000-06-15",
                "tags": ["birth"],
            },
            {
                "evidence_id": "ev:event",
                "text": "L'événement a eu lieu en 2032.",
                "space": "reference",
                "status": "verified",
                "confidence": 1.0,
                "temporal_context": "2032",
                "tags": ["event"],
            },
        ],
        allow_calculations=True,
        max_answer_characters=max_answer_characters,
        max_evidence_ids=2,
        max_calculations=max_calculations,
    )


def response(
    expression: str = "calendar_age(2000-06-15,2032,precision=year)",
    *,
    answer: str = "À cette date, le calcul donne 999 ans, selon la mémoire.",
):
    return {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "request_id": REQUEST_ID,
        "answer": answer,
        "confidence": 0.9,
        "evidence_ids": list(EVIDENCE_IDS),
        "calculations": [
            {
                "calculation_id": "calc:model-output",
                "expression": expression,
                "reported_result": "999",
                "unit": None,
                "evidence_ids": list(EVIDENCE_IDS),
            }
        ],
        "abstention": {
            "abstained": False,
            "reason": "none",
            "missing_information": [],
        },
    }


class MATLMCalculationsTests(unittest.TestCase):
    def test_day_precision_recomputes_civil_age_and_curriculum_id(self) -> None:
        original = response(
            "calendar_age(2000-06-15,2032-06-14,precision=day)",
            answer="À cette date, le calcul donne 999 ans.",
        )
        untouched = copy.deepcopy(original)

        corrected = reexecute_matlm_calculations(original, capsule())

        calculation = corrected["calculations"][0]
        expected_id = "calc:" + hashlib.sha256(REQUEST_ID.encode("utf-8")).hexdigest()[:20]
        self.assertEqual(calculation["calculation_id"], expected_id)
        self.assertEqual(calculation["reported_result"], "31")
        self.assertEqual(calculation["unit"], "ans")
        self.assertEqual(calculation["evidence_ids"], EVIDENCE_IDS)
        self.assertEqual(corrected["answer"], "À cette date, le calcul donne 31 ans.")
        self.assertEqual(original, untouched)

    def test_day_precision_counts_age_on_the_birthday(self) -> None:
        corrected = reexecute_matlm_calculations(
            response("calendar_age(2000-06-15,2032-06-15,precision=day)"),
            capsule(),
        )
        self.assertEqual(corrected["calculations"][0]["reported_result"], "32")

    def test_year_precision_returns_interval_or_exact_result(self) -> None:
        interval = reexecute_matlm_calculations(response(), capsule())
        self.assertEqual(interval["calculations"][0]["reported_result"], "31..32")
        self.assertIn("le calcul donne entre 31 et 32 ans", interval["answer"])

        exact = reexecute_matlm_calculations(
            response("calendar_age(2000-01-01,2032,precision=year)"),
            capsule(),
        )
        self.assertEqual(exact["calculations"][0]["reported_result"], "32")

    def test_month_precision_supports_exact_and_birthday_interval(self) -> None:
        exact = reexecute_matlm_calculations(
            response("calendar_age(2000-06-15,2032-05,precision=month)"),
            capsule(),
        )
        self.assertEqual(exact["calculations"][0]["reported_result"], "31")

        interval = reexecute_matlm_calculations(
            response("calendar_age(2000-06-15,2032-06,precision=month)"),
            capsule(),
        )
        self.assertEqual(interval["calculations"][0]["reported_result"], "31..32")

    def test_valid_leap_dates_are_supported(self) -> None:
        corrected = reexecute_matlm_calculations(
            response("calendar_age(2000-02-29,2024-02-29,precision=day)"),
            capsule(),
        )
        self.assertEqual(corrected["calculations"][0]["reported_result"], "24")

    def test_unknown_or_non_canonical_expressions_are_never_executed(self) -> None:
        expressions = [
            "2032 - 2000",
            "__import__('os').system('whoami')",
            "calendar_age(2000-06-15,2032,precision=year);danger()",
            "calendar_age(2000-06-15, 2032,precision=year)",
            "Calendar_Age(2000-06-15,2032,precision=year)",
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(MATLMCalculationError, "format strict"):
                    reexecute_matlm_calculations(response(expression), capsule())

    def test_precision_must_match_the_iso_date_shape(self) -> None:
        expressions = [
            "calendar_age(2000-06-15,2032,precision=day)",
            "calendar_age(2000-06-15,2032-06,precision=year)",
            "calendar_age(2000-06-15,2032-06-15,precision=month)",
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(MATLMCalculationError, "correspond pas à precision"):
                    reexecute_matlm_calculations(response(expression), capsule())

    def test_invalid_iso_dates_and_event_before_birth_are_rejected(self) -> None:
        invalid = [
            "calendar_age(2001-02-29,2032,precision=year)",
            "calendar_age(2000-06-15,2032-13,precision=month)",
            "calendar_age(2000-06-15,2032-02-30,precision=day)",
            "calendar_age(0000-01-01,2032,precision=year)",
        ]
        for expression in invalid:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(MATLMCalculationError, "date ISO valide"):
                    reexecute_matlm_calculations(response(expression), capsule())

        with self.assertRaisesRegex(MATLMCalculationError, "précéder la naissance"):
            reexecute_matlm_calculations(
                response("calendar_age(2000-06-15,1999-12-31,precision=day)"),
                capsule(),
            )

    def test_multiple_calculations_receive_unique_ids_and_ordered_corrections(self) -> None:
        original = response(
            "calendar_age(2000-06-15,2032-06-14,precision=day)",
            answer="Le calcul donne 999 ans; le calcul donne 888 ans.",
        )
        original["calculations"].append(
            {
                "calculation_id": "calc:model-output-2",
                "expression": "calendar_age(2000-06-15,2032,precision=year)",
                "reported_result": "888",
                "unit": "années",
                "evidence_ids": ["ev:birth"],
            }
        )

        corrected = reexecute_matlm_calculations(original, capsule())

        ids = [item["calculation_id"] for item in corrected["calculations"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(corrected["calculations"][1]["evidence_ids"], ["ev:birth"])
        self.assertEqual(
            corrected["answer"],
            "Le calcul donne 31 ans; le calcul donne entre 31 et 32 ans.",
        )

    def test_answer_is_left_unchanged_when_formula_count_is_ambiguous(self) -> None:
        text = "Le calcul donne 999 ans puis une seconde valeur non structurée."
        original = response(answer=text)
        original["calculations"].append(
            {
                "calculation_id": "calc:second",
                "expression": "calendar_age(2000-06-15,2032-06-15,precision=day)",
                "reported_result": "888",
                "unit": "ans",
                "evidence_ids": ["ev:event"],
            }
        )
        corrected = reexecute_matlm_calculations(original, capsule())
        self.assertEqual(corrected["answer"], text)

    def test_answer_correction_cannot_exceed_capsule_bound(self) -> None:
        text = "le calcul donne 9 ans"
        bounded_capsule = capsule(max_answer_characters=len(text))
        with self.assertRaisesRegex(MATLMCalculationError, "borne autorisée"):
            reexecute_matlm_calculations(response(answer=text), bounded_capsule)

    def test_no_calculation_returns_an_independent_revalidated_copy(self) -> None:
        original = response(answer="Aucun calcul demandé.")
        original["calculations"] = []
        corrected = reexecute_matlm_calculations(original, capsule())
        self.assertEqual(corrected, original)
        self.assertIsNot(corrected, original)
        self.assertIsNot(corrected["evidence_ids"], original["evidence_ids"])

    def test_initial_contract_validation_is_not_bypassed(self) -> None:
        invalid = response()
        invalid["calculations"][0]["evidence_ids"] = ["ev:invented"]
        with self.assertRaises(ContractValidationError):
            reexecute_matlm_calculations(invalid, capsule())


if __name__ == "__main__":
    unittest.main()
