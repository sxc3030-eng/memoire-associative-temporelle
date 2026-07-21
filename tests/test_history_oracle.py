from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.history_oracle import (  # noqa: E402
    HISTORY_CALCULATION_VERSION,
    HistoryValidationError,
    build_ground_truth,
    calculation_catalog,
    civil_year_ordinal,
    derive_calculable_data,
    normalize_historical_event,
)


def event(event_id: str = "evt-1", **changes):
    value = {
        "id": event_id,
        "subject": "Alexandrie",
        "predicate": "population_estimee",
        "object": 1200,
        "valid_from": {"year": 1},
        "valid_to": {"year": 11},
        "recorded_order": 1,
        "source": {"id": "archive-1", "label": "Archive", "confidence": 0.8},
        "context": {"region": "mediterranee"},
        "measurements": [{"name": "distance", "value": 2.5, "unit": "km"}],
        "coordinates": {"latitude": 45.5, "longitude": -73.5},
        "birth_year": -20,
    }
    value.update(changes)
    return value


class HistoryOracleValidationTests(unittest.TestCase):
    def test_civil_calendar_has_no_year_zero_and_crossing_is_one_year(self) -> None:
        self.assertEqual(civil_year_ordinal(-1), 0)
        self.assertEqual(civil_year_ordinal(1), 1)
        crossing = event(valid_from={"year": -1}, valid_to={"year": 1})
        duration = next(
            item for item in derive_calculable_data(crossing)
            if item["kind"] == "duration_years"
        )
        self.assertEqual(duration["value"], 1)
        with self.assertRaises(HistoryValidationError):
            normalize_historical_event(event(valid_from={"year": 0}))

    def test_validation_rejects_non_finite_values_and_bad_coordinates(self) -> None:
        with self.assertRaises(HistoryValidationError):
            normalize_historical_event(
                event(measurements=[{"name": "distance", "value": float("nan"), "unit": "m"}])
            )
        with self.assertRaises(HistoryValidationError):
            normalize_historical_event(event(coordinates={"latitude": 91, "longitude": 0}))

    def test_normalization_is_strict_json_and_preserves_source(self) -> None:
        normalized = normalize_historical_event(event())
        encoded = json.dumps(normalized, ensure_ascii=False, allow_nan=False)
        self.assertIn("archive-1", encoded)
        self.assertEqual(normalized["schema_version"], "history-event-v1")
        self.assertEqual(normalized["measurements"][0]["value"], "2.5")

    def test_invalid_calendar_day_and_reversed_same_year_range_are_rejected(self) -> None:
        with self.assertRaises(HistoryValidationError):
            normalize_historical_event(
                event(valid_from={"year": 2025, "month": 2, "day": 31})
            )
        with self.assertRaises(HistoryValidationError):
            normalize_historical_event(
                event(
                    valid_from={"year": 2025, "month": 12, "day": 31},
                    valid_to={"year": 2025, "month": 1, "day": 1},
                )
            )

    def test_source_provenance_is_preserved_and_unknown_field_is_rejected(self) -> None:
        source = {
            "id": "archive-detaillee",
            "label": "Registre municipal",
            "confidence": 0.75,
            "url": "https://example.test/archive/42",
            "reference": "volume-7",
            "page": "42",
            "accessed_at": "2026-07-21",
            "statement_id": "statement-9",
            "revision_id": "revision-3",
            "license": "CC0",
            "metadata": {"collection": "recensements", "verified": True},
        }

        normalized = normalize_historical_event(event(source=source))

        self.assertEqual(normalized["source"], source)
        with self.assertRaises(HistoryValidationError):
            normalize_historical_event(
                event(source={"id": "archive-1", "champ_inconnu": "secret"})
            )

    def test_extreme_numeric_exponent_is_rejected(self) -> None:
        with self.assertRaises(HistoryValidationError):
            normalize_historical_event(
                event(
                    measurements=[
                        {"name": "distance", "value": "1e1001", "unit": "m"}
                    ]
                )
            )


class HistoryOracleCalculationTests(unittest.TestCase):
    def test_all_available_temporal_measure_and_age_calculations_are_lineaged(self) -> None:
        calculations = derive_calculable_data(event())
        kinds = {item["kind"] for item in calculations}
        self.assertTrue({
            "duration_years", "temporal_midpoint", "age_at_start", "normalized_measurement"
        }.issubset(kinds))
        normalized = next(item for item in calculations if item["kind"] == "normalized_measurement")
        self.assertEqual(Decimal(str(normalized["value"])), Decimal("2500"))
        self.assertEqual(normalized["unit"], "m")
        for calculation in calculations:
            self.assertEqual(calculation["status"], "computed_from_sources")
            self.assertEqual(calculation["calculation_version"], HISTORY_CALCULATION_VERSION)
            self.assertIn("evt-1", calculation["dependency_event_ids"])
            self.assertTrue(calculation["formula"])

    def test_previous_event_enables_change_rate_and_distance(self) -> None:
        previous = event(
            "evt-0",
            valid_from={"year": -10},
            valid_to={"year": -5},
            recorded_order=0,
            measurements=[{"name": "distance", "value": 1, "unit": "km"}],
            coordinates={"latitude": 44.0, "longitude": -72.0},
        )
        current = event(valid_from={"year": 1}, valid_to={"year": 2})
        calculations = derive_calculable_data(current, previous)
        kinds = {item["kind"] for item in calculations}
        self.assertTrue({
            "interval_from_previous", "absolute_change", "percent_change",
            "annual_rate", "distance_from_previous"
        }.issubset(kinds))
        change = next(item for item in calculations if item["kind"] == "absolute_change")
        self.assertEqual(Decimal(str(change["value"])), Decimal("1500"))
        self.assertEqual(change["dependency_event_ids"], ["evt-0", "evt-1"])
        distance = next(item for item in calculations if item["kind"] == "distance_from_previous")
        self.assertFalse(distance["exact"])
        self.assertGreater(float(distance["value"]), 0)

    def test_unknown_units_and_currency_are_not_guessed(self) -> None:
        calculations = derive_calculable_data(
            event(measurements=[{"name": "prix", "value": 10, "unit": "CAD"}])
        )
        self.assertNotIn("normalized_measurement", {item["kind"] for item in calculations})
        catalog = calculation_catalog()
        catalog_names = {item["name"] for item in catalog["calculations"]}
        self.assertTrue(
            {"duration_years", "duration_months", "duration_days"}.issubset(
                catalog_names
            )
        )
        self.assertEqual(
            catalog["units"]["currencies"],
            "not_converted_without_dated_sourced_rates",
        )
        self.assertFalse(catalog["free_form_formulas"])

    def test_day_duration_is_exact_across_leap_day(self) -> None:
        calculations = derive_calculable_data(
            event(
                valid_from={"year": 2024, "month": 2, "day": 28},
                valid_to={"year": 2024, "month": 3, "day": 1},
            )
        )

        duration = next(
            item for item in calculations if item["kind"] == "duration_days"
        )
        self.assertEqual(duration["value"], 2)
        self.assertEqual(duration["unit"], "day")
        self.assertTrue(duration["exact"])

    def test_partial_calendar_precision_is_not_presented_as_exact_elapsed_time(self) -> None:
        calculations = derive_calculable_data(
            event(valid_from={"year": 2020}, valid_to={"year": 2021})
        )
        duration = next(
            item for item in calculations if item["kind"] == "duration_years"
        )
        midpoint = next(
            item for item in calculations if item["kind"] == "temporal_midpoint"
        )
        self.assertFalse(duration["exact"])
        self.assertFalse(midpoint["exact"])

    def test_fahrenheit_conversion_is_not_exact_and_negative_kelvin_is_rejected(self) -> None:
        calculations = derive_calculable_data(
            event(
                measurements=[
                    {"name": "temperature", "value": 32, "unit": "F"}
                ]
            )
        )
        temperature = next(
            item
            for item in calculations
            if item["kind"] == "normalized_measurement"
        )
        self.assertEqual(Decimal(str(temperature["value"])), Decimal("273.15"))
        self.assertEqual(temperature["unit"], "K")
        self.assertFalse(temperature["exact"])

        with self.assertRaises(HistoryValidationError):
            derive_calculable_data(
                event(
                    measurements=[
                        {"name": "temperature", "value": "-0.01", "unit": "K"}
                    ]
                )
            )


class HistoryGroundTruthTests(unittest.TestCase):
    def test_contradictions_are_preserved_and_latest_state_is_indexed(self) -> None:
        left = event("left", object="active", valid_from={"year": 100}, valid_to={"year": 110}, recorded_order=1)
        right = event("right", object="archivee", valid_from={"year": 100}, valid_to={"year": 110}, recorded_order=2)
        later = event("later", object="reconstruite", valid_from={"year": 200}, valid_to={"year": 210}, recorded_order=3)
        truth = build_ground_truth([right, later, left])
        self.assertEqual(len(truth["contradictions"]), 1)
        self.assertEqual(set(truth["contradictions"][0]["event_ids"]), {"left", "right"})
        self.assertEqual(
            truth["latest_by_subject_predicate"]["alexandrie|population_estimee"],
            "later",
        )
        self.assertEqual(
            set(truth),
            {
                "schema_version", "events", "by_id", "calculations",
                "contradictions", "latest_by_subject_predicate",
            },
        )

    def test_duplicate_event_ids_are_refused(self) -> None:
        with self.assertRaises(HistoryValidationError):
            build_ground_truth([event("same"), event("same", recorded_order=2)])

    def test_homonyms_are_separated_by_context_entity_key(self) -> None:
        port = event(
            "port",
            object="port actif",
            context={"entity_key": "alexandrie-port", "region": "mediterranee"},
        )
        observatory = event(
            "observatory",
            object="observatoire actif",
            context={"entity_key": "alexandrie-observatoire", "region": "amerique"},
        )

        truth = build_ground_truth([port, observatory])

        self.assertEqual(truth["contradictions"], [])
        self.assertEqual(
            truth["latest_by_subject_predicate"][
                "alexandrie-port|population_estimee"
            ],
            "port",
        )
        self.assertEqual(
            truth["latest_by_subject_predicate"][
                "alexandrie-observatoire|population_estimee"
            ],
            "observatory",
        )

    def test_recorded_order_can_differ_from_chronology_without_negative_interval(self) -> None:
        earlier = event(
            "earlier",
            valid_from={"year": 100},
            valid_to={"year": 101},
            recorded_order=20,
            measurements=[{"name": "distance", "value": 1, "unit": "km"}],
        )
        later = event(
            "later",
            valid_from={"year": 200},
            valid_to={"year": 201},
            recorded_order=1,
            measurements=[{"name": "distance", "value": 2, "unit": "km"}],
        )

        truth = build_ground_truth([earlier, later])
        intervals = [
            item
            for item in truth["calculations"]
            if item["kind"] == "interval_from_previous"
        ]

        self.assertEqual([item["id"] for item in truth["events"]], ["later", "earlier"])
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["event_id"], "later")
        self.assertEqual(intervals[0]["value"], 100)
        self.assertTrue(all(Decimal(str(item["value"])) >= 0 for item in intervals))

    def test_retract_removes_target_from_effective_truth(self) -> None:
        obsolete = event("obsolete", recorded_order=1)
        retraction = event(
            "retraction",
            subject="Archives",
            predicate="decision",
            object="annule obsolete",
            valid_from={"year": 20},
            valid_to={"year": 21},
            recorded_order=2,
            context={"entity_key": "archives"},
            retracts=["obsolete"],
        )

        truth = build_ground_truth([obsolete, retraction])

        self.assertIn("obsolete", truth["by_id"])
        self.assertNotIn(
            "alexandrie|population_estimee",
            truth["latest_by_subject_predicate"],
        )
        self.assertFalse(
            any(item["event_id"] == "obsolete" for item in truth["calculations"])
        )

    def test_supersede_resolves_same_instant_without_silent_contradiction(self) -> None:
        original = event(
            "original",
            object="ancienne valeur",
            recorded_order=1,
            context={"entity_key": "alexandrie-port"},
        )
        replacement = event(
            "replacement",
            object="valeur corrigee",
            recorded_order=2,
            context={"entity_key": "alexandrie-port"},
            supersedes=["original"],
        )

        truth = build_ground_truth([replacement, original])

        self.assertEqual(truth["contradictions"], [])
        self.assertEqual(
            truth["latest_by_subject_predicate"][
                "alexandrie-port|population_estimee"
            ],
            "replacement",
        )

    def test_correction_cycle_is_rejected(self) -> None:
        first = event(
            "first",
            context={"entity_key": "alexandrie-port"},
            supersedes=["second"],
        )
        second = event(
            "second",
            recorded_order=2,
            context={"entity_key": "alexandrie-port"},
            supersedes=["first"],
        )

        with self.assertRaises(HistoryValidationError):
            build_ground_truth([first, second])


if __name__ == "__main__":
    unittest.main()
