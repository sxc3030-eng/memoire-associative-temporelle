from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.history_stress_lab import (  # noqa: E402
    HistoryStressConfig,
    generate_history_scenario,
    history_stress_catalog,
    run_history_stress,
)


class HistoryStressConfigTests(unittest.TestCase):
    def test_catalog_and_default_config_are_strict_json(self) -> None:
        catalog = history_stress_catalog()

        self.assertEqual(catalog["schema_version"], "history-stress-v1")
        self.assertEqual(catalog["calculation_count"], len(catalog["calculations"]))
        self.assertGreater(catalog["calculation_count"], 0)
        self.assertFalse(catalog["provenance_policy"]["free_form_formula_execution"])
        self.assertEqual(
            catalog["provenance_policy"]["oracle_derivations"], "inferred"
        )
        json.dumps(catalog, ensure_ascii=False, allow_nan=False)
        json.dumps(
            generate_history_scenario(
                HistoryStressConfig(
                    event_count=4,
                    duplicate_rate=0,
                    contradiction_rate=0,
                    out_of_order_rate=0,
                    max_queries=4,
                )
            ),
            ensure_ascii=False,
            allow_nan=False,
        )

    def test_config_rejects_unbounded_or_ambiguous_values(self) -> None:
        defaults = HistoryStressConfig()
        invalid = (
            ("event_count", 1),
            ("event_count", 10_001),
            ("seed", -1),
            ("episode_size", 0),
            ("episode_size", 65),
            ("duplicate_rate", -0.01),
            ("duplicate_rate", 0.76),
            ("contradiction_rate", float("nan")),
            ("out_of_order_rate", float("inf")),
            ("max_queries", 0),
            ("max_queries", 5_001),
        )
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    replace(defaults, **{field: value})
        with self.assertRaises(TypeError):
            replace(defaults, event_count=True)


class HistoryScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HistoryStressConfig(
            event_count=12,
            seed=42,
            episode_size=2,
            duplicate_rate=0.25,
            contradiction_rate=0.25,
            out_of_order_rate=0.5,
            max_queries=24,
        )

    def test_scenario_is_deterministic_temporal_and_provenanced(self) -> None:
        first = generate_history_scenario(self.config)
        second = generate_history_scenario(self.config)

        self.assertEqual(first, second)
        self.assertEqual(first["coverage"]["earliest_year"], -2400)
        self.assertEqual(first["coverage"]["latest_year"], 2026)
        self.assertFalse(first["coverage"]["contains_year_zero"])
        self.assertGreater(first["coverage"]["source_ingestion_inversions"], 0)
        self.assertEqual(first["coverage"]["duplicate_submissions"], 3)
        self.assertEqual(first["coverage"]["contradictory_events"], 3)
        self.assertEqual(len({event["id"] for event in first["events"]}), 15)

        for event in first["events"]:
            self.assertNotEqual(event["valid_from"]["year"], 0)
            if event.get("valid_to"):
                self.assertNotEqual(event["valid_to"]["year"], 0)
            if event.get("birth_year") is not None:
                self.assertNotEqual(event["birth_year"], 0)
        self.assertTrue(
            all(record["source"]["type"] == "observed" for record in first["source_records"])
        )
        self.assertTrue(
            all(record["source"]["type"] == "inferred" for record in first["derived_records"])
        )
        self.assertLessEqual(
            {
                "schema_version",
                "events",
                "by_id",
                "calculations",
                "contradictions",
                "latest_by_subject_predicate",
            },
            set(first["ground_truth"]),
        )
        self.assertGreater(len(first["calculations"]), 0)
        semantic_queries = [
            query
            for query in first["queries"]
            if query["score_group"] == "semantic"
        ]
        plumbing_queries = [
            query
            for query in first["queries"]
            if query["score_group"] == "plumbing"
        ]
        self.assertLessEqual(
            {"date", "latest", "context", "contradiction"},
            {query["kind"] for query in semantic_queries},
        )
        self.assertLessEqual(
            {"marker", "derived"},
            {query["kind"] for query in plumbing_queries},
        )
        self.assertTrue(
            all(
                "historymarker" not in query["query"]
                and "derivedmarker" not in query["query"]
                for query in semantic_queries
            )
        )

    def test_contradictory_pair_keeps_two_independent_episodes(self) -> None:
        scenario = generate_history_scenario(self.config)
        records = {
            record["event"]["id"]: record for record in scenario["source_records"]
        }

        for pair in scenario["contradiction_pairs"]:
            self.assertNotEqual(pair["left_id"], pair["right_id"])
            self.assertNotEqual(
                records[pair["left_id"]]["episode_id"],
                records[pair["right_id"]]["episode_id"],
            )
        contradiction_queries = [
            query
            for query in scenario["queries"]
            if query["kind"] == "contradiction"
        ]
        self.assertEqual(len(contradiction_queries), len(scenario["contradiction_pairs"]))
        self.assertTrue(
            all(
                query["score_group"] == "semantic"
                and query["match_mode"] == "all_episodes"
                and len(set(query["expected_episode_ids"])) == 2
                and not query["expected_markers"]
                for query in contradiction_queries
            )
        )


class HistoryStressExecutionTests(unittest.TestCase):
    def test_small_run_is_isolated_deduplicated_and_queryable(self) -> None:
        report = run_history_stress(
            HistoryStressConfig(
                event_count=8,
                seed=7,
                episode_size=2,
                duplicate_rate=0.25,
                contradiction_rate=0.25,
                out_of_order_rate=0.5,
                max_queries=16,
            )
        )

        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["pipeline"]["drained"])
        self.assertEqual(report["pipeline"]["failed"], 0)
        self.assertTrue(report["pipeline"]["completed_count_exact"])
        self.assertTrue(report["pipeline"]["scoring_gate_passed"])
        self.assertTrue(report["pipeline"]["deduplication_exact"])
        self.assertTrue(report["pipeline"]["memory_event_count_exact"])
        self.assertTrue(report["pipeline"]["reader_writer_separated"])
        self.assertEqual(
            report["pipeline"]["source_counts"]["observed"],
            report["scenario"]["normalized_source_events"],
        )
        if report["scenario"]["derived_calculations"]:
            self.assertEqual(
                report["pipeline"]["source_counts"]["inferred"],
                report["scenario"]["derived_calculations"],
            )
        self.assertGreater(report["retrieval"]["queries"], 0)
        self.assertEqual(report["retrieval"]["status"], "scored")
        self.assertEqual(
            report["retrieval"]["top1_percent"],
            report["retrieval"]["semantic"]["top1_percent"],
        )
        self.assertEqual(
            report["retrieval"]["top5_percent"],
            report["retrieval"]["semantic"]["top5_percent"],
        )
        self.assertGreater(report["retrieval"]["plumbing_diagnostics"]["queries"], 0)
        self.assertEqual(
            report["retrieval"]["by_kind"]["marker"]["top5_percent"], 100.0
        )
        self.assertLessEqual(
            {"date", "latest", "context", "contradiction"},
            set(report["retrieval"]["by_kind"]),
        )
        self.assertGreaterEqual(report["retrieval"]["semantic"]["top5_percent"], 0.0)
        self.assertLessEqual(report["retrieval"]["semantic"]["top5_percent"], 100.0)
        self.assertEqual(report["provenance"]["derived_results_auto_promoted"], 0)
        self.assertFalse(report["temporal_semantics"]["event_time_equals_ingestion_time"])
        self.assertFalse(report["temporal_semantics"]["native_bitemporal_ranking"])
        self.assertTrue(report["isolation"]["temporary_storage_removed_after_run"])
        self.assertEqual(report["isolation"]["writes_outside_temporary_directory"], 0)
        json.dumps(report, ensure_ascii=False, allow_nan=False)

    def test_incomplete_pipeline_is_never_scored(self) -> None:
        config = HistoryStressConfig(
            event_count=4,
            seed=9,
            episode_size=1,
            duplicate_rate=0,
            contradiction_rate=0.25,
            out_of_order_rate=0.5,
            max_queries=8,
        )
        with patch(
            "memory_agent.history_stress_lab.MemoryPipeline.wait_until_idle",
            return_value=False,
        ):
            report = run_history_stress(config)

        self.assertEqual(report["status"], "incomplete")
        self.assertIn("pipeline_not_drained", report["incomplete_reasons"])
        self.assertFalse(report["pipeline"]["scoring_gate_passed"])
        self.assertEqual(report["retrieval"]["status"], "not_scored")
        self.assertEqual(report["retrieval"]["total_queries_executed"], 0)
        self.assertIsNone(report["retrieval"]["top1_percent"])
        self.assertIsNone(report["retrieval"]["top5_percent"])
        self.assertEqual(report["performance"]["recall_latency_ms"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
