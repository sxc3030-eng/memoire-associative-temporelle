from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.math_engine import (  # noqa: E402
    REGISTRY_VERSION,
    MathEngine,
    MathEngineError,
    MathLimits,
    catalog,
    catalog_entries,
    evaluate,
)
from memory_agent.memory import MemoryEngine  # noqa: E402


class MathEngineEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = MathEngine()

    def test_public_api_respects_precedence_and_returns_strict_json(self) -> None:
        result = self.engine.evaluate("2 + 3 * 4")

        self.assertEqual(result["result"], 14)
        self.assertEqual(result["display"], "14")
        self.assertEqual(result["type"], "integer")
        self.assertTrue(result["exact"])
        self.assertEqual(result["operations"], ["*", "+"])
        self.assertEqual(result["operations_count"], 2)
        self.assertGreaterEqual(result["duration_ms"], 0)
        self.assertEqual(result["algorithm_version"], REGISTRY_VERSION)
        self.assertEqual(result["verification"]["status"], "policy_validated")
        self.assertFalse(result["verification"]["independent_oracle"])
        self.assertEqual(result["verification"]["memory_writes"], 0)
        json.dumps(result, ensure_ascii=False, allow_nan=False)

        self.assertEqual(evaluate("6 * 7")["result"], 42)
        self.assertEqual(catalog()["version"], REGISTRY_VERSION)
        self.assertEqual(len(catalog_entries()), catalog()["count"])

    def test_every_operator_and_unary_operator_is_supported(self) -> None:
        cases = {
            "20 + 5": 25,
            "20 - 5": 15,
            "20 * 5": 100,
            "20 / 5": 4.0,
            "22 // 5": 4,
            "22 % 5": 2,
            "2 ** 10": 1024,
            "-5 + +2": -3,
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(self.engine.evaluate(expression)["result"], expected)

        division = self.engine.evaluate("1 / 2")
        self.assertEqual(division["type"], "float")
        self.assertFalse(division["exact"])

    def test_constants_roots_logs_trigonometry_and_integer_algorithms(self) -> None:
        self.assertEqual(self.engine.evaluate("sqrt(81)")["result"], 9.0)
        self.assertEqual(self.engine.evaluate("isqrt(82)")["result"], 9)
        self.assertEqual(self.engine.evaluate("log2(2 ** 12)")["result"], 12.0)
        self.assertTrue(
            math.isclose(
                self.engine.evaluate("sin(pi / 2)")["result"],
                1.0,
                rel_tol=1e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                self.engine.evaluate("tau / (2 * pi)")["result"],
                1.0,
                rel_tol=1e-15,
            )
        )
        self.assertEqual(self.engine.evaluate("gcd(84, 126)")["result"], 42)
        self.assertEqual(self.engine.evaluate("lcm(12, 18)")["result"], 36)
        self.assertEqual(self.engine.evaluate("factorial(10)")["result"], 3_628_800)
        self.assertEqual(self.engine.evaluate("comb(10, 3)")["result"], 120)
        self.assertEqual(self.engine.evaluate("perm(10, 3)")["result"], 720)
        self.assertEqual(self.engine.evaluate("hypot(3, 4)")["result"], 5.0)

    def test_fraction_is_exact_reduced_and_serializable(self) -> None:
        result = self.engine.evaluate("frac(1, 3) + frac(1, 6)")

        self.assertEqual(result["result"], {"numerator": 1, "denominator": 2})
        self.assertEqual(result["display"], "1/2")
        self.assertEqual(result["type"], "fraction")
        self.assertTrue(result["exact"])
        self.assertEqual(result["approximation"], 0.5)
        self.assertEqual(result["functions_used"], ["frac"])
        json.dumps(result, allow_nan=False)

        rational_mean = self.engine.evaluate(
            "mean([frac(1, 3), frac(2, 3)])"
        )
        self.assertEqual(
            rational_mean["result"], {"numerator": 1, "denominator": 2}
        )
        self.assertTrue(rational_mean["exact"])

    def test_large_exact_integers_survive_javascript_json_limits(self) -> None:
        integer = self.engine.evaluate("2 ** 100")
        fraction = self.engine.evaluate("frac(2 ** 100, 3)")

        self.assertEqual(integer["result"], str(2**100))
        self.assertEqual(integer["display"], str(2**100))
        self.assertIsNone(integer["approximation"])
        self.assertEqual(fraction["result"]["numerator"], str(2**100))
        self.assertEqual(fraction["result"]["denominator"], 3)
        encoded = json.dumps(
            {"integer": integer, "fraction": fraction},
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
        self.assertEqual(decoded["integer"]["result"], str(2**100))

    def test_statistics_accept_only_bounded_numeric_lists_or_tuples(self) -> None:
        self.assertEqual(self.engine.evaluate("mean([2, 4, 6, 8])")["result"], 5)
        self.assertEqual(self.engine.evaluate("median((1, 9, 3))")["result"], 3)
        self.assertEqual(self.engine.evaluate("pvariance([2, 4, 6])")["result"], 8 / 3)
        self.assertTrue(
            math.isclose(
                self.engine.evaluate("pstdev([2, 4, 6])")["result"],
                math.sqrt(8 / 3),
                rel_tol=1e-15,
            )
        )

        with self.assertRaises(MathEngineError) as context:
            self.engine.evaluate("mean(1, 2)")
        self.assertEqual(context.exception.code, "invalid_arguments")

    def test_domain_and_zero_division_errors_are_public_and_bounded(self) -> None:
        cases = {
            "1 / 0": "division_by_zero",
            "frac(1, 0)": "division_by_zero",
            "sqrt(-1)": "domain_error",
            "log(0)": "domain_error",
            "factorial(-1)": "domain_error",
            "variance([1])": "domain_error",
        }
        for expression, code in cases.items():
            with self.subTest(expression=expression):
                with self.assertRaises(MathEngineError) as context:
                    self.engine.evaluate(expression)
                self.assertEqual(context.exception.code, code)
                self.assertEqual(str(context.exception), context.exception.message)
                self.assertLessEqual(len(context.exception.message), 500)

    def test_calculation_and_catalog_do_not_write_to_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            memory = MemoryEngine(Path(temporary_directory) / "memory.sqlite3")
            try:
                before = memory.stats()
                for expression in (
                    "2 + 2",
                    "sqrt(144)",
                    "mean([1, 2, 3])",
                    "frac(3, 7)",
                ):
                    self.engine.evaluate(expression)
                self.engine.catalog()
                after = memory.stats()
                self.assertEqual(after["events"], before["events"])
                self.assertEqual(after["episodes"], before["episodes"])
                self.assertEqual(after["concepts"], before["concepts"])
            finally:
                memory.close()


class MathEngineSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = MathEngine()

    def test_arbitrary_python_constructs_are_rejected_without_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            marker = Path(temporary_directory) / "should-not-exist.txt"
            probes = [
                "__import__('os').getcwd()",
                "open('should-not-exist.txt', 'w')",
                "(1).__class__",
                "[10, 20][0]",
                "lambda x: x",
                "(lambda: 1)()",
                "[x for x in [1, 2, 3]]",
                "{x for x in [1, 2, 3]}",
                "(x for x in [1, 2, 3])",
                "{'x': 1}",
                "1 if True else 0",
                "sqrt(x=4)",
                "globals()",
                "import os",
                "'texte'",
            ]
            for expression in probes:
                with self.subTest(expression=expression):
                    with self.assertRaises(MathEngineError):
                        self.engine.evaluate(expression)
            self.assertFalse(marker.exists())

    def test_unknown_names_and_non_finite_literals_are_rejected(self) -> None:
        for expression in ("secret", "unknown(2)", "True", "1e999"):
            with self.subTest(expression=expression):
                with self.assertRaises(MathEngineError):
                    self.engine.evaluate(expression)


class MathEngineLimitTests(unittest.TestCase):
    def assert_limit(self, engine: MathEngine, expression: str) -> str:
        with self.assertRaises(MathEngineError) as context:
            engine.evaluate(expression)
        self.assertIn(context.exception.code, {"resource_limit", "numeric_limit"})
        return context.exception.code

    def test_expression_nodes_depth_and_collection_limits(self) -> None:
        defaults = MathLimits()
        self.assertEqual(
            self.assert_limit(
                MathEngine(replace(defaults, max_expression_chars=8)),
                "1 + 2 + 3 + 4",
            ),
            "resource_limit",
        )
        self.assertEqual(
            self.assert_limit(
                MathEngine(replace(defaults, max_nodes=7)), "1 + 2 + 3"
            ),
            "resource_limit",
        )
        self.assertEqual(
            self.assert_limit(
                MathEngine(replace(defaults, max_depth=4)),
                "1 + (2 + (3 + 4))",
            ),
            "resource_limit",
        )
        self.assertEqual(
            self.assert_limit(
                MathEngine(replace(defaults, max_collection_items=2)),
                "mean([1, 2, 3])",
            ),
            "resource_limit",
        )

    def test_exponent_factorial_combinatorial_bits_and_digits_are_bounded(self) -> None:
        defaults = MathLimits()
        self.assertEqual(
            self.assert_limit(
                MathEngine(replace(defaults, max_exponent=5)), "2 ** 6"
            ),
            "resource_limit",
        )
        self.assertEqual(
            self.assert_limit(MathEngine(), "factorial(501)"),
            "resource_limit",
        )
        self.assertEqual(
            self.assert_limit(MathEngine(), "comb(1001, 2)"),
            "resource_limit",
        )
        self.assertEqual(
            self.assert_limit(
                MathEngine(replace(defaults, max_integer_bits=32)), "2 ** 40"
            ),
            "numeric_limit",
        )
        self.assertEqual(
            self.assert_limit(
                MathEngine(replace(defaults, max_output_digits=3)), "1000"
            ),
            "numeric_limit",
        )

    def test_negative_fraction_power_is_estimated_before_large_denominator(self) -> None:
        engine = MathEngine(replace(MathLimits(), max_integer_bits=128))

        code = self.assert_limit(engine, "frac(2 ** 64, 1) ** -4")

        self.assertEqual(code, "numeric_limit")


class MathEngineCatalogTests(unittest.TestCase):
    def test_catalog_is_versioned_categorized_operational_and_unique(self) -> None:
        engine = MathEngine()
        payload = engine.catalog()
        entries = engine.catalog_entries()

        self.assertEqual(payload["version"], REGISTRY_VERSION)
        self.assertEqual(payload["count"], len(entries))
        self.assertGreaterEqual(payload["count"], 35)
        self.assertEqual(len({entry["name"] for entry in entries}), len(entries))
        category_ids = {category["id"] for category in payload["categories"]}
        self.assertEqual({entry["category"] for entry in entries}, category_ids)
        self.assertEqual(
            payload["learning_levels"],
            [
                {"level": 1, "state": "received"},
                {"level": 2, "state": "observed"},
                {"level": 3, "state": "consolidated"},
                {"level": 4, "state": "operational"},
            ],
        )
        for entry in entries:
            self.assertEqual(entry["learning_level"], 4)
            self.assertEqual(entry["maturity"], "operational")
            self.assertIn(entry["exactness"], {"true", "false", "depends", "preserved"})
            self.assertIn(entry["exact"], {True, False, None})
        json.dumps(payload, ensure_ascii=False, allow_nan=False)

    def test_example_catalog_matches_runtime_registry_and_examples(self) -> None:
        document = json.loads(
            (PROJECT_ROOT / "examples" / "mathematiques-core.json").read_text(
                encoding="utf-8"
            )
        )["catalogue"]
        engine = MathEngine()
        runtime = engine.catalog()
        documented_functions = {
            name
            for category in document["categories"].values()
            for name in category["fonctions"]
        }

        self.assertEqual(document["version"], runtime["version"])
        self.assertEqual(document["learning_level"], 4)
        self.assertEqual(document["maturity"], "operational")
        self.assertEqual(document["learning_levels"], runtime["learning_levels"])
        self.assertEqual(documented_functions, {entry["name"] for entry in runtime["functions"]})
        self.assertEqual(set(document["categories"]), {item["id"] for item in runtime["categories"]})
        for example in document["exemples"]:
            with self.subTest(expression=example["expression"]):
                calculated = engine.evaluate(example["expression"])
                self.assertEqual(calculated["display"], example["affichage"])
                self.assertEqual(calculated["type"], example["type"])
                self.assertEqual(calculated["exact"], example["exact"])


if __name__ == "__main__":
    unittest.main()
