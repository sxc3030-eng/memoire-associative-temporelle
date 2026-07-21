from __future__ import annotations

from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import memory_agent.matlm_inference as inference_module
from memory_agent.matlm_heldout_benchmark import (
    BenchmarkArm,
    MATLMHeldoutBenchmarkError,
    REPORT_SCHEMA_VERSION,
    benchmark_plan,
    build_benchmark_arms,
    load_balanced_heldout,
    normalize_exact_answer,
    run_heldout_benchmark,
    write_atomic_report,
)
from memory_agent.matlm_inference import InferenceConfig, validate_inference_config
from memory_agent.memory_native_curriculum import (
    SYNTHETIC_TASK_ORDER,
    build_synthetic_memory_curriculum,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_matlm_heldout.py"


def _write_dataset(root: Path, *, count: int = 27) -> tuple[Path, list[dict]]:
    rows = build_synthetic_memory_curriculum(seed=987_654_321, count=count)["examples"]
    path = root / "heldout.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path, rows


def _local_assets(root: Path) -> tuple[Path, Path]:
    base = root / "granite-base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            {"model_type": "granite", "architectures": ["GraniteForCausalLM"]}
        ),
        encoding="utf-8",
    )
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM"}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"test-only")
    return base, adapter


def _arm(name: str) -> BenchmarkArm:
    return BenchmarkArm(
        name=name,
        inference=InferenceConfig(
            adapter_path=None if name == "base" else Path("test-adapter"),
            base_model="ibm-granite/granite-3.3-2b-instruct",
            load_mode="bf16",
        ),
    )


class _Tracker:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0
        self.opens: dict[str, int] = {"base": 0, "adapter": 0}
        self.calls: dict[str, list[str]] = {"base": [], "adapter": []}


class _FakeSession(AbstractContextManager):
    effective_mode = "fake-bf16"

    def __init__(self, arm: BenchmarkArm, tracker: _Tracker) -> None:
        self.arm = arm
        self.tracker = tracker

    def __enter__(self):
        self.tracker.opens[self.arm.name] += 1
        self.tracker.active += 1
        self.tracker.maximum_active = max(
            self.tracker.maximum_active, self.tracker.active
        )
        return self

    def __exit__(self, error_type, error, traceback):
        self.tracker.active -= 1
        return False

    def ask(self, capsule):
        request_id = capsule["request_id"]
        self.tracker.calls[self.arm.name].append(request_id)
        target = _TARGETS[request_id]
        answer = json.loads(json.dumps(target, ensure_ascii=False))
        if self.arm.name == "base":
            # La métrique normalisée doit accepter casse et espaces tout en
            # laissant full_target_exact à faux (confiance différente).
            answer["answer"] = "  ".join(answer["answer"].upper().split(" "))
            answer["confidence"] = max(0.0, answer["confidence"] - 0.01)
        return answer


_TARGETS: dict[str, dict] = {}


class MATLMHeldoutBenchmarkTests(unittest.TestCase):
    def test_balanced_selection_covers_nine_tasks_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, _ = _write_dataset(Path(directory), count=27)
            first = load_balanced_heldout(dataset, limit=10)
            second = load_balanced_heldout(dataset, limit=10)

        self.assertEqual(tuple(first.task_counts), SYNTHETIC_TASK_ORDER)
        self.assertEqual(first.task_counts[SYNTHETIC_TASK_ORDER[0]], 2)
        self.assertTrue(
            all(first.task_counts[task] >= 1 for task in SYNTHETIC_TASK_ORDER)
        )
        self.assertLessEqual(max(first.task_counts.values()) - min(first.task_counts.values()), 1)
        self.assertEqual(first.selection_sha256, second.selection_sha256)
        self.assertEqual(
            [case.example_id for case in first.cases],
            [case.example_id for case in second.cases],
        )
        with self.assertRaisesRegex(MATLMHeldoutBenchmarkError, "compris entre 9"):
            load_balanced_heldout(dataset, limit=8)

    def test_each_arm_opens_once_uses_same_cases_and_scores_by_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, rows = _write_dataset(Path(directory), count=27)
            selection = load_balanced_heldout(dataset, limit=18)
        _TARGETS.clear()
        _TARGETS.update({row["example_id"]: row["target"] for row in rows})
        tracker = _Tracker()

        report = run_heldout_benchmark(
            selection,
            (_arm("base"), _arm("adapter")),
            session_factory=lambda arm: _FakeSession(arm, tracker),
        )

        expected_ids = [case.example_id for case in selection.cases]
        self.assertEqual(report["schema_version"], REPORT_SCHEMA_VERSION)
        self.assertEqual(tracker.opens, {"base": 1, "adapter": 1})
        self.assertEqual(tracker.maximum_active, 1)
        self.assertEqual(tracker.calls["base"], expected_ids)
        self.assertEqual(tracker.calls["adapter"], expected_ids)
        base, adapter = report["arms"]
        self.assertEqual(base["global_metrics"]["attempted"], 18)
        self.assertEqual(base["global_metrics"]["rates"]["answer_exact_normalized"], 1.0)
        self.assertEqual(base["global_metrics"]["rates"]["all_required_exact"], 1.0)
        self.assertEqual(base["global_metrics"]["rates"]["full_target_exact"], 0.0)
        self.assertEqual(adapter["global_metrics"]["rates"]["full_target_exact"], 1.0)
        self.assertTrue(
            all(
                task_metrics["attempted"] == 2
                for task_metrics in adapter["metrics_by_task_type"].values()
            )
        )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn(rows[0]["target"]["answer"], serialized)
        self.assertNotIn("raw_output", serialized)

    def test_errors_are_bounded_and_do_not_abort_remaining_cases(self) -> None:
        class Broken(_FakeSession):
            def ask(self, capsule):
                self.tracker.calls[self.arm.name].append(capsule["request_id"])
                raise RuntimeError("sortie-secrète-" + "X" * 20_000)

        with tempfile.TemporaryDirectory() as directory:
            dataset, _ = _write_dataset(Path(directory), count=9)
            selection = load_balanced_heldout(dataset, limit=9)
        tracker = _Tracker()
        report = run_heldout_benchmark(
            selection,
            (_arm("base"),),
            session_factory=lambda arm: Broken(arm, tracker),
        )

        arm = report["arms"][0]
        self.assertEqual(arm["global_metrics"]["model_error"], 9)
        self.assertEqual(len(tracker.calls["base"]), 9)
        self.assertTrue(
            all(len(case["error"]["message"]) <= 400 for case in arm["cases"])
        )
        self.assertTrue(all(case["prediction_answer_sha256"] is None for case in arm["cases"]))

    def test_atomic_report_refuses_implicit_overwrite(self) -> None:
        report = {"schema_version": REPORT_SCHEMA_VERSION, "value": 1}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            written = write_atomic_report(output, report)
            self.assertEqual(json.loads(written.read_text(encoding="utf-8")), report)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            with self.assertRaisesRegex(MATLMHeldoutBenchmarkError, "existe déjà"):
                write_atomic_report(output, report)
            report["value"] = 2
            write_atomic_report(output, report, replace=True)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["value"], 2)

    def test_base_config_is_explicit_and_runtime_skips_peft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, adapter = _local_assets(Path(directory))
            base_config = validate_inference_config(
                InferenceConfig(adapter_path=None, base_model=str(base), load_mode="bf16")
            )
            arms = build_benchmark_arms(
                compare="both",
                base_model=str(base),
                adapter_path=adapter,
                load_mode="bf16",
            )
            self.assertIsNone(base_config.adapter_path)
            self.assertEqual([arm.name for arm in arms], ["base", "adapter"])

            peft = Mock()
            fake_model = SimpleNamespace(config=SimpleNamespace(use_cache=False))
            fake_model.eval = Mock()
            fake_xpu = SimpleNamespace(manual_seed_all=Mock(), empty_cache=Mock())
            fake_torch = SimpleNamespace(
                manual_seed=Mock(),
                xpu=fake_xpu,
            )
            stack = {"torch": fake_torch, "PeftModel": peft}
            with (
                patch(
                    "memory_agent.matlm_inference._runtime_imports",
                    return_value=stack,
                ) as imports,
                patch("memory_agent.matlm_inference._xpu_device", return_value="xpu:0"),
                patch("memory_agent.matlm_inference._load_tokenizer", return_value=object()),
                patch(
                    "memory_agent.matlm_inference._load_base_model",
                    return_value=(fake_model, "bf16", None),
                ),
            ):
                assets = inference_module._load_runtime_assets(base_config)

        imports.assert_called_once_with(require_peft=False)
        peft.from_pretrained.assert_not_called()
        self.assertIs(assets.model, fake_model)
        self.assertTrue(fake_model.config.use_cache)

    def test_report_and_plan_hide_absolute_local_asset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, rows = _write_dataset(root, count=9)
            base, adapter = _local_assets(root)
            selection = load_balanced_heldout(dataset, limit=9)
            arms = build_benchmark_arms(
                compare="both",
                base_model=str(base),
                adapter_path=adapter,
                load_mode="bf16",
            )
            _TARGETS.clear()
            _TARGETS.update({row["example_id"]: row["target"] for row in rows})
            tracker = _Tracker()
            report = run_heldout_benchmark(
                selection,
                arms,
                session_factory=lambda arm: _FakeSession(arm, tracker),
            )
            plan = benchmark_plan(selection, arms)

            serialized = json.dumps({"report": report, "plan": plan})
            self.assertNotIn(str(root), serialized)
            self.assertEqual(
                report["dataset"]["path"], f"{root.name}/heldout.jsonl"
            )
            self.assertEqual(
                report["arms"][0]["model"]["base_model"],
                f"{root.name}/granite-base",
            )
            self.assertEqual(
                report["arms"][1]["model"]["adapter"],
                f"{root.name}/adapter",
            )

    def test_cli_dry_run_never_imports_ml_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = _write_dataset(root, count=9)
            base, _ = _local_assets(root)
            blocker = root / "sitecustomize.py"
            blocker.write_text(
                "import importlib.abc, sys\n"
                "class Block(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname.split('.')[0] in {'torch','transformers','peft'}:\n"
                "            raise AssertionError('ML import forbidden: ' + fullname)\n"
                "        return None\n"
                "sys.meta_path.insert(0, Block())\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dataset",
                    str(dataset),
                    "--base-model",
                    str(base),
                    "--compare",
                    "base",
                    "--limit",
                    "9",
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["status"], "dry-run")
        self.assertEqual(plan["selected_count"], 9)
        self.assertEqual(plan["arms"][0]["name"], "base")
        self.assertEqual(result.stderr, "")

    def test_normalization_preserves_punctuation(self) -> None:
        self.assertEqual(normalize_exact_answer("  ÉTAT\nFICTIF  "), "état fictif")
        self.assertNotEqual(normalize_exact_answer("état fictif."), "état fictif")


if __name__ == "__main__":
    unittest.main()
