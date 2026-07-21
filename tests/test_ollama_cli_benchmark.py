from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.matlm_heldout_benchmark import load_balanced_heldout
from memory_agent.memory_native_curriculum import build_synthetic_memory_curriculum
from memory_agent.ollama_cli_benchmark import (
    _first_json_mapping,
    _validate_raw_generated_answer,
    CommandResult,
    OllamaCLIBenchmarkError,
    OllamaCLIConfig,
    OllamaCLITimeoutError,
    REPORT_SCHEMA_VERSION,
    answer_anchor_metrics,
    benchmark_plan,
    render_ollama_prompt,
    run_ollama_cli_benchmark,
    validate_ollama_cli_config,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_ollama_heldout.py"


def _write_dataset(root: Path, *, count: int = 9) -> Path:
    rows = build_synthetic_memory_curriculum(seed=515_151, count=count)["examples"]
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
    return path


class _FakeRunner:
    def __init__(self, targets, *, invalidate_first: bool = False) -> None:
        self.targets = [json.loads(json.dumps(target)) for target in targets]
        self.invalidate_first = invalidate_first
        self.run_index = 0
        self.commands: list[tuple[tuple[str, ...], str | None, float]] = []

    def run(
        self,
        command,
        *,
        input_text,
        timeout_seconds,
        stdout_limit,
        stderr_limit,
    ):
        del stdout_limit, stderr_limit
        clean = tuple(command)
        self.commands.append((clean, input_text, timeout_seconds))
        if clean[1:] == ("--version",):
            return CommandResult(0, "ollama version 9.9-test\n", 1.0)
        if clean[1] == "show":
            return CommandResult(0, "test-only local model metadata\n", 1.0)
        if clean[1:] == ("list",):
            return CommandResult(
                0,
                "NAME ID SIZE MODIFIED\n"
                "qwen2.5:14b-instruct-q4_0 5449194ff803 8.5 GB 2 months ago\n"
                "qwen2.5:14b 0123456789ab 8.5 GB 2 months ago\n",
                1.0,
            )
        if clean[1] == "stop":
            return CommandResult(0, "", 1.0)
        if clean[1] != "run":
            raise AssertionError(clean)
        target = self.targets[self.run_index]
        if self.invalidate_first and self.run_index == 0:
            target["extra_field"] = "contract-only-failure"
        self.run_index += 1
        return CommandResult(
            0,
            json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            2.0,
        )


class _TimeoutOnceRunner(_FakeRunner):
    def run(self, command, **kwargs):
        if tuple(command)[1] == "run" and self.run_index == 0:
            self.commands.append(
                (tuple(command), kwargs.get("input_text"), kwargs["timeout_seconds"])
            )
            self.run_index += 1
            raise OllamaCLITimeoutError("secret raw timeout details")
        return super().run(command, **kwargs)


class OllamaCLIBenchmarkTests(unittest.TestCase):
    def test_raw_contract_validation_does_not_repair_unicode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection = load_balanced_heldout(
                _write_dataset(Path(directory)),
                limit=9,
            )
        case = selection.cases[0]
        corrupted = dict(case.target)
        corrupted["answer"] = corrupted["answer"].replace("é", "�", 1)

        actual = _validate_raw_generated_answer(
            json.dumps(corrupted, ensure_ascii=False),
            case.capsule,
        )

        self.assertIn("�", actual["answer"])

    def test_content_parser_does_not_mistake_nested_abstention_for_root(self) -> None:
        corrupted = (
            'prefix {"abstained":false,"reason":"none",'
            '"missing_information":[]} suffix'
        )

        self.assertIsNone(_first_json_mapping(corrupted))

    def test_same_selection_contract_and_content_are_scored_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _write_dataset(root)
            selection = load_balanced_heldout(dataset, limit=9)
            runner = _FakeRunner(
                [case.target for case in selection.cases],
                invalidate_first=True,
            )
            report = run_ollama_cli_benchmark(
                selection,
                OllamaCLIConfig(
                    model="qwen2.5:14b-instruct-q4_0",
                    executable="ollama-test",
                    expected_manifest_id="5449194FF803",
                ),
                runner=runner,
                resolve_executable=False,
            )

        self.assertEqual(report["schema_version"], REPORT_SCHEMA_VERSION)
        self.assertEqual(report["model"]["manifest_id"], "5449194ff803")
        self.assertTrue(report["model"]["manifest_id_verified"])
        self.assertEqual(
            report["dataset"]["selection_sha256"], selection.selection_sha256
        )
        self.assertEqual(report["global_metrics"]["attempted"], 9)
        self.assertEqual(report["global_metrics"]["contract_valid"], 8)
        self.assertEqual(report["global_metrics"]["answer_anchors_all"], 9)
        self.assertEqual(
            report["global_metrics"]["rates"]["answer_anchor_recall_mean"], 1.0
        )
        self.assertEqual(report["global_metrics"]["content_core_exact"], 9)
        self.assertEqual(report["global_metrics"]["content_all_exact"], 9)
        self.assertEqual(report["global_metrics"]["invented_evidence_ids"], 0)
        self.assertTrue(report["model_release_succeeded"])

        run_calls = [call for call in runner.commands if call[0][1] == "run"]
        self.assertEqual(len(run_calls), 9)
        self.assertTrue(
            all(
                command == (
                    "ollama-test",
                    "run",
                    "qwen2.5:14b-instruct-q4_0",
                    "--format",
                    "json",
                    "--nowordwrap",
                )
                for command, _, _ in run_calls
            )
        )
        self.assertTrue(all(prompt and '"target":' not in prompt for _, prompt, _ in run_calls))
        self.assertEqual(runner.commands[-1][0][1], "stop")

        serialized = json.dumps(report, ensure_ascii=False)
        first_case = selection.cases[0]
        self.assertNotIn(str(root), serialized)
        self.assertNotIn(first_case.example_id, serialized)
        self.assertNotIn(first_case.capsule["question"], serialized)
        self.assertNotIn(first_case.capsule["evidence"][0]["text"], serialized)
        self.assertNotIn(first_case.target["answer"], serialized)

    def test_timeout_is_redacted_and_does_not_abort_following_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory))
            selection = load_balanced_heldout(dataset, limit=9)
            runner = _TimeoutOnceRunner([case.target for case in selection.cases])
            report = run_ollama_cli_benchmark(
                selection,
                OllamaCLIConfig(model="qwen2.5:14b", executable="ollama-test"),
                runner=runner,
                resolve_executable=False,
            )

        self.assertEqual(report["status"], "complete_with_failures")
        self.assertEqual(report["global_metrics"]["attempted"], 9)
        self.assertEqual(report["global_metrics"]["execution_error"], 1)
        self.assertEqual(report["global_metrics"]["content_core_exact"], 8)
        self.assertNotIn("secret raw", json.dumps(report))
        self.assertEqual(report["cases"][0]["error"]["message"], "delai de generation depasse")

    def test_dry_run_does_not_resolve_or_execute_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _write_dataset(root)
            selection = load_balanced_heldout(dataset, limit=9)
            impossible = str(root / "does-not-exist" / "ollama.exe")
            plan = benchmark_plan(
                selection,
                OllamaCLIConfig(model="qwen2.5:14b", executable=impossible),
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dataset",
                    str(dataset),
                    "--model",
                    "qwen2.5:14b",
                    "--ollama-executable",
                    impossible,
                    "--limit",
                    "9",
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        self.assertEqual(plan["status"], "dry-run")
        self.assertEqual(plan["selection_sha256"], selection.selection_sha256)
        self.assertEqual(result.returncode, 0, result.stderr)
        cli_plan = json.loads(result.stdout)
        self.assertEqual(cli_plan["status"], "dry-run")
        self.assertEqual(cli_plan["selected_count"], 9)
        self.assertEqual(cli_plan["model"], "qwen2.5:14b")
        self.assertEqual(result.stderr, "")

    def test_model_name_cannot_be_used_as_an_option_or_second_command(self) -> None:
        for value in ("--help", "qwen model", "qwen;ollama pull x", "qwen\nstop"):
            with self.subTest(value=value):
                with self.assertRaises(OllamaCLIBenchmarkError):
                    validate_ollama_cli_config(
                        OllamaCLIConfig(model=value),
                        resolve_executable=False,
                    )

    def test_expected_manifest_id_is_normalized_and_must_be_hex(self) -> None:
        clean = validate_ollama_cli_config(
            OllamaCLIConfig(expected_manifest_id="ABCDEF012345")
        )

        self.assertEqual(clean.expected_manifest_id, "abcdef012345")
        with self.assertRaisesRegex(OllamaCLIBenchmarkError, "hexadecimaux"):
            validate_ollama_cli_config(
                OllamaCLIConfig(expected_manifest_id="not-a-manifest")
            )

    def test_manifest_mismatch_aborts_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection = load_balanced_heldout(
                _write_dataset(Path(directory)),
                limit=9,
            )
        runner = _FakeRunner([case.target for case in selection.cases])

        with self.assertRaisesRegex(OllamaCLIBenchmarkError, "ne correspond pas"):
            run_ollama_cli_benchmark(
                selection,
                OllamaCLIConfig(
                    model="qwen2.5:14b-instruct-q4_0",
                    executable="ollama-test",
                    expected_manifest_id="deadbeefcafe",
                ),
                runner=runner,
                resolve_executable=False,
            )

        self.assertFalse(any(command[0][1] == "run" for command in runner.commands))

    def test_anchor_recall_accepts_paraphrase_but_not_wrong_factual_anchor(self) -> None:
        target = (
            "Non. Pour Personne fictive SYN-ABC-1, le code est code-fictif-42 "
            "au 2092-06-04 et le calcul donne 27 ans."
        )
        paraphrase = (
            "Non : SYN-ABC-1 correspond bien a code-fictif-42; "
            "au 2092-06-04, cela fait 27 ans."
        )
        wrong = paraphrase.replace("code-fictif-42", "code-fictif-43")

        accepted = answer_anchor_metrics(paraphrase, target)
        rejected = answer_anchor_metrics(wrong, target)
        self.assertTrue(accepted["answer_anchors_all"])
        self.assertEqual(accepted["answer_anchor_recall"], 1.0)
        self.assertFalse(rejected["answer_anchors_all"])
        self.assertLess(rejected["answer_anchor_recall"], 1.0)

    def test_prompt_reuses_native_contract_without_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory))
            selection = load_balanced_heldout(dataset, limit=9)
        prompt = render_ollama_prompt(selection.cases[0])
        self.assertIn("memory_native_request", prompt)
        self.assertIn("memory-native-answer-v1", prompt)
        self.assertIn(selection.cases[0].capsule["request_id"], prompt)
        self.assertNotIn('"target":', prompt)


if __name__ == "__main__":
    unittest.main()
