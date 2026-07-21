from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.matlm_training import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    MANIFEST_SCHEMA,
    MAX_EVALUATION_HISTORY,
    MATLMTrainingError,
    TrainingConfig,
    build_run_manifest,
    complete_run_manifest,
    load_training_jsonl,
    summarize_evaluation_history,
    validate_output_path,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_matlm.py"


def _example(identifier: str = "example-1", *, task: str = "direct_recall") -> dict:
    answer = "Le fait est documenté. [claim-1] [source-1]"
    return {
        "schema_version": "memory-native-sft-example-v1",
        "example_id": identifier,
        "task": task,
        "memory_capsule": {"schema_version": "facts-only-capsule-v1", "facts": []},
        "messages": [
            {"role": "system", "content": "Utilise uniquement la capsule."},
            {"role": "user", "content": "CAPSULE={}\nQUESTION=Quel fait ?"},
            {"role": "assistant", "content": answer},
        ],
        "target": {
            "answer": answer,
            "grounding": "fully_supported",
            "citations": {"claim_ids": ["claim-1"], "source_ids": ["source-1"]},
        },
        "provenance": {
            "facts_sha256": "a" * 64,
            "claim_ids": ["claim-1"],
            "template_version": "memory-native-templates-v1",
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _load_script_module():
    specification = importlib.util.spec_from_file_location("train_matlm_for_tests", SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class MATLMTrainingTests(unittest.TestCase):
    def test_evaluation_history_keeps_latest_and_is_bounded(self) -> None:
        log_history = [
            {"loss": 1.0, "step": 1},
            *[
                {
                    "eval_loss": 1.0 / (index + 1),
                    "eval_runtime": index + 0.5,
                    "epoch": index / 10,
                    "step": index * 10,
                    "ignored_payload": "not-an-evaluation-metric",
                }
                for index in range(MAX_EVALUATION_HISTORY + 3)
            ],
        ]

        summary = summarize_evaluation_history(log_history)

        self.assertEqual(summary["record_count"], MAX_EVALUATION_HISTORY + 3)
        self.assertEqual(len(summary["history"]), MAX_EVALUATION_HISTORY)
        self.assertTrue(summary["history_truncated"])
        self.assertEqual(summary["history"][0]["step"], 30)
        self.assertEqual(summary["latest"]["step"], (MAX_EVALUATION_HISTORY + 2) * 10)
        self.assertEqual(summary["latest"], summary["history"][-1])
        self.assertNotIn("ignored_payload", json.dumps(summary))

    def test_complete_manifest_requires_saved_nonempty_adapter_first(self) -> None:
        manifest = {"schema_version": MANIFEST_SCHEMA, "status": "planned"}
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            with self.assertRaisesRegex(MATLMTrainingError, "complete interdit"):
                complete_run_manifest(
                    manifest,
                    adapter_dir=adapter,
                    training_metrics={"train_loss": 0.1},
                    evaluation_log_history=[{"eval_loss": 0.02, "step": 300}],
                    assistant_masking_modes=["assistant-mask:template"],
                    tokenization_calls_truncated=0,
                )
            self.assertEqual(manifest["status"], "planned")

            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"saved-adapter")
            completed = complete_run_manifest(
                manifest,
                adapter_dir=adapter,
                training_metrics={"train_loss": 0.1},
                evaluation_log_history=[
                    {"eval_loss": 0.03736, "step": 100},
                    {"eval_loss": 0.02320, "step": 200},
                    {"eval_loss": 0.022718, "step": 300},
                ],
                assistant_masking_modes=["assistant-mask:template"],
                tokenization_calls_truncated=0,
            )

        self.assertEqual(manifest["status"], "planned")
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["result"]["evaluation"]["latest"]["eval_loss"], 0.022718)
        self.assertEqual(len(completed["result"]["evaluation"]["history"]), 3)
        self.assertEqual(
            completed["result"]["adapter_artifacts"]["weights"],
            "adapter_model.safetensors",
        )

    def test_validates_jsonl_and_builds_reproducible_safe_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "train.jsonl"
            _write_jsonl(training_path, [_example("one"), _example("two", task="abstain")])
            loaded = load_training_jsonl(training_path)
            output = root / "run"
            config = TrainingConfig(train_jsonl=training_path, output_dir=output)
            manifest = build_run_manifest(config, loaded.summary, None, status="dry-run")

        self.assertEqual(loaded.summary.example_count, 2)
        self.assertEqual(loaded.summary.task_counts, {"abstain": 1, "direct_recall": 1})
        self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA)
        self.assertEqual(manifest["model"]["base_model"], DEFAULT_BASE_MODEL)
        self.assertIsNone(manifest["model"]["cache_dir"])
        self.assertEqual(manifest["model"]["mode"], "qlora-nf4")
        self.assertEqual(manifest["model"]["fallback"], "bf16-attention")
        self.assertEqual(manifest["model"]["batch_size"], 1)
        self.assertEqual(manifest["model"]["sequence_length"], 512)
        self.assertTrue(manifest["model"]["gradient_checkpointing"])
        self.assertFalse(manifest["model"]["push_to_hub"])
        self.assertFalse(manifest["safety"]["external_inference_api"])
        self.assertNotIn("rows", json.dumps(manifest))

    def test_rejects_evaluation_fields_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaked = _example()
            leaked["expected_answer_fragments"] = ["secret"]
            leaked_path = root / "leaked.jsonl"
            _write_jsonl(leaked_path, [leaked])
            with self.assertRaisesRegex(MATLMTrainingError, "champ d'évaluation interdit"):
                load_training_jsonl(leaked_path)

            duplicate_path = root / "duplicate.jsonl"
            _write_jsonl(duplicate_path, [_example("same"), _example("same")])
            with self.assertRaisesRegex(MATLMTrainingError, "example_id répété"):
                load_training_jsonl(duplicate_path)

    def test_rejects_train_eval_overlap_and_broad_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            _write_jsonl(train_path, [_example("overlap")])
            _write_jsonl(eval_path, [_example("overlap", task="other")])
            train = load_training_jsonl(train_path)
            evaluation = load_training_jsonl(eval_path)
            config = TrainingConfig(train_jsonl=train_path, eval_jsonl=eval_path, output_dir=root / "run")
            with self.assertRaisesRegex(MATLMTrainingError, "fuite train/eval"):
                build_run_manifest(config, train.summary, evaluation.summary, status="dry-run")
            with self.assertRaisesRegex(MATLMTrainingError, "racine"):
                validate_output_path(Path(root.anchor), [train_path])

    def test_output_must_not_overwrite_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "train.jsonl"
            _write_jsonl(training_path, [_example()])
            output = root / "run"
            output.mkdir()
            (output / "keep.txt").write_text("user data", encoding="utf-8")
            with self.assertRaisesRegex(MATLMTrainingError, "doit être vide"):
                validate_output_path(output, [training_path])

    def test_assistant_mask_is_used_and_prompt_tokens_are_ignored(self) -> None:
        module = _load_script_module()

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return {
                    "input_ids": [10, 11, 12, 13],
                    "assistant_masks": [0, 0, 1, 1],
                }

        encoded = module.tokenize_example(FakeTokenizer(), _example(), sequence_length=512)
        self.assertEqual(encoded["input_ids"], [10, 11, 12, 13])
        self.assertEqual(encoded["labels"], [-100, -100, 12, 13])
        self.assertEqual(encoded["masking_mode"], "assistant-mask:template")

    def test_cli_dry_run_needs_no_ml_import_and_writes_no_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "train.jsonl"
            output = root / "run"
            manifest_path = root / "dry-run-manifest.json"
            _write_jsonl(training_path, [_example()])

            blocker = root / "sitecustomize.py"
            blocker.write_text(
                "import importlib.abc, sys\n"
                "class Block(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname.split('.')[0] in "
                "{'torch','transformers','peft','accelerate','bitsandbytes'}:\n"
                "            raise AssertionError('ML import forbidden in dry-run: ' + fullname)\n"
                "        return None\n"
                "sys.meta_path.insert(0, Block())\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(root)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--train-jsonl",
                    str(training_path),
                    "--output-dir",
                    str(output),
                    "--manifest-output",
                    str(manifest_path),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "dry-run")
            self.assertTrue(manifest_path.is_file())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
