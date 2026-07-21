from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.matlm_inference import (
    ANSWER_JSON_SCHEMA,
    INFERENCE_PLAN_SCHEMA,
    INFERENCE_STATUS_SCHEMA,
    InferenceConfig,
    MATLMInferenceError,
    MATLMInferenceSession,
    dry_run_plan,
    extract_json_object,
    inference_status,
    pretrained_options,
    validate_generated_answer,
    validate_inference_config,
)
from memory_agent.matlm_protocol import interactive_ready_frame
from memory_agent.native_llm_contract import ANSWER_SCHEMA_VERSION, build_capsule
from scripts.ask_matlm import _interactive


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ask_matlm.py"


def _assets(root: Path) -> tuple[Path, Path]:
    base = root / "granite-base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            {
                "model_type": "granite",
                "architectures": ["GraniteForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": str(base),
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"local-adapter")
    return base, adapter


def _capsule():
    return build_capsule(
        request_id="infer-001",
        question="Quelle année est indiquée ?",
        evidence=[
            {
                "evidence_id": "matlm:ev:1234",
                "text": "La date indiquée est 1898.",
                "space": "reference",
                "status": "verified",
                "confidence": 1.0,
                "temporal_context": "1898",
                "tags": ["date"],
            }
        ],
        max_evidence_ids=2,
    )


def _answer():
    return {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "request_id": "infer-001",
        "answer": "1898",
        "confidence": 0.99,
        "evidence_ids": ["matlm:ev:1234"],
        "calculations": [],
        "abstention": {
            "abstained": False,
            "reason": "none",
            "missing_information": [],
        },
    }


class MATLMInferenceTests(unittest.TestCase):
    def test_interactive_mode_emits_ready_before_the_first_answer(self) -> None:
        class FakeSession:
            def ask(self, capsule):
                self.capsule = capsule
                return _answer()

        session = FakeSession()
        input_stream = io.StringIO(json.dumps(_capsule()) + "\n")
        output_stream = io.StringIO()
        with (
            patch("scripts.ask_matlm.sys.stdin", input_stream),
            patch("scripts.ask_matlm.sys.stdout", output_stream),
        ):
            result = _interactive(session)

        frames = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertEqual(result, 0)
        self.assertEqual(frames, [interactive_ready_frame(), _answer()])
        self.assertEqual(session.capsule, _capsule())

    def test_session_reexecutes_calendar_age_before_returning_answer(self) -> None:
        evidence_ids = ["matlm:ev:birth", "matlm:ev:event"]
        capsule = build_capsule(
            request_id="infer-calendar-age",
            question="Quel âge ?",
            evidence=[
                {
                    "evidence_id": evidence_id,
                    "text": text,
                    "space": "reference",
                    "status": "verified",
                    "confidence": 1.0,
                    "temporal_context": temporal,
                    "tags": ["date"],
                }
                for evidence_id, text, temporal in (
                    (evidence_ids[0], "Naissance le 2065-04-23.", "2065-04-23"),
                    (evidence_ids[1], "Événement le 2092-06-04.", "2092-06-04"),
                )
            ],
            max_evidence_ids=2,
            max_calculations=1,
        )
        generated = {
            "schema_version": ANSWER_SCHEMA_VERSION,
            "request_id": "infer-calendar-age",
            "answer": "Le calcul donne 26 ans, selon la précision disponible.",
            "confidence": 0.98,
            "evidence_ids": evidence_ids,
            "calculations": [
                {
                    "calculation_id": "calc:model-invented",
                    "expression": "calendar_age(2065-04-23,2092-06-04,precision=day)",
                    "reported_result": "26",
                    "unit": "ans",
                    "evidence_ids": evidence_ids,
                }
            ],
            "abstention": {
                "abstained": False,
                "reason": "none",
                "missing_information": [],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            base, adapter = _assets(Path(directory))
            config = InferenceConfig(adapter_path=adapter, base_model=str(base))
            with (
                patch(
                    "memory_agent.matlm_inference._load_runtime_assets",
                    return_value=SimpleNamespace(effective_mode="bf16"),
                ),
                patch("memory_agent.matlm_inference._release_runtime_assets"),
                patch(
                    "memory_agent.matlm_inference._generate_text",
                    return_value=json.dumps(generated),
                ),
            ):
                with MATLMInferenceSession(config) as session:
                    actual = session.ask(capsule)
        self.assertIn("27 ans", actual["answer"])
        self.assertEqual(actual["calculations"][0]["reported_result"], "27")
        self.assertEqual(
            actual["calculations"][0]["calculation_id"],
            "calc:"
            + hashlib.sha256(b"infer-calendar-age").hexdigest()[:20],
        )

    def test_config_is_local_granite_peft_and_network_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, adapter = _assets(Path(directory))
            offline = validate_inference_config(
                InferenceConfig(adapter_path=adapter, base_model=str(base), load_mode="auto")
            )
            online = validate_inference_config(
                InferenceConfig(
                    adapter_path=adapter,
                    base_model=str(base),
                    allow_model_download=True,
                )
            )

            self.assertEqual(offline.adapter_path, adapter.resolve())
            self.assertEqual(offline.base_model, str(base.resolve()))
            self.assertEqual(
                pretrained_options(offline),
                {"local_files_only": True, "trust_remote_code": False},
            )
            self.assertEqual(
                pretrained_options(online),
                {"local_files_only": False, "trust_remote_code": False},
            )

            bad_base = Path(directory) / "other-model"
            bad_base.mkdir()
            (bad_base / "config.json").write_text(
                json.dumps({"model_type": "llama"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(MATLMInferenceError, "architecture Granite"):
                validate_inference_config(
                    InferenceConfig(adapter_path=adapter, base_model=str(bad_base))
                )

    def test_status_and_dry_run_need_no_runtime_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, adapter = _assets(Path(directory))
            config = InferenceConfig(
                adapter_path=adapter,
                base_model=str(base),
                load_mode="bf16",
            )
            with patch(
                "memory_agent.matlm_inference._runtime_imports",
                side_effect=AssertionError("heavy import forbidden"),
            ):
                status = inference_status(config)
                plan = dry_run_plan(config, _capsule())

        self.assertEqual(status["schema_version"], INFERENCE_STATUS_SCHEMA)
        self.assertEqual(status["network"], "offline")
        self.assertEqual(plan["schema_version"], INFERENCE_PLAN_SCHEMA)
        self.assertEqual(plan["status"], "dry-run")
        self.assertFalse(plan["generation"]["do_sample"])
        self.assertEqual(plan["generation"]["num_beams"], 1)
        self.assertEqual(plan["capsule"]["evidence_count"], 1)

    def test_status_reports_missing_assets_without_raising(self) -> None:
        missing = Path(tempfile.gettempdir()) / "matlm-definitely-missing-adapter"
        status = inference_status(InferenceConfig(adapter_path=missing))

        self.assertFalse(status["ready_for_runtime_attempt"])
        self.assertEqual(status["schema_version"], INFERENCE_STATUS_SCHEMA)
        self.assertTrue(any("adaptateur PEFT" in issue for issue in status["issues"]))

    def test_extracts_one_json_object_and_rejects_extra_model_text(self) -> None:
        raw = json.dumps(_answer(), ensure_ascii=False)
        fenced = "```json\n" + raw + "\n```"
        self.assertEqual(json.loads(extract_json_object(fenced)), _answer())
        self.assertEqual(validate_generated_answer(fenced, _capsule()), _answer())

        with self.assertRaisesRegex(MATLMInferenceError, "texte interdit avant"):
            extract_json_object("Voici la réponse: " + raw)
        with self.assertRaisesRegex(MATLMInferenceError, "après l'objet JSON"):
            extract_json_object(raw + "\nexplication")
        with self.assertRaisesRegex(MATLMInferenceError, "après l'objet JSON"):
            extract_json_object(raw + raw)

    def test_generated_answer_is_validated_against_capsule_evidence(self) -> None:
        invalid = _answer()
        invalid["evidence_ids"] = ["matlm:ev:invented"]
        with self.assertRaisesRegex(MATLMInferenceError, "preuves absentes"):
            validate_generated_answer(json.dumps(invalid), _capsule())

    def test_session_allows_one_model_and_releases_it_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, adapter = _assets(Path(directory))
            config = InferenceConfig(adapter_path=adapter, base_model=str(base))
            assets_one = SimpleNamespace(effective_mode="bf16")
            assets_two = SimpleNamespace(effective_mode="qlora-nf4")
            released = []

            with (
                patch(
                    "memory_agent.matlm_inference._load_runtime_assets",
                    side_effect=[assets_one, assets_two],
                ),
                patch(
                    "memory_agent.matlm_inference._release_runtime_assets",
                    side_effect=lambda assets: released.append(assets),
                ),
                patch(
                    "memory_agent.matlm_inference._generate_text",
                    return_value=json.dumps(_answer()),
                ),
            ):
                first = MATLMInferenceSession(config).load()
                second = MATLMInferenceSession(config)
                try:
                    self.assertTrue(first.loaded)
                    self.assertEqual(first.effective_mode, "bf16")
                    self.assertEqual(first.ask(_capsule()), _answer())
                    self.assertEqual(
                        first.generate_json(
                            _capsule(),
                            output_schema=ANSWER_JSON_SCHEMA,
                            mode="specialized",
                        ),
                        _answer(),
                    )
                    with self.assertRaisesRegex(MATLMInferenceError, "déjà chargé"):
                        second.load()
                finally:
                    first.close()
                second.load()
                self.assertEqual(second.effective_mode, "qlora-nf4")
                second.close()

        self.assertEqual(released, [assets_one, assets_two])

    def test_load_failure_releases_global_model_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, adapter = _assets(Path(directory))
            config = InferenceConfig(adapter_path=adapter, base_model=str(base))
            good = SimpleNamespace(effective_mode="bf16")
            with (
                patch(
                    "memory_agent.matlm_inference._load_runtime_assets",
                    side_effect=[MATLMInferenceError("dépendance absente"), good],
                ),
                patch("memory_agent.matlm_inference._release_runtime_assets"),
            ):
                with self.assertRaisesRegex(MATLMInferenceError, "dépendance absente"):
                    MATLMInferenceSession(config).load()
                recovered = MATLMInferenceSession(config).load()
                recovered.close()

    def test_cli_status_and_dry_run_do_not_import_ml_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, adapter = _assets(root)
            capsule_path = root / "capsule.json"
            capsule_path.write_text(
                json.dumps(_capsule(), ensure_ascii=False), encoding="utf-8"
            )
            blocker = root / "sitecustomize.py"
            blocker.write_text(
                "import importlib.abc, sys\n"
                "class Block(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname.split('.')[0] in "
                "{'torch','transformers','peft','accelerate','bitsandbytes'}:\n"
                "            raise AssertionError('ML import forbidden: ' + fullname)\n"
                "        return None\n"
                "sys.meta_path.insert(0, Block())\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(root)
            common = [
                sys.executable,
                str(SCRIPT_PATH),
                "--adapter",
                str(adapter),
                "--base-model",
                str(base),
            ]
            status_run = subprocess.run(
                [*common, "--status"],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            dry_run = subprocess.run(
                [*common, "--capsule", str(capsule_path), "--dry-run"],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        self.assertEqual(status_run.returncode, 0, status_run.stderr)
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertEqual(json.loads(status_run.stdout)["schema_version"], INFERENCE_STATUS_SCHEMA)
        self.assertEqual(json.loads(dry_run.stdout)["status"], "dry-run")
        self.assertEqual(status_run.stderr, "")
        self.assertEqual(dry_run.stderr, "")


if __name__ == "__main__":
    unittest.main()
