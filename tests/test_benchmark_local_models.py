from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import benchmark_local_models as benchmark  # noqa: E402


class FakeClient:
    def __init__(self, responses: dict[tuple[str, str], dict]):
        self.responses = responses
        self.calls: list[tuple[str, str, object]] = []

    def request(self, method: str, path: str, payload=None):
        self.calls.append((method, path, payload))
        response = self.responses[(method, path)]
        return json.loads(json.dumps(response))


class BenchmarkValidationTests(unittest.TestCase):
    def test_json_and_endpoint_validation_are_strict_and_local(self) -> None:
        self.assertEqual(
            benchmark.validate_local_endpoint("http://127.0.0.1:11434"),
            "http://127.0.0.1:11434",
        )
        self.assertEqual(
            benchmark.validate_local_endpoint("http://[::1]:1234/v1"),
            "http://[::1]:1234/v1",
        )
        for endpoint in (
            "https://127.0.0.1:11434",
            "http://example.com:11434",
            "http://user:secret@127.0.0.1:11434",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(benchmark.BenchmarkError):
                    benchmark.validate_local_endpoint(endpoint)
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark.strict_json_loads('{"a":1,"a":2}')
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark.strict_json_loads('{"a":NaN}')

    def test_dataset_contract_and_separate_capsules(self) -> None:
        dataset = {
            "evaluation_questions": [
                {
                    "id": "q-1",
                    "question": "Quelle est la couleur ?",
                    "expected_answer_fragments": ["bleu"],
                    "forbidden_answer_fragments": ["rouge"],
                    "answer_status": "answerable",
                    "capsule": {"evidence": ["ancienne"]},
                },
                {
                    "id": "q-2",
                    "question": "Information absente ?",
                    "expected_answer_fragments": [],
                    "forbidden_answer_fragments": ["invention"],
                    "answer_status": "unanswerable",
                },
            ]
        }
        capsules = {"capsules": {"q-1": {"evidence": ["bleu"]}}}

        questions = benchmark.parse_questions(dataset, capsules)

        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].capsule, {"evidence": ["bleu"]})
        self.assertIsNone(questions[1].capsule)
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark.parse_questions(
                {"evaluation_questions": [{**dataset["evaluation_questions"][0], "id": "q", "answer_status": "maybe"}]}
            )
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark.parse_questions(dataset, {"capsules": {"absente": {}}})


class InventoryTests(unittest.TestCase):
    def test_ollama_tags_are_deduplicated_by_digest_and_classified(self) -> None:
        client = FakeClient({
            ("GET", "/api/tags"): {
                "models": [
                    {"name": "qwen:latest", "digest": "sha256:text", "size": 10, "details": {}},
                    {"name": "qwen:7b", "digest": "sha256:text", "size": 10, "details": {}},
                    {"name": "llama-vision:11b", "digest": "sha256:vision", "details": {"families": ["mllama", "clip"]}},
                    {"name": "coder:1.5b-base", "digest": "sha256:base", "details": {}},
                ]
            }
        })

        models = benchmark.inventory_ollama_http(client)

        self.assertEqual(len(models), 3)
        text = next(model for model in models if model.digest == "sha256:text")
        self.assertEqual(text.model_id, "qwen:7b")
        self.assertEqual(set(text.tags), {"qwen:latest", "qwen:7b"})
        self.assertEqual(
            {model.kind for model in models}, {"text", "vision", "base"}
        )
        self.assertEqual(
            [model.kind for model in benchmark.filter_models(models, {"vision"}, [], 10)],
            ["vision"],
        )
        selected_alias = benchmark.filter_models(models, {"text"}, ["qwen:latest"], 10)
        self.assertEqual(selected_alias[0].model_id, "qwen:7b")

    def test_explicit_ollama_manifest_path_is_read_without_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "models" / "manifests" / "registry.ollama.ai" / "library" / "qwen"
            root.mkdir(parents=True)
            manifest = {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": "sha256:same",
                        "size": 123,
                    }
                ]
            }
            (root / "latest").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "7b").write_text(json.dumps(manifest), encoding="utf-8")

            models = benchmark.inventory_ollama_manifests(Path(directory) / "models")

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].model_id, "qwen:7b")
        self.assertEqual(set(models[0].tags), {"qwen:7b", "qwen:latest"})
        self.assertEqual(models[0].size_bytes, 123)

    def test_lm_studio_openai_inventory_and_generation_contract(self) -> None:
        client = FakeClient({
            ("GET", "/models"): {"data": [{"id": "local-text"}]},
            ("POST", "/chat/completions"): {
                "choices": [{"message": {"content": "reponse locale"}}]
            },
        })
        models = benchmark.inventory_lmstudio(client)
        backend = benchmark.LMStudioBackend(client)

        answer = backend.generate(
            models[0].model_id,
            [{"role": "user", "content": "test"}],
            32,
        )

        self.assertEqual(answer, "reponse locale")
        self.assertEqual(models[0].backend, "lmstudio")
        method, path, payload = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/chat/completions"))
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["temperature"], 0)


class ScoringAndRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.answerable = benchmark.EvaluationQuestion(
            "q1", "Quelle ville ?", ("Quebec",), ("Montreal",),
            "answerable", {"evidence": [{"text": "Quebec"}]},
        )
        self.unanswerable = benchmark.EvaluationQuestion(
            "q2", "Qui est inconnu ?", (), ("Alice",), "unanswerable", None
        )

    def test_scoring_distinguishes_accuracy_abstention_and_hallucination(self) -> None:
        correct = benchmark.score_answer("La ville est Quebec.", self.answerable)
        forbidden = benchmark.score_answer("La ville est Montreal.", self.answerable)
        abstention = benchmark.score_answer("JE_NE_SAIS_PAS", self.unanswerable)
        invented = benchmark.score_answer("C'est Alice", self.unanswerable)

        self.assertTrue(correct["correct"])
        self.assertFalse(correct["hallucinated"])
        self.assertTrue(forbidden["hallucinated"])
        self.assertTrue(abstention["correct"])
        self.assertTrue(abstention["abstained"])
        self.assertTrue(invented["hallucinated"])

    def test_baseline_and_capsule_use_same_question_and_report_errors(self) -> None:
        class FakeBackend:
            name = "fake-local"

            def __init__(self):
                self.messages: list[list[dict[str, str]]] = []

            def generate(self, model_id, messages, max_tokens):
                self.messages.append(messages)
                if len(self.messages) == 1:
                    raise benchmark.RequestFailure("timeout", "delai depasse")
                return "Quebec"

        class Clock:
            def __init__(self):
                self.value = 0.0

            def __call__(self):
                self.value += 0.01
                return self.value

        backend = FakeBackend()
        model = benchmark.ModelInfo(
            "ollama", "qwen:7b", "sha256:x", ("qwen:7b",), "text"
        )

        report = benchmark.run_benchmark(
            backend, [model], [self.answerable], max_tokens=32, clock=Clock()
        )

        self.assertEqual(report["summary"]["requests"], 2)
        self.assertEqual(report["summary"]["errors"], 1)
        result = report["models"][0]
        self.assertEqual(result["metrics"]["baseline"]["errors"], 1)
        self.assertEqual(result["metrics"]["memory"]["accuracy_percent"], 100.0)
        self.assertEqual(result["results"][0]["error"]["code"], "timeout")
        self.assertNotIn("capsule suivante", backend.messages[0][0]["content"].casefold())
        self.assertIn("preuves ci-dessous", backend.messages[1][0]["content"].casefold())
        self.assertEqual(
            backend.messages[0][1]["content"], backend.messages[1][1]["content"]
        )
        json.dumps(report, ensure_ascii=False, allow_nan=False)

    def test_request_quota_prevents_heavy_run(self) -> None:
        models = [
            benchmark.ModelInfo("ollama", f"m{i}", str(i), (f"m{i}",), "text")
            for i in range(51)
        ]
        questions = [self.answerable] * 10
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark.run_benchmark(object(), models, questions)

    def test_ollama_base_model_uses_generate_endpoint(self) -> None:
        client = FakeClient({
            ("POST", "/api/generate"): {"response": "Quebec"},
        })
        backend = benchmark.OllamaBackend(client)
        model = benchmark.ModelInfo(
            "ollama", "coder:base", "sha256:base", ("coder:base",), "base"
        )

        report = benchmark.run_benchmark(
            backend, [model], [self.answerable], max_tokens=16
        )

        self.assertEqual(report["summary"]["errors"], 0)
        self.assertEqual(
            [path for _, path, _ in client.calls],
            ["/api/generate", "/api/generate", "/api/generate"],
        )
        self.assertEqual(report["models"][0]["release"]["status"], "released")


if __name__ == "__main__":
    unittest.main()
