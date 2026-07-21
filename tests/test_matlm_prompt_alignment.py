from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_matlm import _generic_final_assistant_mask, _render_chat
from memory_agent.matlm_bridge import strict_chat_messages
from memory_agent.matlm_inference import _render_prompt
from memory_agent.memory_native_curriculum import build_synthetic_memory_curriculum


def _canonical(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class _VisibleChatTokenizer:
    """Gabarit déterministe qui rend chaque octet de message observable."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        if tokenize or kwargs:
            raise AssertionError("ce test compare uniquement le texte rendu")
        rendered = "".join(
            f"<|{message['role']}|>\n{message.get('content', '')}\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|assistant|>\n"
        return rendered


class _CharacterTokenizer:
    """Encode un caractère par entier pour exposer le prompt de secours."""

    def __call__(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("les jetons spéciaux doivent rester désactivés")
        return {"input_ids": [ord(character) for character in text]}


class MATLMPromptAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = build_synthetic_memory_curriculum(
            seed=20_260_723,
            count=90,
        )["examples"]

    def test_curriculum_prefix_is_the_shared_strict_prefix_for_every_task(self) -> None:
        seen_tasks = set()
        for row in self.rows:
            with self.subTest(task=row["task"], example_id=row["example_id"]):
                seen_tasks.add(row["task"])
                self.assertEqual(
                    row["messages"][:-1],
                    strict_chat_messages(row["memory_capsule"]),
                )
                self.assertEqual(
                    row["messages"][-1],
                    {"role": "assistant", "content": _canonical(row["target"])},
                )
        self.assertEqual(len(seen_tasks), 9)

    def test_training_and_inference_render_identical_prefix_bytes(self) -> None:
        tokenizer = _VisibleChatTokenizer()
        for row in self.rows:
            with self.subTest(task=row["task"], example_id=row["example_id"]):
                training_prompt = _render_chat(
                    tokenizer,
                    row["messages"][:-1],
                    prompt=True,
                )
                inference_prompt = _render_prompt(
                    tokenizer,
                    row["memory_capsule"],
                )
                self.assertEqual(training_prompt, inference_prompt)
                self.assertEqual(
                    training_prompt.encode("utf-8"),
                    inference_prompt.encode("utf-8"),
                )

    def test_generic_fallback_prefix_is_also_identical(self) -> None:
        tokenizer = _CharacterTokenizer()
        for row in self.rows:
            with self.subTest(task=row["task"], example_id=row["example_id"]):
                input_ids, labels = _generic_final_assistant_mask(
                    tokenizer,
                    row["messages"],
                )
                boundary = next(
                    index for index, label in enumerate(labels) if label != -100
                )
                training_prompt = "".join(chr(value) for value in input_ids[:boundary])
                inference_prompt = _render_prompt(
                    tokenizer,
                    row["memory_capsule"],
                )
                self.assertEqual(training_prompt, inference_prompt)

    def test_target_character_envelope_is_bounded_before_tokenizer_audit(self) -> None:
        # Ce garde-fou ne prétend pas convertir caractères -> tokens. Il force
        # simplement une nouvelle mesure avec le vrai tokenizer si les cibles
        # dépassent l'enveloppe actuellement auditée (638 octets au maximum).
        payloads = [_canonical(row["target"]) for row in self.rows]
        self.assertLessEqual(max(map(len, payloads)), 700)
        self.assertLessEqual(
            max(len(payload.encode("utf-8")) for payload in payloads),
            700,
        )


if __name__ == "__main__":
    unittest.main()
