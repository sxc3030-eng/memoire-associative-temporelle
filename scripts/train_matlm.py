#!/usr/bin/env python3
"""Entraîne localement l'adaptateur MAT-LM, sans service d'inférence externe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import inspect
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_agent.matlm_training import (  # noqa: E402
    DEFAULT_ATTENTION_MODULES,
    DEFAULT_BASE_MODEL,
    MATLMTrainingError,
    TrainingConfig,
    build_run_manifest,
    complete_run_manifest,
    load_training_jsonl,
    validate_config,
    validate_output_path,
    write_manifest,
)


CALENDAR_AGE_TOOL = {
    "type": "function",
    "function": {
        "name": "calendar_age",
        "description": "Calcule un âge civil à partir de dates fournies par la capsule.",
        "parameters": {
            "type": "object",
            "properties": {
                "birth_date": {"type": "string"},
                "event_date": {"type": "string"},
                "event_precision": {"type": "string"},
            },
            "required": ["birth_date", "event_date", "event_precision"],
        },
    },
}


class MATLMRuntimeError(RuntimeError):
    """Training could not run with the local ML stack."""


def _runtime_imports() -> dict[str, Any]:
    """Import the heavy stack only after validation and outside dry-run mode."""

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        package = getattr(error, "name", None) or "inconnue"
        raise MATLMRuntimeError(
            "dépendance d'entraînement absente: "
            f"{package}. Installez requirements-training.txt et PyTorch XPU."
        ) from error
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _has_tool_calls(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(message.get("tool_calls") for message in messages)


def _template_call(tokenizer: Any, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Any:
    options = dict(kwargs)
    if _has_tool_calls(messages):
        options["tools"] = [CALENDAR_AGE_TOOL]
    try:
        return tokenizer.apply_chat_template(list(messages), **options)
    except TypeError:
        if "tools" not in options:
            raise
        options.pop("tools")
        return tokenizer.apply_chat_template(list(messages), **options)


def _template_safe_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Represent tool traffic as text for chat templates without tool support."""

    result: list[dict[str, Any]] = []
    for message in messages:
        role = str(message["role"])
        content = str(message.get("content", ""))
        if message.get("tool_calls"):
            serialized = _json_text({"tool_calls": message["tool_calls"]})
            content = f"{content}\n<appel_outil>{serialized}</appel_outil>".strip()
        if role == "tool":
            name = str(message.get("name", "outil"))
            call_id = str(message.get("tool_call_id", "inconnu"))
            content = f"<résultat_outil nom={name!r} id={call_id!r}>{content}</résultat_outil>"
            role = "user"
        result.append({"role": role, "content": content})
    return result


def _flat_integers(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise MATLMRuntimeError("le tokenizer n'a pas retourné une liste de jetons")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise MATLMRuntimeError("le tokenizer a retourné des jetons invalides") from error


def _preferred_assistant_mask(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int]] | None:
    encoded = _template_call(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        return None
    mask_value = None
    for key in ("assistant_masks", "assistant_mask", "assistant_tokens_mask"):
        if key in encoded:
            mask_value = encoded[key]
            break
    if mask_value is None:
        return None
    input_ids = _flat_integers(encoded["input_ids"])
    mask = _flat_integers(mask_value)
    if len(input_ids) != len(mask) or not any(mask):
        return None
    return input_ids, [-100 if flag == 0 else token for token, flag in zip(input_ids, mask)]


def _render_chat(tokenizer: Any, messages: Sequence[Mapping[str, Any]], *, prompt: bool) -> str:
    rendered = _template_call(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=prompt,
    )
    if not isinstance(rendered, str):
        raise MATLMRuntimeError("le gabarit de dialogue n'a pas retourné de texte")
    return rendered


def _common_prefix_length(first: Sequence[int], second: Sequence[int]) -> int:
    index = 0
    maximum = min(len(first), len(second))
    while index < maximum and first[index] == second[index]:
        index += 1
    return index


def _final_assistant_mask(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int]]:
    if messages[-1].get("role") != "assistant":
        raise MATLMRuntimeError("la conversation ne se termine pas par l'assistant")
    full_text = _render_chat(tokenizer, messages, prompt=False)
    prompt_text = _render_chat(tokenizer, messages[:-1], prompt=True)
    full_ids = _flat_integers(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    prompt_ids = _flat_integers(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    boundary = _common_prefix_length(full_ids, prompt_ids)
    if boundary >= len(full_ids):
        raise MATLMRuntimeError("impossible d'isoler les jetons de la réponse assistant")
    labels = [-100] * boundary + full_ids[boundary:]
    return full_ids, labels


def _generic_final_assistant_mask(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int]]:
    def render(items: Sequence[Mapping[str, Any]]) -> str:
        blocks: list[str] = []
        for item in items:
            role = str(item["role"])
            content = str(item.get("content", ""))
            if item.get("tool_calls"):
                content += "\n" + _json_text({"tool_calls": item["tool_calls"]})
            blocks.append(f"<|{role}|>\n{content}\n")
        return "".join(blocks)

    prompt_text = render(messages[:-1]) + "<|assistant|>\n"
    full_text = prompt_text + str(messages[-1]["content"]) + "\n"
    full_ids = _flat_integers(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    prompt_ids = _flat_integers(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    boundary = _common_prefix_length(full_ids, prompt_ids)
    if boundary >= len(full_ids):
        raise MATLMRuntimeError("impossible d'isoler la réponse avec le gabarit de secours")
    return full_ids, [-100] * boundary + full_ids[boundary:]


def tokenize_example(tokenizer: Any, row: Mapping[str, Any], sequence_length: int) -> dict[str, Any]:
    """Tokenize one chat while masking non-assistant tokens from the loss."""

    messages = list(row["messages"])
    candidates: list[tuple[str, Sequence[Mapping[str, Any]]]] = [("template", messages)]
    safe_messages = _template_safe_messages(messages)
    if safe_messages != messages:
        candidates.append(("template-tool-text", safe_messages))

    errors: list[str] = []
    for label, candidate in candidates:
        try:
            preferred = _preferred_assistant_mask(tokenizer, candidate)
            if preferred is not None:
                input_ids, labels = preferred
                masking_mode = f"assistant-mask:{label}"
                break
        except Exception as error:  # Tokenizer templates expose several error classes.
            errors.append(f"{label}/assistant-mask: {type(error).__name__}: {error}")
    else:
        for label, candidate in candidates:
            try:
                input_ids, labels = _final_assistant_mask(tokenizer, candidate)
                masking_mode = f"final-assistant:{label}"
                break
            except Exception as error:
                errors.append(f"{label}/final-assistant: {type(error).__name__}: {error}")
        else:
            try:
                input_ids, labels = _generic_final_assistant_mask(tokenizer, messages)
                masking_mode = "final-assistant:generic"
            except Exception as error:
                errors.append(f"generic: {type(error).__name__}: {error}")
                detail = " | ".join(errors)[-2_000:]
                raise MATLMRuntimeError(f"échec du gabarit de dialogue: {detail}") from error

    if len(input_ids) != len(labels) or not input_ids:
        raise MATLMRuntimeError("le tokenizer a produit une séquence incohérente")
    truncated = len(input_ids) > sequence_length
    if truncated:
        # The target is at the end. Keeping the suffix protects it from prompt truncation.
        input_ids = input_ids[-sequence_length:]
        labels = labels[-sequence_length:]
    if not any(label != -100 for label in labels):
        raise MATLMRuntimeError(
            "aucun jeton assistant ne reste après troncature; augmentez --sequence-length"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "masking_mode": masking_mode,
        "truncated": truncated,
    }


class _ChatDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]], tokenizer: Any, sequence_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.masking_modes: set[str] = set()
        self.truncated_examples = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        encoded = tokenize_example(self.tokenizer, self.rows[index], self.sequence_length)
        self.masking_modes.add(str(encoded.pop("masking_mode")))
        if encoded.pop("truncated"):
            self.truncated_examples += 1
        return encoded


class _CausalLMCollator:
    def __init__(self, torch_module: Any, pad_token_id: int) -> None:
        self.torch = torch_module
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * padding)
            attention_masks.append(list(feature["attention_mask"]) + [0] * padding)
            labels.append(list(feature["labels"]) + [-100] * padding)
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(attention_masks, dtype=self.torch.long),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
        }


def _device_details(torch: Any, config: TrainingConfig) -> tuple[str, dict[str, Any]]:
    if config.device == "cpu":
        return "cpu", {"type": "cpu", "name": "CPU"}
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        raise MATLMRuntimeError(
            "PyTorch XPU ne détecte pas l'Intel Arc. Vérifiez le pilote et la roue PyTorch XPU."
        )
    count = int(xpu.device_count())
    if config.device_index >= count:
        raise MATLMRuntimeError(
            f"XPU {config.device_index} demandé, mais seulement {count} périphérique(s) détecté(s)"
        )
    xpu.set_device(config.device_index)
    name = str(xpu.get_device_name(config.device_index))
    bf16_supported = None
    if hasattr(xpu, "is_bf16_supported"):
        bf16_supported = bool(xpu.is_bf16_supported())
    return f"xpu:{config.device_index}", {
        "type": "xpu",
        "index": config.device_index,
        "count": count,
        "name": name,
        "bf16_supported": bf16_supported,
    }


def _clear_device(torch: Any, device: str) -> None:
    gc.collect()
    if device.startswith("xpu") and getattr(torch, "xpu", None) is not None:
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass


def _model_source(config: TrainingConfig) -> str:
    candidate = Path(config.base_model).expanduser()
    return str(candidate.resolve()) if candidate.exists() else config.base_model


def _from_pretrained_options(config: TrainingConfig) -> dict[str, Any]:
    options: dict[str, Any] = {
        "local_files_only": not config.allow_model_download,
        "trust_remote_code": False,
    }
    if config.cache_dir is not None:
        options["cache_dir"] = str(config.cache_dir)
    return options


def _load_model(
    stack: Mapping[str, Any], config: TrainingConfig, device: str
) -> tuple[Any, str, str | None]:
    torch = stack["torch"]
    model_class = stack["AutoModelForCausalLM"]
    source = _model_source(config)
    common = {
        **_from_pretrained_options(config),
        "low_cpu_mem_usage": True,
    }
    quantization_error: str | None = None
    model = None

    if config.mode == "qlora-nf4":
        try:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as error:
                raise MATLMRuntimeError("bitsandbytes est absent pour QLoRA NF4") from error
            compute_dtype = torch.bfloat16
            quantization = stack["BitsAndBytesConfig"](
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model = model_class.from_pretrained(
                source,
                quantization_config=quantization,
                torch_dtype=compute_dtype,
                device_map={"": device},
                **common,
            )
            prepare = stack["prepare_model_for_kbit_training"]
            try:
                model = prepare(
                    model,
                    use_gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            except TypeError:
                model = prepare(model, use_gradient_checkpointing=True)
            return model, "qlora-nf4-all-linear", None
        except Exception as error:
            quantization_error = f"{type(error).__name__}: {error}"[-2_000:]
            model = None
            _clear_device(torch, device)
            if config.fallback == "none":
                raise MATLMRuntimeError(
                    "chargement QLoRA NF4 impossible et repli désactivé: " + quantization_error
                ) from error

    precision = config.fallback_precision
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if precision == "bf16" and device.startswith("xpu"):
        check = getattr(torch.xpu, "is_bf16_supported", None)
        if callable(check) and not check():
            raise MATLMRuntimeError("ce périphérique XPU ne déclare pas le support BF16")
    try:
        model = model_class.from_pretrained(
            source,
            torch_dtype=dtype,
            device_map={"": device},
            **common,
        )
    except Exception as error:
        suffix = f"; échec QLoRA initial: {quantization_error}" if quantization_error else ""
        raise MATLMRuntimeError(
            f"chargement LoRA {precision} impossible: {type(error).__name__}: {error}{suffix}"
        ) from error
    return model, f"{precision}-lora-attention", quantization_error


def _apply_lora(stack: Mapping[str, Any], model: Any, config: TrainingConfig, mode: str) -> Any:
    if mode.startswith("qlora"):
        target_modules: str | list[str] = "all-linear"
    else:
        suffixes = {name.rsplit(".", 1)[-1] for name, _module in model.named_modules()}
        missing = [name for name in config.attention_modules if name not in suffixes]
        if missing:
            raise MATLMRuntimeError(
                "modules d'attention absents du modèle: "
                + ", ".join(missing)
                + ". Ajustez --attention-modules."
            )
        target_modules = list(config.attention_modules)
    lora = stack["LoraConfig"](
        task_type="CAUSAL_LM",
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = stack["get_peft_model"](model, lora)
    if hasattr(model, "config"):
        model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0 or total <= 0 or trainable >= total:
        raise MATLMRuntimeError("la sélection LoRA n'a pas isolé un sous-ensemble entraînable")
    return model


def _training_arguments(
    stack: Mapping[str, Any], config: TrainingConfig, output_dir: Path, precision: str, has_eval: bool
) -> Any:
    arguments_class = stack["TrainingArguments"]
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "seed": config.seed,
        "data_seed": config.seed,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_torch",
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "report_to": [],
        "push_to_hub": False,
        "save_safetensors": True,
        "bf16": precision == "bf16",
        "fp16": precision == "fp16",
    }
    parameters = inspect.signature(arguments_class.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[strategy_key] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = config.save_steps
    return arguments_class(**kwargs)


def run_training(
    config: TrainingConfig,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]] | None,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Load one base model, train one adapter, save locally, then release it."""

    if config.allow_model_download:
        # Le transport HTTP direct est plus prévisible sur Windows/Intel que le
        # téléchargeur Xet parallèle, tout en restant un simple téléchargement
        # de fichiers et jamais une API d'inférence.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    stack = _runtime_imports()
    torch = stack["torch"]
    device, device_manifest = _device_details(torch, config)
    stack["set_seed"](config.seed)
    random.seed(config.seed)
    if device.startswith("xpu"):
        torch.xpu.manual_seed_all(config.seed)
        try:
            torch.xpu.reset_peak_memory_stats(config.device_index)
        except Exception:
            pass

    tokenizer = None
    model = None
    trainer = None
    manifest["runtime"] = {"device": device_manifest}
    write_manifest(manifest_path, manifest, replace=True)
    try:
        tokenizer = stack["AutoTokenizer"].from_pretrained(
            _model_source(config),
            use_fast=True,
            **_from_pretrained_options(config),
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise MATLMRuntimeError("le tokenizer ne définit ni jeton PAD ni jeton EOS")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model, effective_mode, quantization_error = _load_model(stack, config, device)
        precision = "bf16" if "bf16" in effective_mode or effective_mode.startswith("qlora") else "fp16"
        model = _apply_lora(stack, model, config, effective_mode)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())

        train_dataset = _ChatDataset(train_rows, tokenizer, config.sequence_length)
        eval_dataset = (
            _ChatDataset(eval_rows, tokenizer, config.sequence_length) if eval_rows else None
        )
        preflight_count = min(16, len(train_dataset))
        for index in range(preflight_count):
            train_dataset[index]
        if eval_dataset is not None:
            for index in range(min(4, len(eval_dataset))):
                eval_dataset[index]

        training_arguments = _training_arguments(
            stack, config, config.output_dir, precision, eval_dataset is not None
        )
        collator = _CausalLMCollator(torch, int(tokenizer.pad_token_id))
        trainer = stack["Trainer"](
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )
        manifest["runtime"].update(
            {
                "effective_mode": effective_mode,
                "effective_precision": precision,
                "quantization_fallback_reason": quantization_error,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "trainable_fraction": trainable / total,
                "assistant_masking_preflight": sorted(train_dataset.masking_modes),
            }
        )
        write_manifest(manifest_path, manifest, replace=True)

        result = trainer.train()
        adapter_dir = config.output_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        if (adapter_dir / "model.safetensors").exists() or (adapter_dir / "pytorch_model.bin").exists():
            raise MATLMRuntimeError("la sortie contient les poids complets au lieu du seul adaptateur")

        if device.startswith("xpu"):
            try:
                torch.xpu.synchronize()
                manifest["runtime"]["xpu_memory"] = {
                    "peak_allocated_bytes": int(
                        torch.xpu.max_memory_allocated(config.device_index)
                    ),
                    "peak_reserved_bytes": int(
                        torch.xpu.max_memory_reserved(config.device_index)
                    ),
                    "allocated_after_save_bytes": int(
                        torch.xpu.memory_allocated(config.device_index)
                    ),
                    "reserved_after_save_bytes": int(
                        torch.xpu.memory_reserved(config.device_index)
                    ),
                }
            except Exception as error:
                manifest["runtime"]["xpu_memory_error"] = (
                    f"{type(error).__name__}: {error}"[-500:]
                )

        manifest = complete_run_manifest(
            manifest,
            adapter_dir=adapter_dir,
            training_metrics=result.metrics,
            evaluation_log_history=(
                getattr(getattr(trainer, "state", None), "log_history", ())
                if eval_dataset is not None
                else None
            ),
            assistant_masking_modes=sorted(train_dataset.masking_modes),
            tokenization_calls_truncated=train_dataset.truncated_examples,
        )
        write_manifest(manifest_path, manifest, replace=True)
        return manifest
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error)[-2_000:],
        }
        write_manifest(manifest_path, manifest, replace=True)
        if isinstance(error, MATLMRuntimeError):
            raise
        raise MATLMRuntimeError(f"entraînement interrompu: {type(error).__name__}: {error}") from error
    finally:
        trainer = None
        model = None
        tokenizer = None
        _clear_device(torch, device)


def _attention_modules(value: str) -> tuple[str, ...]:
    modules = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modules:
        raise argparse.ArgumentTypeError("au moins un module d'attention est requis")
    return modules


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache local des poids; permet de garder le modèle sur D:.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--mode", choices=("qlora-nf4", "bf16-lora"), default="qlora-nf4")
    parser.add_argument(
        "--fallback", choices=("bf16-attention", "none"), default="bf16-attention"
    )
    parser.add_argument("--fallback-precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--attention-modules",
        type=_attention_modules,
        default=DEFAULT_ATTENTION_MODULES,
        help="Suffixes séparés par des virgules pour le repli LoRA attention.",
    )
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Autorise seulement le téléchargement des poids; aucune API d'inférence.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Copie facultative du manifeste; en mode réel, le manifeste principal reste dans output-dir.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        train_jsonl=arguments.train_jsonl,
        eval_jsonl=arguments.eval_jsonl,
        output_dir=arguments.output_dir,
        cache_dir=arguments.cache_dir,
        base_model=arguments.base_model,
        mode=arguments.mode,
        fallback=arguments.fallback,
        fallback_precision=arguments.fallback_precision,
        attention_modules=arguments.attention_modules,
        device=arguments.device,
        device_index=arguments.device_index,
        sequence_length=arguments.sequence_length,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        epochs=arguments.epochs,
        max_steps=arguments.max_steps,
        learning_rate=arguments.learning_rate,
        warmup_ratio=arguments.warmup_ratio,
        weight_decay=arguments.weight_decay,
        lora_rank=arguments.lora_rank,
        lora_alpha=arguments.lora_alpha,
        lora_dropout=arguments.lora_dropout,
        logging_steps=arguments.logging_steps,
        save_steps=arguments.save_steps,
        seed=arguments.seed,
        allow_model_download=arguments.allow_model_download,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = validate_config(_config_from_arguments(arguments))
        train = load_training_jsonl(config.train_jsonl)
        evaluation = load_training_jsonl(config.eval_jsonl) if config.eval_jsonl else None
        inputs = [train.summary.path]
        if evaluation is not None:
            inputs.append(evaluation.summary.path)
        output_dir = validate_output_path(config.output_dir, inputs)
        config = replace(
            config,
            train_jsonl=train.summary.path,
            eval_jsonl=evaluation.summary.path if evaluation else None,
            output_dir=output_dir,
        )
        status = "dry-run" if arguments.dry_run else "planned"
        manifest = build_run_manifest(
            config,
            train.summary,
            evaluation.summary if evaluation else None,
            status=status,
        )
        if arguments.dry_run:
            if arguments.manifest_output is not None:
                write_manifest(arguments.manifest_output, manifest)
            sys.stdout.write(_json_text(manifest) + "\n")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "run-manifest.json"
        write_manifest(manifest_path, manifest)
        completed = run_training(
            config,
            train.rows,
            evaluation.rows if evaluation else None,
            manifest,
            manifest_path,
        )
        if arguments.manifest_output is not None:
            destination = arguments.manifest_output.expanduser().resolve()
            if destination != manifest_path.resolve():
                write_manifest(destination, completed)
        sys.stdout.write(
            _json_text(
                {
                    "status": completed["status"],
                    "manifest": str(manifest_path),
                    "adapter": completed.get("result", {}).get("adapter_dir"),
                }
            )
            + "\n"
        )
        return 0
    except (MATLMTrainingError, MATLMRuntimeError) as error:
        sys.stderr.write(f"Erreur MAT-LM: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
