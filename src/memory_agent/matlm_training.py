"""Pure-Python configuration and validation for local MAT-LM training.

This module intentionally imports no machine-learning framework.  It is safe
to use it for validating a curriculum or producing a dry-run manifest on a
machine where PyTorch, Transformers and PEFT are not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "matlm-training-run-v1"
EXAMPLE_SCHEMA = "memory-native-sft-example-v1"
DEFAULT_BASE_MODEL = "ibm-granite/granite-3.3-2b-instruct"
DEFAULT_ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

MAX_DATASET_BYTES = 256 * 1024 * 1024
MAX_EXAMPLES = 100_000
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_MESSAGES = 32
MAX_MESSAGE_CHARACTERS = 256_000
MAX_EVALUATION_HISTORY = 64
_MAX_EVALUATION_METRICS_PER_RECORD = 24
_MAX_METRIC_STRING_CHARACTERS = 256

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_FORBIDDEN_EVALUATION_KEYS = frozenset(
    {
        "evaluation_questions",
        "expected_answer_fragments",
        "forbidden_answer_fragments",
        "supporting_claim_ids",
        "answer_status",
        "evaluation_kind",
    }
)
_PACKAGE_NAMES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
_ADAPTER_WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")


class MATLMTrainingError(ValueError):
    """A local training input or configuration is unsafe or invalid."""


@dataclass(frozen=True)
class DatasetSummary:
    """Bounded metadata for a validated JSONL curriculum."""

    path: Path
    sha256: str
    bytes: int
    example_count: int
    example_ids: frozenset[str]
    task_counts: Mapping[str, int]
    provenance_sha256: tuple[str, ...]

    def manifest_value(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "example_count": self.example_count,
            "task_counts": dict(sorted(self.task_counts.items())),
            "provenance_sha256": list(self.provenance_sha256),
        }


@dataclass(frozen=True)
class LoadedCurriculum:
    """Validated rows and their non-sensitive summary."""

    rows: tuple[dict[str, Any], ...]
    summary: DatasetSummary


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible MAT-LM adapter training configuration."""

    train_jsonl: Path
    output_dir: Path
    eval_jsonl: Path | None = None
    cache_dir: Path | None = None
    base_model: str = DEFAULT_BASE_MODEL
    mode: str = "qlora-nf4"
    fallback: str = "bf16-attention"
    fallback_precision: str = "bf16"
    attention_modules: tuple[str, ...] = DEFAULT_ATTENTION_MODULES
    device: str = "xpu"
    device_index: int = 0
    sequence_length: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    logging_steps: int = 5
    save_steps: int = 50
    seed: int = 20_260_721
    allow_model_download: bool = False


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MATLMTrainingError(f"clé JSON répétée: {key}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise MATLMTrainingError(f"nombre JSON non fini interdit: {value}")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_EVALUATION_KEYS:
                return key
            nested = _contains_forbidden_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _contains_forbidden_key(item)
            if nested is not None:
                return nested
    return None


def _clean_text(value: Any, field: str, *, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise MATLMTrainingError(f"{field} doit être une chaîne")
    if "\x00" in value or len(value) > maximum or (not allow_empty and not value.strip()):
        raise MATLMTrainingError(f"{field} est vide, contient NUL ou dépasse {maximum} caractères")
    return value


def _validate_message(message: Any, *, line_number: int, index: int) -> dict[str, Any]:
    field = f"ligne {line_number}, messages[{index}]"
    if not isinstance(message, dict):
        raise MATLMTrainingError(f"{field} doit être un objet")
    role = message.get("role")
    if role not in _ALLOWED_ROLES:
        raise MATLMTrainingError(f"{field}.role est inconnu")
    content = message.get("content", "")
    _clean_text(
        content,
        f"{field}.content",
        maximum=MAX_MESSAGE_CHARACTERS,
        allow_empty=role == "assistant" and bool(message.get("tool_calls")),
    )
    if "tool_calls" in message:
        calls = message["tool_calls"]
        if role != "assistant" or not isinstance(calls, list) or not 1 <= len(calls) <= 8:
            raise MATLMTrainingError(f"{field}.tool_calls doit contenir 1 à 8 appels assistant")
        try:
            _canonical(calls)
        except (TypeError, ValueError) as error:
            raise MATLMTrainingError(f"{field}.tool_calls n'est pas du JSON strict") from error
    return message


def _validate_example(row: Any, *, line_number: int) -> tuple[str, str, str]:
    field = f"ligne {line_number}"
    if not isinstance(row, dict):
        raise MATLMTrainingError(f"{field}: la racine doit être un objet")
    forbidden = _contains_forbidden_key(row)
    if forbidden is not None:
        raise MATLMTrainingError(
            f"{field}: champ d'évaluation interdit dans l'entraînement: {forbidden}"
        )
    required = {
        "schema_version",
        "example_id",
        "task",
        "memory_capsule",
        "messages",
        "target",
        "provenance",
    }
    missing = sorted(required - set(row))
    if missing:
        raise MATLMTrainingError(f"{field}: champs requis absents: {', '.join(missing)}")
    if row["schema_version"] != EXAMPLE_SCHEMA:
        raise MATLMTrainingError(f"{field}.schema_version doit valoir {EXAMPLE_SCHEMA}")
    example_id = _clean_text(row["example_id"], f"{field}.example_id", maximum=256)
    if not _IDENTIFIER.fullmatch(example_id):
        raise MATLMTrainingError(f"{field}.example_id n'est pas un identifiant stable")
    task = _clean_text(row["task"], f"{field}.task", maximum=128)
    if not isinstance(row["memory_capsule"], dict):
        raise MATLMTrainingError(f"{field}.memory_capsule doit être un objet")
    messages = row["messages"]
    if not isinstance(messages, list) or not 2 <= len(messages) <= MAX_MESSAGES:
        raise MATLMTrainingError(f"{field}.messages doit contenir 2 à {MAX_MESSAGES} messages")
    validated_messages = [
        _validate_message(message, line_number=line_number, index=index)
        for index, message in enumerate(messages)
    ]
    final_message = validated_messages[-1]
    if final_message.get("role") != "assistant" or not str(final_message.get("content", "")).strip():
        raise MATLMTrainingError(f"{field}: le dernier message doit être une réponse assistant")
    target = row["target"]
    if not isinstance(target, dict):
        raise MATLMTrainingError(f"{field}.target doit être un objet")
    target_answer = _clean_text(target.get("answer"), f"{field}.target.answer", maximum=MAX_MESSAGE_CHARACTERS)
    if target_answer != final_message["content"]:
        try:
            structured_answer = json.loads(
                final_message["content"],
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite,
            )
        except (json.JSONDecodeError, MATLMTrainingError) as error:
            raise MATLMTrainingError(
                f"{field}: le dernier message ne correspond ni à target.answer ni à target"
            ) from error
        if structured_answer != target:
            raise MATLMTrainingError(
                f"{field}: l'objet du dernier message assistant diffère de target"
            )
    provenance = row["provenance"]
    if not isinstance(provenance, dict):
        raise MATLMTrainingError(f"{field}.provenance doit être un objet")
    provenance_hashes = [
        provenance[key]
        for key in ("facts_sha256", "generator_sha256")
        if key in provenance
    ]
    if (
        len(provenance_hashes) != 1
        or not isinstance(provenance_hashes[0], str)
        or not _SHA256.fullmatch(provenance_hashes[0])
    ):
        raise MATLMTrainingError(
            f"{field}.provenance exige exactement un facts_sha256 ou generator_sha256 valide"
        )
    try:
        _canonical(row)
    except (TypeError, ValueError) as error:
        raise MATLMTrainingError(f"{field} contient une valeur non JSON") from error
    return example_id, task, provenance_hashes[0]


def load_training_jsonl(path: str | Path) -> LoadedCurriculum:
    """Read and strictly validate one bounded UTF-8 JSONL curriculum."""

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise MATLMTrainingError(f"curriculum introuvable: {dataset_path}")
    if dataset_path.suffix.lower() != ".jsonl":
        raise MATLMTrainingError("le curriculum doit avoir l'extension .jsonl")
    size = dataset_path.stat().st_size
    if size <= 0 or size > MAX_DATASET_BYTES:
        raise MATLMTrainingError(
            f"le curriculum doit contenir entre 1 et {MAX_DATASET_BYTES} octets"
        )

    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    task_counts: dict[str, int] = {}
    provenance_hashes: set[str] = set()
    try:
        with dataset_path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                if line_number > MAX_EXAMPLES:
                    raise MATLMTrainingError(f"le curriculum dépasse {MAX_EXAMPLES} exemples")
                if len(raw_line) > MAX_LINE_BYTES:
                    raise MATLMTrainingError(
                        f"ligne {line_number}: dépasse {MAX_LINE_BYTES} octets"
                    )
                if not raw_line.strip():
                    raise MATLMTrainingError(f"ligne {line_number}: ligne vide interdite")
                try:
                    text = raw_line.decode("utf-8", errors="strict")
                    row = json.loads(
                        text,
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=_reject_non_finite,
                    )
                except UnicodeDecodeError as error:
                    raise MATLMTrainingError(
                        f"ligne {line_number}: encodage UTF-8 invalide"
                    ) from error
                except json.JSONDecodeError as error:
                    raise MATLMTrainingError(f"ligne {line_number}: JSON invalide") from error
                example_id, task, provenance_sha256 = _validate_example(
                    row, line_number=line_number
                )
                if example_id in identifiers:
                    raise MATLMTrainingError(f"example_id répété: {example_id}")
                identifiers.add(example_id)
                task_counts[task] = task_counts.get(task, 0) + 1
                provenance_hashes.add(provenance_sha256)
                rows.append(row)
    except OSError as error:
        raise MATLMTrainingError(f"impossible de lire le curriculum: {dataset_path}") from error
    if not rows:
        raise MATLMTrainingError("le curriculum est vide")
    return LoadedCurriculum(
        rows=tuple(rows),
        summary=DatasetSummary(
            path=dataset_path,
            sha256=digest.hexdigest(),
            bytes=size,
            example_count=len(rows),
            example_ids=frozenset(identifiers),
            task_counts=dict(sorted(task_counts.items())),
            provenance_sha256=tuple(sorted(provenance_hashes)),
        ),
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_path(output_dir: str | Path, input_paths: Sequence[Path]) -> Path:
    """Reject broad, ambiguous or already-populated training destinations."""

    output = Path(output_dir).expanduser().resolve()
    if len(str(output)) > 1_024:
        raise MATLMTrainingError("le chemin de sortie dépasse 1024 caractères")
    anchor = Path(output.anchor).resolve()
    protected = {anchor, Path.home().resolve()}
    if output in protected:
        raise MATLMTrainingError("la sortie ne peut pas être une racine ou le dossier personnel")
    for input_path in input_paths:
        source = input_path.resolve()
        if output == source or _is_relative_to(source, output):
            raise MATLMTrainingError("la sortie ne peut pas contenir un curriculum source")
    if output.exists():
        if not output.is_dir():
            raise MATLMTrainingError(f"la sortie existe et n'est pas un dossier: {output}")
        try:
            if next(output.iterdir(), None) is not None:
                raise MATLMTrainingError(f"la sortie doit être vide pour éviter un écrasement: {output}")
        except OSError as error:
            raise MATLMTrainingError(f"impossible d'inspecter la sortie: {output}") from error
    return output


def _validate_base_model(value: str) -> str:
    model = value.strip()
    if not model or "://" in model:
        raise MATLMTrainingError("base_model doit être un identifiant Hugging Face ou un dossier local")
    candidate = Path(model).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise MATLMTrainingError("le chemin du modèle de base doit être un dossier")
        return str(candidate.resolve())
    if not _MODEL_ID.fullmatch(model):
        raise MATLMTrainingError("identifiant du modèle de base invalide")
    return model


def _validate_cache_dir(value: Path | None) -> Path | None:
    if value is None:
        return None
    cache = Path(value).expanduser().resolve()
    if len(str(cache)) > 1_024:
        raise MATLMTrainingError("le chemin du cache dépasse 1024 caractères")
    if cache in {Path(cache.anchor).resolve(), Path.home().resolve()}:
        raise MATLMTrainingError("le cache ne peut pas être une racine ou le dossier personnel")
    if cache.exists() and not cache.is_dir():
        raise MATLMTrainingError("le cache doit être un dossier")
    return cache


def validate_config(config: TrainingConfig) -> TrainingConfig:
    """Validate all hyperparameters before any ML dependency is imported."""

    _validate_base_model(config.base_model)
    _validate_cache_dir(config.cache_dir)
    if config.mode not in {"qlora-nf4", "bf16-lora"}:
        raise MATLMTrainingError("mode doit valoir qlora-nf4 ou bf16-lora")
    if config.fallback not in {"bf16-attention", "none"}:
        raise MATLMTrainingError("fallback doit valoir bf16-attention ou none")
    if config.fallback_precision not in {"bf16", "fp16"}:
        raise MATLMTrainingError("fallback_precision doit valoir bf16 ou fp16")
    if config.device not in {"xpu", "cpu"}:
        raise MATLMTrainingError("device doit valoir xpu ou cpu")
    if not 0 <= config.device_index <= 31:
        raise MATLMTrainingError("device_index doit être compris entre 0 et 31")
    if not 128 <= config.sequence_length <= 4_096:
        raise MATLMTrainingError("sequence_length doit être compris entre 128 et 4096")
    if config.batch_size != 1:
        raise MATLMTrainingError("batch_size est fixé à 1 pour borner la mémoire vidéo")
    if not 1 <= config.gradient_accumulation_steps <= 1_024:
        raise MATLMTrainingError("gradient_accumulation_steps doit être compris entre 1 et 1024")
    if not 0 < config.epochs <= 100:
        raise MATLMTrainingError("epochs doit être compris entre 0 et 100")
    if config.max_steps == 0 or config.max_steps < -1 or config.max_steps > 10_000_000:
        raise MATLMTrainingError("max_steps doit valoir -1 ou un entier positif borné")
    if not 0 < config.learning_rate <= 0.1:
        raise MATLMTrainingError("learning_rate doit être positif et inférieur ou égal à 0.1")
    if not 0 <= config.warmup_ratio < 1:
        raise MATLMTrainingError("warmup_ratio doit être compris entre 0 inclus et 1 exclu")
    if not 0 <= config.weight_decay <= 1:
        raise MATLMTrainingError("weight_decay doit être compris entre 0 et 1")
    if not 1 <= config.lora_rank <= 512:
        raise MATLMTrainingError("lora_rank doit être compris entre 1 et 512")
    if not 1 <= config.lora_alpha <= 2_048:
        raise MATLMTrainingError("lora_alpha doit être compris entre 1 et 2048")
    if not 0 <= config.lora_dropout < 1:
        raise MATLMTrainingError("lora_dropout doit être compris entre 0 inclus et 1 exclu")
    if not 1 <= config.logging_steps <= 100_000 or not 1 <= config.save_steps <= 1_000_000:
        raise MATLMTrainingError("logging_steps et save_steps doivent être positifs et bornés")
    if not 0 <= config.seed <= 2**32 - 1:
        raise MATLMTrainingError("seed doit être un entier 32 bits non signé")
    if not config.attention_modules or len(config.attention_modules) > 32:
        raise MATLMTrainingError("attention_modules doit contenir entre 1 et 32 noms")
    for module in config.attention_modules:
        if not _IDENTIFIER.fullmatch(module):
            raise MATLMTrainingError(f"nom de module LoRA invalide: {module}")
    return config


def installed_training_versions() -> dict[str, str | None]:
    """Report package versions without importing their runtime modules."""

    result: dict[str, str | None] = {}
    for package in _PACKAGE_NAMES:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _config_manifest_value(config: TrainingConfig) -> dict[str, Any]:
    return {
        "base_model": _validate_base_model(config.base_model),
        "cache_dir": (
            str(_validate_cache_dir(config.cache_dir)) if config.cache_dir is not None else None
        ),
        "mode": config.mode,
        "fallback": config.fallback,
        "fallback_precision": config.fallback_precision,
        "attention_modules": list(config.attention_modules),
        "device": config.device,
        "device_index": config.device_index,
        "sequence_length": config.sequence_length,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "epochs": config.epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "bias": "none",
        },
        "gradient_checkpointing": True,
        "optimizer": "adamw_torch",
        "assistant_token_masking": "template-mask-then-final-assistant-fallback",
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "seed": config.seed,
        "allow_model_download": config.allow_model_download,
        "push_to_hub": False,
        "report_to": [],
    }


def build_run_manifest(
    config: TrainingConfig,
    train: DatasetSummary,
    evaluation: DatasetSummary | None,
    *,
    status: str,
) -> dict[str, Any]:
    """Build a reproducibility manifest with no examples or answer keys."""

    validate_config(config)
    if evaluation is not None:
        overlap = sorted(train.example_ids & evaluation.example_ids)
        if overlap:
            preview = ", ".join(overlap[:5])
            raise MATLMTrainingError(f"fuite train/eval: example_id commun: {preview}")
        if train.sha256 == evaluation.sha256:
            raise MATLMTrainingError("fuite train/eval: les deux fichiers sont identiques")
    reproducible = {
        "model": _config_manifest_value(config),
        "data": {
            "train": train.manifest_value(),
            "evaluation": evaluation.manifest_value() if evaluation else None,
            "overlap_checked": evaluation is not None,
        },
        "output_dir": str(config.output_dir.resolve()),
    }
    fingerprint = hashlib.sha256(_canonical(reproducible).encode("utf-8")).hexdigest()
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_sha256": fingerprint,
        **reproducible,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": installed_training_versions(),
        },
        "safety": {
            "external_inference_api": False,
            "push_to_hub": False,
            "trust_remote_code": False,
            "one_model_loaded_at_a_time": True,
            "benchmark_fields_forbidden_in_training": sorted(_FORBIDDEN_EVALUATION_KEYS),
        },
    }


def _json_metric_value(value: Any) -> str | int | float | bool | None:
    """Convert a Trainer scalar to a finite, bounded JSON manifest value."""

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError, RuntimeError):
            value = str(value)
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if value == value and abs(value) != float("inf"):
            return value
        return str(value)
    if isinstance(value, str):
        return value[:_MAX_METRIC_STRING_CHARACTERS]
    return str(value)[:_MAX_METRIC_STRING_CHARACTERS]


def json_manifest_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded JSON-safe scalar metrics without retaining model objects."""

    result: dict[str, Any] = {}
    for raw_key, value in metrics.items():
        key = str(raw_key)[:128]
        if key in result or len(result) >= 128:
            continue
        result[key] = _json_metric_value(value)
    return result


def summarize_evaluation_history(
    log_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extract a bounded evaluation trail from ``Trainer.state.log_history``.

    Training-only log rows are ignored.  Keeping the tail guarantees that the
    final evaluation is retained even for a very long run while preventing the
    reproducibility manifest from growing without bound.
    """

    history: deque[dict[str, Any]] = deque(maxlen=MAX_EVALUATION_HISTORY)
    record_count = 0
    for raw_record in log_history:
        if not isinstance(raw_record, Mapping):
            continue
        has_evaluation_metric = any(str(key).startswith("eval_") for key in raw_record)
        if not has_evaluation_metric:
            continue
        record: dict[str, Any] = {
            key: _json_metric_value(raw_record[key])
            for key in ("step", "epoch")
            if key in raw_record
        }
        for raw_key, value in raw_record.items():
            key = str(raw_key)
            if not key.startswith("eval_"):
                continue
            if len(key) > 128 or len(record) >= _MAX_EVALUATION_METRICS_PER_RECORD:
                continue
            record[key] = _json_metric_value(value)
        if not any(key.startswith("eval_") for key in record):
            continue
        history.append(record)
        record_count += 1

    retained = list(history)
    return {
        "latest": dict(retained[-1]) if retained else None,
        "history": retained,
        "record_count": record_count,
        "history_limit": MAX_EVALUATION_HISTORY,
        "history_truncated": record_count > len(retained),
    }


def complete_run_manifest(
    manifest: Mapping[str, Any],
    *,
    adapter_dir: str | Path,
    training_metrics: Mapping[str, Any],
    evaluation_log_history: Sequence[Mapping[str, Any]] | None,
    assistant_masking_modes: Sequence[str],
    tokenization_calls_truncated: int,
) -> dict[str, Any]:
    """Build a completed manifest only after a saved LoRA adapter is verified."""

    adapter = Path(adapter_dir).expanduser().resolve()
    configuration = adapter / "adapter_config.json"
    weights = next(
        (adapter / name for name in _ADAPTER_WEIGHT_FILENAMES if (adapter / name).is_file()),
        None,
    )
    if not adapter.is_dir() or not configuration.is_file() or weights is None:
        raise MATLMTrainingError(
            "l'adaptateur n'est pas sauvegardé complètement; état complete interdit"
        )
    if configuration.stat().st_size <= 0 or weights.stat().st_size <= 0:
        raise MATLMTrainingError(
            "les fichiers de l'adaptateur sont vides; état complete interdit"
        )
    if not isinstance(tokenization_calls_truncated, int) or tokenization_calls_truncated < 0:
        raise MATLMTrainingError("compteur de troncature invalide")

    completed = dict(manifest)
    completed["status"] = "complete"
    completed["result"] = {
        "adapter_dir": str(adapter),
        "adapter_artifacts": {
            "configuration": configuration.name,
            "weights": weights.name,
            "weights_bytes": weights.stat().st_size,
        },
        "metrics": json_manifest_metrics(training_metrics),
        "evaluation": (
            summarize_evaluation_history(evaluation_log_history)
            if evaluation_log_history is not None
            else None
        ),
        "assistant_masking_modes": sorted({str(mode)[:128] for mode in assistant_masking_modes}),
        "tokenization_calls_truncated": tokenization_calls_truncated,
    }
    return completed


def write_manifest(path: str | Path, manifest: Mapping[str, Any], *, replace: bool = False) -> Path:
    """Atomically write a bounded JSON manifest."""

    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".json":
        raise MATLMTrainingError("le manifeste doit avoir l'extension .json")
    if output.exists() and not replace:
        raise MATLMTrainingError(f"le manifeste existe déjà: {output}")
    encoded = json.dumps(
        dict(manifest), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    if len(encoded.encode("utf-8")) > 1_000_000:
        raise MATLMTrainingError("le manifeste dépasse 1 000 000 octets")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    temporary.replace(output)
    return output


__all__ = [
    "DEFAULT_ATTENTION_MODULES",
    "DEFAULT_BASE_MODEL",
    "DatasetSummary",
    "LoadedCurriculum",
    "MANIFEST_SCHEMA",
    "MAX_EVALUATION_HISTORY",
    "MATLMTrainingError",
    "TrainingConfig",
    "build_run_manifest",
    "complete_run_manifest",
    "installed_training_versions",
    "json_manifest_metrics",
    "load_training_jsonl",
    "validate_config",
    "validate_output_path",
    "summarize_evaluation_history",
    "write_manifest",
]
