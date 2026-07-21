"""Inférence locale MAT-LM: Granite + adaptateur PEFT, un modèle à la fois."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Iterator, Mapping, Sequence
import unicodedata

from .matlm_bridge import strict_chat_messages
from .matlm_calculations import MATLMCalculationError, reexecute_matlm_calculations
from .matlm_training import DEFAULT_BASE_MODEL
from .native_llm_contract import (
    ANSWER_JSON_SCHEMA,
    MAX_CAPSULE_BYTES,
    ContractValidationError,
    validate_answer,
    validate_capsule,
)


INFERENCE_STATUS_SCHEMA = "matlm-inference-status-v1"
INFERENCE_PLAN_SCHEMA = "matlm-inference-plan-v1"
_MODEL_REFERENCE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)
_LOAD_MODES = frozenset({"auto", "qlora-nf4", "bf16"})
_PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors")
_MAX_CONFIG_BYTES = 1_000_000
_MAX_GENERATED_CHARACTERS = 65_536
_REPLACEMENT_CHARACTER = "\ufffd"
_MODEL_LOCK = threading.Lock()


class MATLMInferenceError(RuntimeError):
    """L'inférence locale ne peut pas respecter son contrat."""


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    # ``None`` is reserved for an explicit, offline Granite baseline.  The
    # user-facing MAT-LM question CLI still requires an adapter; only the
    # held-out benchmark opts into the baseline deliberately.
    adapter_path: Path | None
    base_model: str = DEFAULT_BASE_MODEL
    load_mode: str = "auto"
    device_index: int = 0
    max_input_tokens: int = 4_096
    max_new_tokens: int = 768
    seed: int = 20_260_721
    allow_model_download: bool = False


@dataclass(slots=True)
class _RuntimeAssets:
    torch: Any
    tokenizer: Any
    model: Any
    device: str
    effective_mode: str
    quantization_fallback_reason: str | None = None
    previous_deterministic: bool | None = None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MATLMInferenceError(f"clé JSON répétée: {key}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise MATLMInferenceError(f"nombre JSON non fini interdit: {value}")


def _read_json_file(path: Path, *, label: str) -> dict[str, Any]:
    try:
        if path.stat().st_size > _MAX_CONFIG_BYTES:
            raise MATLMInferenceError(f"{label} dépasse {_MAX_CONFIG_BYTES} octets")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except MATLMInferenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MATLMInferenceError(f"{label} n'est pas un JSON UTF-8 valide: {path}") from error
    if not isinstance(value, dict):
        raise MATLMInferenceError(f"{label} doit contenir un objet JSON")
    return value


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith((".", "~", "/", "\\"))
        or "\\" in value
        or bool(re.match(r"^[A-Za-z]:", value))
    )


def _granite_local_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise MATLMInferenceError(f"dossier du modèle Granite introuvable: {resolved}")
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise MATLMInferenceError(f"config.json absent du modèle Granite: {resolved}")
    config = _read_json_file(config_path, label="configuration Granite")
    model_type = str(config.get("model_type", "")).casefold()
    architectures = config.get("architectures", [])
    architecture_names = (
        [str(value).casefold() for value in architectures]
        if isinstance(architectures, list)
        else []
    )
    if "granite" not in model_type and not any("granite" in value for value in architecture_names):
        raise MATLMInferenceError("le modèle de base local ne déclare pas une architecture Granite")
    return resolved


def _base_model_reference(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MATLMInferenceError("base_model doit être une référence stable")
    candidate = Path(value).expanduser()
    if candidate.exists() or _looks_like_path(value):
        return str(_granite_local_directory(candidate))
    if not _MODEL_REFERENCE.fullmatch(value) or "granite" not in value.casefold():
        raise MATLMInferenceError(
            "base_model doit être un identifiant Granite ou un dossier Granite local"
        )
    return value


def _adapter_directory(value: Any) -> Path:
    try:
        path = Path(value).expanduser().resolve()
    except TypeError as error:
        raise MATLMInferenceError("adapter_path doit être un chemin local") from error
    if not path.is_dir():
        raise MATLMInferenceError(f"adaptateur PEFT local introuvable: {path}")
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise MATLMInferenceError(f"adapter_config.json absent: {path}")
    config = _read_json_file(config_path, label="configuration PEFT")
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise MATLMInferenceError("l'adaptateur PEFT doit être de type LORA")
    task_type = str(config.get("task_type", "")).upper()
    if task_type and task_type != "CAUSAL_LM":
        raise MATLMInferenceError("l'adaptateur PEFT doit cibler CAUSAL_LM")
    weights = path / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size <= 0:
        raise MATLMInferenceError(
            "adapter_model.safetensors absent ou vide; les poids pickle .bin ne sont pas acceptés"
        )
    return path


def _validate_scalars(config: InferenceConfig) -> None:
    if not isinstance(config, InferenceConfig):
        raise TypeError("config doit être une InferenceConfig")
    if config.load_mode not in _LOAD_MODES:
        raise MATLMInferenceError("load_mode doit valoir auto, qlora-nf4 ou bf16")
    if isinstance(config.device_index, bool) or not 0 <= config.device_index <= 31:
        raise MATLMInferenceError("device_index doit être compris entre 0 et 31")
    if isinstance(config.max_input_tokens, bool) or not 256 <= config.max_input_tokens <= 65_536:
        raise MATLMInferenceError("max_input_tokens doit être compris entre 256 et 65 536")
    if isinstance(config.max_new_tokens, bool) or not 32 <= config.max_new_tokens <= 8_192:
        raise MATLMInferenceError("max_new_tokens doit être compris entre 32 et 8 192")
    if isinstance(config.seed, bool) or not 0 <= config.seed <= 2**32 - 1:
        raise MATLMInferenceError("seed doit être un entier 32 bits non signé")
    if not isinstance(config.allow_model_download, bool):
        raise MATLMInferenceError("allow_model_download doit être un booléen")


def validate_inference_config(config: InferenceConfig) -> InferenceConfig:
    """Valide chemins et bornes sans importer Torch, Transformers ou PEFT."""

    _validate_scalars(config)
    return replace(
        config,
        adapter_path=(
            _adapter_directory(config.adapter_path)
            if config.adapter_path is not None
            else None
        ),
        base_model=_base_model_reference(config.base_model),
    )


def installed_inference_versions() -> dict[str, str | None]:
    """Inspecte les distributions installées sans importer la pile ML."""

    result: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def inference_status(config: InferenceConfig) -> dict[str, Any]:
    """Retourne un état JSON utile même lorsque les dépendances sont absentes."""

    scalar_error = None
    try:
        _validate_scalars(config)
    except (TypeError, MATLMInferenceError) as error:
        scalar_error = str(error)
    versions = installed_inference_versions()
    adapter_error = None
    adapter = None
    if config.adapter_path is not None:
        try:
            adapter = _adapter_directory(config.adapter_path)
        except (TypeError, MATLMInferenceError) as error:
            try:
                adapter = Path(config.adapter_path).expanduser().resolve()
            except TypeError:
                adapter = config.adapter_path
            adapter_error = str(error)
    base_error = None
    try:
        base = _base_model_reference(config.base_model)
    except MATLMInferenceError as error:
        base = str(config.base_model)
        base_error = str(error)
    required = ("torch", "transformers", "accelerate")
    if config.adapter_path is not None:
        required = (*required, "peft")
    if config.load_mode == "qlora-nf4":
        required = (*required, "bitsandbytes")
    missing = [package for package in required if versions.get(package) is None]
    issues = [value for value in (scalar_error, adapter_error, base_error) if value]
    if missing:
        issues.append("dépendances absentes: " + ", ".join(missing))
    return {
        "schema_version": INFERENCE_STATUS_SCHEMA,
        "ready_for_runtime_attempt": not issues,
        "network": "allowed-for-model-files" if config.allow_model_download else "offline",
        "device": {"type": "xpu", "index": config.device_index},
        "model": {
            "base": base,
            "base_resolution": (
                "local-directory"
                if Path(base).is_dir()
                else ("download-allowed" if config.allow_model_download else "local-cache-required")
            ),
            "adapter": str(adapter) if adapter is not None else None,
            "load_mode": config.load_mode,
        },
        "dependencies": versions,
        "issues": issues,
    }


def dry_run_plan(
    config: InferenceConfig, capsule: Mapping[str, Any] | str | bytes
) -> dict[str, Any]:
    """Valide tout ce qui est possible sans charger la pile ML."""

    clean_config = validate_inference_config(config)
    clean_capsule = validate_capsule(capsule)
    return {
        "schema_version": INFERENCE_PLAN_SCHEMA,
        "status": "dry-run",
        "network": "allowed-for-model-files" if clean_config.allow_model_download else "offline",
        "device": {"type": "xpu", "index": clean_config.device_index},
        "model": {
            "base": clean_config.base_model,
            "adapter": (
                str(clean_config.adapter_path)
                if clean_config.adapter_path is not None
                else None
            ),
            "load_mode": clean_config.load_mode,
        },
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "seed": clean_config.seed,
            "max_input_tokens": clean_config.max_input_tokens,
            "max_new_tokens": clean_config.max_new_tokens,
        },
        "capsule": {
            "request_id": clean_capsule["request_id"],
            "evidence_count": len(clean_capsule["evidence"]),
            "characters": len(_canonical_json(clean_capsule)),
        },
        "dependencies": installed_inference_versions(),
    }


def pretrained_options(config: InferenceConfig) -> dict[str, Any]:
    """Options communes qui bloquent le réseau tant qu'il n'est pas autorisé."""

    return {
        "local_files_only": not config.allow_model_download,
        "trust_remote_code": False,
    }


@contextmanager
def _offline_environment(enabled: bool) -> Iterator[None]:
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if enabled:
            for key in keys:
                os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _runtime_imports(*, require_peft: bool) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if require_peft:
            from peft import PeftModel
        else:
            PeftModel = None
    except ImportError as error:
        package = getattr(error, "name", None) or "inconnue"
        raise MATLMInferenceError(
            "dépendance d'inférence absente: "
            f"{package}. Installez la pile locale MAT-LM et PyTorch XPU."
        ) from error
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def _xpu_device(torch: Any, index: int) -> str:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        raise MATLMInferenceError(
            "PyTorch XPU ne détecte pas l'Intel Arc; vérifiez le pilote et la roue XPU."
        )
    count = int(xpu.device_count())
    if index >= count:
        raise MATLMInferenceError(
            f"XPU {index} demandé, mais seulement {count} périphérique(s) détecté(s)"
        )
    xpu.set_device(index)
    check = getattr(xpu, "is_bf16_supported", None)
    if callable(check) and not check():
        raise MATLMInferenceError("ce périphérique XPU ne déclare pas le support BF16")
    return f"xpu:{index}"


def _load_tokenizer(stack: Mapping[str, Any], config: InferenceConfig) -> Any:
    options = pretrained_options(config)
    tokenizer_class = stack["AutoTokenizer"]
    tokenizer = None
    if config.adapter_path is not None:
        try:
            tokenizer = tokenizer_class.from_pretrained(
                str(config.adapter_path),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
        except Exception:
            tokenizer = None
    if tokenizer is None:
        try:
            tokenizer = tokenizer_class.from_pretrained(
                config.base_model,
                use_fast=True,
                **options,
            )
        except Exception as error:
            raise MATLMInferenceError(
                "tokenizer Granite absent ou illisible; utilisez un dossier local valide"
                + (" ou --allow-model-download" if not config.allow_model_download else "")
            ) from error
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise MATLMInferenceError("le tokenizer ne définit ni PAD ni EOS")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_base_model(
    stack: Mapping[str, Any], config: InferenceConfig, device: str
) -> tuple[Any, str, str | None]:
    torch = stack["torch"]
    model_class = stack["AutoModelForCausalLM"]
    common = {
        **pretrained_options(config),
        "low_cpu_mem_usage": True,
        "device_map": {"": device},
        "use_safetensors": True,
    }
    quantization_error = None
    if config.load_mode in {"auto", "qlora-nf4"}:
        try:
            quantization = stack["BitsAndBytesConfig"](
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = model_class.from_pretrained(
                config.base_model,
                quantization_config=quantization,
                torch_dtype=torch.bfloat16,
                **common,
            )
            return model, "qlora-nf4", None
        except Exception as error:
            quantization_error = f"{type(error).__name__}: {error}"[-1_000:]
            if config.load_mode == "qlora-nf4":
                raise MATLMInferenceError(
                    "chargement QLoRA/NF4 impossible: " + quantization_error
                ) from error
            gc.collect()
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass
    try:
        model = model_class.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            **common,
        )
    except Exception as error:
        network_hint = (
            "" if config.allow_model_download else " Modèle absent du cache local; fournissez un dossier local ou --allow-model-download."
        )
        fallback = f" Échec NF4 initial: {quantization_error}." if quantization_error else ""
        raise MATLMInferenceError(
            f"chargement Granite BF16 impossible: {type(error).__name__}: {error}.{fallback}{network_hint}"
        ) from error
    return model, "bf16", quantization_error


def _load_runtime_assets(config: InferenceConfig) -> _RuntimeAssets:
    stack = _runtime_imports(require_peft=config.adapter_path is not None)
    torch = stack["torch"]
    device = _xpu_device(torch, config.device_index)
    tokenizer = None
    base_model = None
    model = None
    try:
        with _offline_environment(not config.allow_model_download):
            tokenizer = _load_tokenizer(stack, config)
            base_model, mode, fallback = _load_base_model(stack, config, device)
            if config.adapter_path is None:
                model = base_model
            else:
                try:
                    model = stack["PeftModel"].from_pretrained(
                        base_model,
                        str(config.adapter_path),
                        is_trainable=False,
                        local_files_only=True,
                    )
                except Exception as error:
                    raise MATLMInferenceError(
                        f"adaptateur PEFT incompatible ou illisible: {type(error).__name__}: {error}"
                    ) from error
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        torch.manual_seed(config.seed)
        torch.xpu.manual_seed_all(config.seed)
        deterministic = getattr(torch, "use_deterministic_algorithms", None)
        deterministic_status = getattr(torch, "are_deterministic_algorithms_enabled", None)
        previous_deterministic = (
            bool(deterministic_status()) if callable(deterministic_status) else None
        )
        if callable(deterministic):
            deterministic(True, warn_only=True)
        return _RuntimeAssets(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            device=device,
            effective_mode=mode,
            quantization_fallback_reason=fallback,
            previous_deterministic=previous_deterministic,
        )
    except Exception:
        model = None
        base_model = None
        tokenizer = None
        gc.collect()
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
        raise


def _release_runtime_assets(assets: _RuntimeAssets) -> None:
    torch = assets.torch
    assets.model = None
    assets.tokenizer = None
    deterministic = getattr(torch, "use_deterministic_algorithms", None)
    if callable(deterministic) and assets.previous_deterministic is not None:
        try:
            deterministic(assets.previous_deterministic, warn_only=True)
        except Exception:
            pass
    gc.collect()
    if getattr(torch, "xpu", None) is not None:
        try:
            torch.xpu.synchronize()
        except Exception:
            pass
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass


def _render_prompt(tokenizer: Any, capsule: Mapping[str, Any]) -> str:
    messages = strict_chat_messages(capsule)
    template = getattr(tokenizer, "apply_chat_template", None)
    if callable(template):
        try:
            rendered = template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as error:
            raise MATLMInferenceError(
                f"le gabarit Granite a refusé le prompt: {type(error).__name__}: {error}"
            ) from error
        if isinstance(rendered, str) and rendered:
            return rendered
    return (
        "<|system|>\n"
        + messages[0]["content"]
        + "\n<|user|>\n"
        + messages[1]["content"]
        + "\n<|assistant|>\n"
    )


def _move_inputs(inputs: Any, device: str) -> Mapping[str, Any]:
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    if not isinstance(inputs, Mapping):
        raise MATLMInferenceError("le tokenizer n'a pas produit un lot de tenseurs")
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _sequence_length(input_ids: Any) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    try:
        first = input_ids[0]
        return len(first)
    except Exception as error:
        raise MATLMInferenceError("impossible de mesurer les jetons du prompt") from error


def _generated_suffix(output_ids: Any, prompt_length: int) -> Any:
    try:
        return output_ids[0][prompt_length:]
    except Exception as error:
        raise MATLMInferenceError("le modèle n'a pas retourné une séquence exploitable") from error


def _generate_text(assets: _RuntimeAssets, capsule: Mapping[str, Any], config: InferenceConfig) -> str:
    torch = assets.torch
    tokenizer = assets.tokenizer
    prompt_text = _render_prompt(tokenizer, capsule)
    try:
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )
    except Exception as error:
        raise MATLMInferenceError(f"tokenisation impossible: {type(error).__name__}: {error}") from error
    inputs = _move_inputs(encoded, assets.device)
    if "input_ids" not in inputs:
        raise MATLMInferenceError("le tokenizer n'a pas retourné input_ids")
    prompt_length = _sequence_length(inputs["input_ids"])
    if prompt_length > config.max_input_tokens:
        raise MATLMInferenceError(
            f"capsule trop grande: {prompt_length} jetons pour une limite de {config.max_input_tokens}"
        )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    try:
        with torch.inference_mode():
            output_ids = assets.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        generated = _generated_suffix(output_ids, prompt_length)
        text = tokenizer.decode(generated, skip_special_tokens=True)
    except Exception as error:
        raise MATLMInferenceError(f"génération locale impossible: {type(error).__name__}: {error}") from error
    if not isinstance(text, str) or not text.strip():
        raise MATLMInferenceError("le modèle a produit une réponse vide")
    if len(text) > _MAX_GENERATED_CHARACTERS:
        raise MATLMInferenceError("la sortie du modèle dépasse la limite de sécurité")
    return text


def extract_json_object(text: Any) -> str:
    """Extrait un unique objet JSON, avec au plus une clôture Markdown simple."""

    if not isinstance(text, str) or not text.strip():
        raise MATLMInferenceError("la sortie du modèle est vide")
    if len(text) > _MAX_GENERATED_CHARACTERS:
        raise MATLMInferenceError("la sortie du modèle dépasse la limite de sécurité")
    stripped = text.strip()
    start = stripped.find("{")
    if start < 0:
        raise MATLMInferenceError("aucun objet JSON trouvé dans la sortie du modèle")
    prefix = stripped[:start].strip()
    if prefix not in {"", "```", "```json", "```JSON"}:
        raise MATLMInferenceError("texte interdit avant l'objet JSON")
    depth = 0
    in_string = False
    escaped = False
    end = None
    for index in range(start, len(stripped)):
        character = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
            if depth < 0:
                raise MATLMInferenceError("accolades JSON incohérentes")
    if end is None or in_string or depth != 0:
        raise MATLMInferenceError("objet JSON incomplet")
    suffix = stripped[end:].strip()
    if suffix not in {"", "```"}:
        raise MATLMInferenceError("texte ou second objet interdit après l'objet JSON")
    return stripped[start:end]


def _lexical_spans(text: str, *, include_replacement: bool) -> Iterator[tuple[int, int]]:
    """Repère les mots Unicode sans traiter la ponctuation comme une lettre perdue."""

    def is_lexical(character: str) -> bool:
        if include_replacement and character == _REPLACEMENT_CHARACTER:
            return True
        # Les catégories L/M/N couvrent lettres, accents combinatoires et nombres.
        return unicodedata.category(character)[:1] in {"L", "M", "N"}

    start: int | None = None
    for index, character in enumerate(text):
        if is_lexical(character):
            if start is None:
                start = index
        elif start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(text)


def _evidence_words(capsule: Mapping[str, Any], evidence_ids: Sequence[str]) -> set[str]:
    cited = set(evidence_ids)
    words: set[str] = set()
    for evidence in capsule["evidence"]:
        if evidence["evidence_id"] not in cited:
            continue
        text = evidence["text"]
        if _REPLACEMENT_CHARACTER in text:
            # Une preuve elle-même corrompue ne peut jamais servir de dictionnaire.
            continue
        for start, end in _lexical_spans(text, include_replacement=False):
            words.add(text[start:end])
    return words


def _repair_corrupted_word(corrupted: str, evidence_words: set[str]) -> str:
    repairs: set[str] = set()
    for candidate in evidence_words:
        if len(candidate) != len(corrupted):
            continue
        if not all(
            (
                observed == _REPLACEMENT_CHARACTER
                and not expected.isascii()
            )
            or observed.casefold() == expected.casefold()
            for observed, expected in zip(corrupted, candidate, strict=True)
        ):
            continue
        # Seuls les emplacements explicitement perdus sont copiés depuis la preuve.
        repairs.add(
            "".join(
                expected if observed == _REPLACEMENT_CHARACTER else observed
                for observed, expected in zip(corrupted, candidate, strict=True)
            )
        )
    if not repairs:
        raise MATLMInferenceError(
            f"caractère de remplacement non ancré dans une preuve citée: {corrupted!r}"
        )
    if len(repairs) != 1:
        raise MATLMInferenceError(
            f"réparation Unicode ambiguë dans les preuves citées: {corrupted!r}"
        )
    return repairs.pop()


def _contains_replacement_character(value: Any) -> bool:
    if isinstance(value, str):
        return _REPLACEMENT_CHARACTER in value
    if isinstance(value, list):
        return any(_contains_replacement_character(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_replacement_character(key) or _contains_replacement_character(item)
            for key, item in value.items()
        )
    return False


def repair_generated_answer_from_evidence(
    answer: Mapping[str, Any] | str | bytes,
    capsule: Mapping[str, Any] | str | bytes,
) -> dict[str, Any]:
    """Répare U+FFFD uniquement par correspondance unique avec une preuve citée.

    La réponse est validée avant et après la réparation. Aucun champ structurel,
    identifiant ou calcul n'est corrigé; un U+FFFD hors du texte de réponse est
    refusé, car aucune substitution lexicale n'y serait suffisamment sûre.
    """

    trusted_capsule = validate_capsule(capsule)
    clean = validate_answer(answer, trusted_capsule)
    text = clean["answer"]
    if _REPLACEMENT_CHARACTER in text:
        words = _evidence_words(trusted_capsule, clean["evidence_ids"])
        pieces: list[str] = []
        cursor = 0
        for start, end in _lexical_spans(text, include_replacement=True):
            token = text[start:end]
            if _REPLACEMENT_CHARACTER not in token:
                continue
            pieces.append(text[cursor:start])
            pieces.append(_repair_corrupted_word(token, words))
            cursor = end
        pieces.append(text[cursor:])
        clean["answer"] = "".join(pieces)
    if _contains_replacement_character(clean):
        raise MATLMInferenceError(
            "caractère de remplacement interdit hors d'un mot réparable de $.answer"
        )
    return validate_answer(clean, trusted_capsule)


def validate_generated_answer(
    generated_text: str,
    capsule: Mapping[str, Any] | str | bytes,
) -> dict[str, Any]:
    """Extrait puis valide la réponse; aucune sortie brute n'est retournée."""

    try:
        return repair_generated_answer_from_evidence(
            extract_json_object(generated_text), capsule
        )
    except ContractValidationError as error:
        raise MATLMInferenceError(f"sortie MAT-LM invalide: {error}") from error


class MATLMInferenceSession:
    """Session explicite qui détient au plus un modèle local XPU."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = validate_inference_config(config)
        self._assets: _RuntimeAssets | None = None
        self._owns_lock = False

    @property
    def loaded(self) -> bool:
        return self._assets is not None

    @property
    def effective_mode(self) -> str | None:
        return self._assets.effective_mode if self._assets is not None else None

    def load(self) -> "MATLMInferenceSession":
        if self.loaded:
            return self
        if not _MODEL_LOCK.acquire(blocking=False):
            raise MATLMInferenceError("un autre modèle MAT-LM est déjà chargé dans ce processus")
        self._owns_lock = True
        try:
            self._assets = _load_runtime_assets(self.config)
            return self
        except Exception:
            self._assets = None
            self._owns_lock = False
            _MODEL_LOCK.release()
            raise

    def ask(self, capsule: Mapping[str, Any] | str | bytes) -> dict[str, Any]:
        clean = validate_capsule(capsule)
        if self._assets is None:
            raise MATLMInferenceError("la session MAT-LM n'est pas chargée")
        generated = _generate_text(self._assets, clean, self.config)
        validated = validate_generated_answer(generated, clean)
        try:
            return reexecute_matlm_calculations(validated, clean)
        except MATLMCalculationError as error:
            raise MATLMInferenceError(
                f"calcul MAT-LM non vérifiable: {' '.join(str(error).split())[:500]}"
            ) from error

    def generate_json(
        self,
        capsule: Mapping[str, Any],
        *,
        output_schema: Mapping[str, Any],
        mode: Any,
    ) -> dict[str, Any]:
        """Implémente le protocole du harnais de comparaison."""

        del mode
        if dict(output_schema) != ANSWER_JSON_SCHEMA:
            raise MATLMInferenceError("le harnais a fourni un schéma de sortie inconnu")
        return self.ask(capsule)

    def close(self) -> None:
        assets, self._assets = self._assets, None
        try:
            if assets is not None:
                _release_runtime_assets(assets)
        finally:
            if self._owns_lock:
                self._owns_lock = False
                _MODEL_LOCK.release()

    def __enter__(self) -> "MATLMInferenceSession":
        return self.load()

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> bool:
        self.close()
        return False


def load_capsule_file(path: str | Path) -> dict[str, Any]:
    """Lit une capsule UTF-8 bornée sans accepter de format annexe."""

    source = Path(path).expanduser().resolve()
    try:
        if not source.is_file():
            raise MATLMInferenceError(f"capsule introuvable: {source}")
        if source.stat().st_size > MAX_CAPSULE_BYTES:
            raise MATLMInferenceError(f"capsule supérieure à {MAX_CAPSULE_BYTES} octets")
        raw = source.read_bytes()
    except MATLMInferenceError:
        raise
    except OSError as error:
        raise MATLMInferenceError(f"impossible de lire la capsule: {source}") from error
    try:
        return validate_capsule(raw)
    except ContractValidationError as error:
        raise MATLMInferenceError(f"capsule native invalide: {error}") from error


__all__ = [
    "INFERENCE_PLAN_SCHEMA",
    "INFERENCE_STATUS_SCHEMA",
    "InferenceConfig",
    "MATLMInferenceError",
    "MATLMInferenceSession",
    "dry_run_plan",
    "extract_json_object",
    "inference_status",
    "installed_inference_versions",
    "load_capsule_file",
    "pretrained_options",
    "repair_generated_answer_from_evidence",
    "validate_generated_answer",
    "validate_inference_config",
]
