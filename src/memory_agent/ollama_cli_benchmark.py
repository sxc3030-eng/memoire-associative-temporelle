"""Benchmark held-out via l'executable Ollama local, sans client HTTP.

Le harnais ne telecharge jamais de modele, lance un seul tag Ollama et ne
conserve ni les capsules, ni les cibles, ni les sorties brutes dans le rapport.
Le score de contenu est calcule separement du respect strict du contrat JSON :
un objet JSON qui contient la bonne reponse mais un champ superflu reste donc
invalide contractuellement sans etre compte comme une erreur de contenu.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping, Protocol, Sequence
import unicodedata

from .matlm_bridge import strict_chat_messages
from .matlm_heldout_benchmark import (
    HeldoutCase,
    HeldoutSelection,
    normalize_exact_answer,
)
from .matlm_inference import MATLMInferenceError, extract_json_object
from .memory_native_curriculum import SYNTHETIC_TASK_ORDER
from .native_llm_contract import ContractValidationError, validate_answer


REPORT_SCHEMA_VERSION = "ollama-cli-heldout-benchmark-v2"
PLAN_SCHEMA_VERSION = "ollama-cli-heldout-plan-v1"
DEFAULT_MODEL = "qwen2.5:14b-instruct-q4_0"
MAX_PROMPT_BYTES = 1_000_000
MAX_STDOUT_BYTES = 262_144
MAX_STDERR_BYTES = 65_536
MAX_TIMEOUT_SECONDS = 86_400.0
MAX_TOTAL_TIMEOUT_SECONDS = 604_800.0
MAX_ERROR_CHARACTERS = 300
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]{0,255}$")
_MODEL_MANIFEST_ID = re.compile(r"^[0-9a-fA-F]{12}$")
_SYNTHETIC_ID = re.compile(r"(?<![A-Za-z0-9])SYN-[A-Za-z0-9-]+", re.IGNORECASE)
_FICTIONAL_VALUE = re.compile(
    r"(?<![\w-])(?:code|état)-fictif-\d+",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_AGE = re.compile(r"(?<!\w)\d{1,3}\s+ans\b", re.IGNORECASE)
_UNKNOWN = re.compile(r"(?<!\w)JE_NE_SAIS_PAS(?!\w)", re.IGNORECASE)
_LEADING_DECISION = re.compile(r"^\s*(oui|non)(?=\s|[.!?:;,])", re.IGNORECASE)


class OllamaCLIBenchmarkError(RuntimeError):
    """Le benchmark Ollama local ne peut pas continuer de facon fiable."""


class OllamaCLITimeoutError(OllamaCLIBenchmarkError):
    """Une commande Ollama a depasse sa limite de temps."""


class OllamaCLIOutputLimitError(OllamaCLIBenchmarkError):
    """Une commande Ollama a depasse sa limite de sortie."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    elapsed_ms: float


class CommandRunner(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        input_text: str | None,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> CommandResult:
        """Execute une commande locale sans shell et avec des sorties bornees."""


@dataclass(frozen=True, slots=True)
class OllamaCLIConfig:
    model: str = DEFAULT_MODEL
    executable: str = "ollama"
    expected_manifest_id: str | None = None
    case_timeout_seconds: float = 180.0
    preflight_timeout_seconds: float = 30.0
    stop_timeout_seconds: float = 30.0
    total_timeout_seconds: float = 14_400.0
    max_prompt_bytes: int = MAX_PROMPT_BYTES
    max_stdout_bytes: int = MAX_STDOUT_BYTES
    max_stderr_bytes: int = MAX_STDERR_BYTES


def _validate_number(
    value: Any,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OllamaCLIBenchmarkError(f"{label} doit etre un nombre")
    clean = float(value)
    if not math.isfinite(clean) or not minimum <= clean <= maximum:
        raise OllamaCLIBenchmarkError(
            f"{label} doit etre compris entre {minimum:g} et {maximum:g}"
        )
    return clean


def _validate_byte_limit(value: Any, *, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OllamaCLIBenchmarkError(f"{label} doit etre un entier")
    if not 1_024 <= value <= maximum:
        raise OllamaCLIBenchmarkError(
            f"{label} doit etre compris entre 1024 et {maximum}"
        )
    return value


def _validate_model_name(value: Any) -> str:
    if not isinstance(value, str) or not _MODEL_NAME.fullmatch(value):
        raise OllamaCLIBenchmarkError(
            "model doit etre un tag Ollama local stable, sans espace ni option"
        )
    return value


def _validate_expected_manifest_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _MODEL_MANIFEST_ID.fullmatch(value):
        raise OllamaCLIBenchmarkError(
            "expected_manifest_id doit contenir 12 caracteres hexadecimaux"
        )
    return value.lower()


def _validate_executable_label(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise OllamaCLIBenchmarkError("executable doit etre un nom ou chemin local")
    if value != value.strip():
        raise OllamaCLIBenchmarkError("executable ne doit pas contenir d'espaces externes")
    return value


def validate_ollama_cli_config(
    config: OllamaCLIConfig,
    *,
    resolve_executable: bool = False,
) -> OllamaCLIConfig:
    """Valide les bornes; le dry-run peut eviter toute recherche d'executable."""

    if not isinstance(config, OllamaCLIConfig):
        raise TypeError("config doit etre une OllamaCLIConfig")
    executable = _validate_executable_label(config.executable)
    if resolve_executable:
        candidate = Path(executable).expanduser()
        has_path_marker = candidate.is_absolute() or any(
            marker in executable for marker in ("/", "\\")
        )
        if has_path_marker:
            resolved = candidate.resolve()
            if not resolved.is_file():
                raise OllamaCLIBenchmarkError("executable Ollama local introuvable")
            executable = str(resolved)
        else:
            found = shutil.which(executable)
            if found is None:
                raise OllamaCLIBenchmarkError("executable Ollama absent du PATH local")
            executable = found
    return replace(
        config,
        model=_validate_model_name(config.model),
        executable=executable,
        expected_manifest_id=_validate_expected_manifest_id(
            config.expected_manifest_id
        ),
        case_timeout_seconds=_validate_number(
            config.case_timeout_seconds,
            label="case_timeout_seconds",
            minimum=1.0,
            maximum=MAX_TIMEOUT_SECONDS,
        ),
        preflight_timeout_seconds=_validate_number(
            config.preflight_timeout_seconds,
            label="preflight_timeout_seconds",
            minimum=1.0,
            maximum=MAX_TIMEOUT_SECONDS,
        ),
        stop_timeout_seconds=_validate_number(
            config.stop_timeout_seconds,
            label="stop_timeout_seconds",
            minimum=1.0,
            maximum=MAX_TIMEOUT_SECONDS,
        ),
        total_timeout_seconds=_validate_number(
            config.total_timeout_seconds,
            label="total_timeout_seconds",
            minimum=1.0,
            maximum=MAX_TOTAL_TIMEOUT_SECONDS,
        ),
        max_prompt_bytes=_validate_byte_limit(
            config.max_prompt_bytes,
            label="max_prompt_bytes",
            maximum=MAX_PROMPT_BYTES,
        ),
        max_stdout_bytes=_validate_byte_limit(
            config.max_stdout_bytes,
            label="max_stdout_bytes",
            maximum=MAX_STDOUT_BYTES,
        ),
        max_stderr_bytes=_validate_byte_limit(
            config.max_stderr_bytes,
            label="max_stderr_bytes",
            maximum=MAX_STDERR_BYTES,
        ),
    )


def _offline_cli_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OLLAMA_NO_CLOUD": "1",
            "OLLAMA_NOHISTORY": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
    )
    return environment


def _kill_quietly(process: subprocess.Popen[bytes]) -> None:
    try:
        process.kill()
    except OSError:
        pass


class SubprocessCommandRunner:
    """Sous-processus sans shell, avec lecteurs bornes et fenetre cachee."""

    def run(
        self,
        command: Sequence[str],
        *,
        input_text: str | None,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> CommandResult:
        clean_command = tuple(str(part) for part in command)
        if not clean_command or any("\x00" in part for part in clean_command):
            raise OllamaCLIBenchmarkError("commande Ollama locale invalide")
        creation_flags = (
            int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        )
        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                clean_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                env=_offline_cli_environment(),
                creationflags=creation_flags,
            )
        except OSError as error:
            raise OllamaCLIBenchmarkError(
                "impossible de demarrer l'executable Ollama local"
            ) from error

        collected: dict[str, bytes] = {"stdout": b"", "stderr": b""}
        overflow: set[str] = set()

        def read_stream(name: str, stream: Any, limit: int) -> None:
            data = bytearray()
            try:
                while True:
                    chunk = stream.read(8_192)
                    if not chunk:
                        break
                    if len(data) < limit + 1:
                        data.extend(chunk[: limit + 1 - len(data)])
                    if len(data) > limit:
                        overflow.add(name)
                        _kill_quietly(process)
            finally:
                collected[name] = bytes(data[:limit])
                try:
                    stream.close()
                except OSError:
                    pass

        assert process.stdout is not None
        assert process.stderr is not None
        readers = (
            threading.Thread(
                target=read_stream,
                args=("stdout", process.stdout, stdout_limit),
                daemon=True,
            ),
            threading.Thread(
                target=read_stream,
                args=("stderr", process.stderr, stderr_limit),
                daemon=True,
            ),
        )
        for reader in readers:
            reader.start()

        try:
            assert process.stdin is not None
            if input_text is not None:
                process.stdin.write(input_text.encode("utf-8"))
            process.stdin.close()
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as error:
                _kill_quietly(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _kill_quietly(process)
                raise OllamaCLITimeoutError("commande Ollama locale expiree") from error
        except BrokenPipeError:
            try:
                return_code = process.wait(timeout=min(timeout_seconds, 5.0))
            except subprocess.TimeoutExpired as error:
                _kill_quietly(process)
                raise OllamaCLITimeoutError("commande Ollama locale expiree") from error
        finally:
            for reader in readers:
                reader.join(timeout=5)
            if process.poll() is None:
                _kill_quietly(process)

        if overflow:
            streams = ",".join(sorted(overflow))
            raise OllamaCLIOutputLimitError(
                f"sortie Ollama trop grande ({streams})"
            )
        return CommandResult(
            returncode=int(return_code),
            stdout=collected["stdout"].decode("utf-8", errors="replace"),
            elapsed_ms=round((time.perf_counter() - started) * 1_000, 3),
        )


def _answer_anchors(value: Any) -> frozenset[str]:
    if not isinstance(value, str):
        return frozenset()
    normalized = unicodedata.normalize("NFKC", value)
    anchors: set[str] = set()
    anchors.update(
        f"id:{match.group(0).upper()}"
        for match in _SYNTHETIC_ID.finditer(normalized)
    )
    anchors.update(
        f"value:{match.group(0).casefold()}"
        for match in _FICTIONAL_VALUE.finditer(normalized)
    )
    anchors.update(f"date:{match.group(0)}" for match in _ISO_DATE.finditer(normalized))
    anchors.update(
        f"age:{' '.join(match.group(0).casefold().split())}"
        for match in _AGE.finditer(normalized)
    )
    if _UNKNOWN.search(normalized):
        anchors.add("unknown:JE_NE_SAIS_PAS")
    decision = _LEADING_DECISION.search(normalized)
    if decision:
        anchors.add(f"decision:{decision.group(1).casefold()}")
    return frozenset(anchors)


def answer_anchor_metrics(prediction: Any, target: Any) -> dict[str, Any]:
    """Rappel d'ancres factuelles, sans retourner les ancres elles-memes."""

    target_anchors = _answer_anchors(target)
    if not target_anchors:
        return {"answer_anchor_recall": None, "answer_anchors_all": None}
    predicted_anchors = _answer_anchors(prediction)
    recall = len(target_anchors & predicted_anchors) / len(target_anchors)
    return {
        "answer_anchor_recall": round(recall, 6),
        "answer_anchors_all": recall == 1.0,
    }


def render_ollama_prompt(case: HeldoutCase) -> str:
    """Rend le meme message systeme/capsule que MAT-LM, sans inclure la cible."""

    messages = strict_chat_messages(case.capsule)
    rendered = (
        "SYSTEM_MESSAGE\n"
        + messages[0]["content"]
        + "\nEND_SYSTEM_MESSAGE\nUSER_MESSAGE\n"
        + messages[1]["content"]
        + "\nEND_USER_MESSAGE\nASSISTANT_JSON\n"
    )
    return rendered


def _safe_error(error: BaseException) -> dict[str, str]:
    kind = re.sub(r"[^A-Za-z0-9_.-]", "", type(error).__name__)[:80] or "Error"
    if isinstance(error, OllamaCLITimeoutError):
        message = "delai de generation depasse"
    elif isinstance(error, OllamaCLIOutputLimitError):
        message = "sortie de generation trop grande"
    elif isinstance(error, MATLMInferenceError):
        message = "sortie incompatible avec le contrat JSON"
    else:
        message = "echec de la commande locale"
    return {"type": kind, "message": message[:MAX_ERROR_CHARACTERS]}


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate key")
        output[key] = value
    return output


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non finite: {value}")


def _first_json_mapping(text: str) -> Mapping[str, Any] | None:
    """Recupere un objet pour le score de contenu, sans le rendre contractuel."""

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
    starts = (index for index, character in enumerate(text) if character == "{")
    for attempt, start in enumerate(starts):
        if attempt >= 32:
            break
        try:
            value, _ = decoder.raw_decode(text[start:])
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping) and any(
            key in value for key in ("answer", "request_id", "evidence_ids")
        ):
            return value
    return None


def _validate_raw_generated_answer(
    generated_text: str,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    """Valide la sortie brute sans appliquer la réparation Unicode de l'UI."""

    try:
        return validate_answer(extract_json_object(generated_text), capsule)
    except ContractValidationError as error:
        raise MATLMInferenceError(f"sortie Ollama invalide: {error}") from error


_CONTENT_METRICS = (
    "answer_exact_normalized",
    "answer_anchors_all",
    "evidence_ids_exact",
    "abstention_exact",
    "calculations_exact",
    "content_core_exact",
    "content_all_exact",
)


def _content_score(
    output: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    case: HeldoutCase,
) -> dict[str, Any]:
    if output is None:
        return {
            **{metric: False for metric in _CONTENT_METRICS},
            "answer_anchor_recall": None,
            "request_id_correct": False,
            "invented_evidence_count": None,
            "evidence_precision": None,
            "evidence_recall": None,
            "evidence_f1": None,
            "prediction_answer_sha256": None,
        }

    answer = output.get("answer")
    answer_exact = isinstance(answer, str) and (
        normalize_exact_answer(answer) == normalize_exact_answer(target["answer"])
    )
    anchor_metrics = answer_anchor_metrics(answer, target["answer"])
    predicted_ids = output.get("evidence_ids")
    valid_predicted_ids = (
        isinstance(predicted_ids, list)
        and all(isinstance(value, str) for value in predicted_ids)
        and len(set(predicted_ids)) == len(predicted_ids)
    )
    target_ids = list(target["evidence_ids"])
    evidence_exact = valid_predicted_ids and predicted_ids == target_ids
    abstention_exact = (
        isinstance(output.get("abstention"), Mapping)
        and dict(output["abstention"]) == target["abstention"]
    )
    calculations_exact = (
        isinstance(output.get("calculations"), list)
        and output["calculations"] == target["calculations"]
    )
    request_id_correct = output.get("request_id") == case.capsule["request_id"]

    if valid_predicted_ids:
        predicted_set = set(predicted_ids)
        target_set = set(target_ids)
        permitted_set = {
            row["evidence_id"] for row in case.capsule["evidence"]
        }
        intersection = len(predicted_set & target_set)
        precision = intersection / len(predicted_set) if predicted_set else float(not target_set)
        recall = intersection / len(target_set) if target_set else float(not predicted_set)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        invented = len(predicted_set - permitted_set)
    else:
        precision = recall = f1 = None
        invented = None

    core_exact = answer_exact and evidence_exact and abstention_exact
    answer_hash = (
        hashlib.sha256(normalize_exact_answer(answer).encode("utf-8")).hexdigest()
        if isinstance(answer, str)
        else None
    )
    return {
        "answer_exact_normalized": answer_exact,
        **anchor_metrics,
        "evidence_ids_exact": evidence_exact,
        "abstention_exact": abstention_exact,
        "calculations_exact": calculations_exact,
        "content_core_exact": core_exact,
        "content_all_exact": core_exact and calculations_exact,
        "request_id_correct": request_id_correct,
        "invented_evidence_count": invented,
        "evidence_precision": round(precision, 6) if precision is not None else None,
        "evidence_recall": round(recall, 6) if recall is not None else None,
        "evidence_f1": round(f1, 6) if f1 is not None else None,
        "prediction_answer_sha256": answer_hash,
    }


def _new_counter() -> dict[str, Any]:
    return {
        "attempted": 0,
        "contract_valid": 0,
        "invalid_contract": 0,
        "execution_error": 0,
        **{metric: 0 for metric in _CONTENT_METRICS},
        "request_id_correct": 0,
        "invented_evidence_ids": 0,
        "evidence_scored": 0,
        "evidence_precision_sum": 0.0,
        "evidence_recall_sum": 0.0,
        "evidence_f1_sum": 0.0,
        "answer_anchor_scored": 0,
        "answer_anchor_recall_sum": 0.0,
    }


def _record(counter: dict[str, Any], result: Mapping[str, Any]) -> None:
    counter["attempted"] += 1
    counter["contract_valid"] += int(bool(result["contract_valid"]))
    counter["invalid_contract"] += int(result["status"] == "invalid_contract")
    counter["execution_error"] += int(result["status"] == "execution_error")
    content = result["content_metrics"]
    for metric in _CONTENT_METRICS:
        counter[metric] += int(bool(content[metric]))
    counter["request_id_correct"] += int(bool(content["request_id_correct"]))
    anchor_recall = content["answer_anchor_recall"]
    if anchor_recall is not None:
        counter["answer_anchor_scored"] += 1
        counter["answer_anchor_recall_sum"] += float(anchor_recall)
    invented = content["invented_evidence_count"]
    if invented is not None:
        counter["invented_evidence_ids"] += int(invented)
        counter["evidence_scored"] += 1
        counter["evidence_precision_sum"] += float(content["evidence_precision"])
        counter["evidence_recall_sum"] += float(content["evidence_recall"])
        counter["evidence_f1_sum"] += float(content["evidence_f1"])


def _finish_counter(counter: Mapping[str, Any]) -> dict[str, Any]:
    attempted = int(counter["attempted"])
    evidence_scored = int(counter["evidence_scored"])
    anchor_scored = int(counter["answer_anchor_scored"])
    return {
        "attempted": attempted,
        "contract_valid": int(counter["contract_valid"]),
        "invalid_contract": int(counter["invalid_contract"]),
        "execution_error": int(counter["execution_error"]),
        **{metric: int(counter[metric]) for metric in _CONTENT_METRICS},
        "request_id_correct": int(counter["request_id_correct"]),
        "invented_evidence_ids": int(counter["invented_evidence_ids"]),
        "evidence_scored": evidence_scored,
        "answer_anchor_scored": anchor_scored,
        "rates": {
            "contract_valid": round(counter["contract_valid"] / attempted, 6)
            if attempted
            else 0.0,
            **{
                metric: round(counter[metric] / attempted, 6) if attempted else 0.0
                for metric in _CONTENT_METRICS
            },
            "request_id_correct": round(counter["request_id_correct"] / attempted, 6)
            if attempted
            else 0.0,
            "answer_anchor_recall_mean": round(
                counter["answer_anchor_recall_sum"] / anchor_scored, 6
            )
            if anchor_scored
            else 0.0,
            "evidence_precision_mean": round(
                counter["evidence_precision_sum"] / evidence_scored, 6
            )
            if evidence_scored
            else 0.0,
            "evidence_recall_mean": round(
                counter["evidence_recall_sum"] / evidence_scored, 6
            )
            if evidence_scored
            else 0.0,
            "evidence_f1_mean": round(
                counter["evidence_f1_sum"] / evidence_scored, 6
            )
            if evidence_scored
            else 0.0,
        },
    }


def _case_key(case: HeldoutCase) -> str:
    return hashlib.sha256(case.example_id.encode("utf-8")).hexdigest()


def _failed_case(case: HeldoutCase, error: BaseException) -> dict[str, Any]:
    content = _content_score(None, case.target, case)
    return {
        "case_sha256": _case_key(case),
        "task_type": case.task_type,
        "status": "execution_error",
        "elapsed_ms": 0.0,
        "contract_valid": False,
        "content_metrics": {
            key: value
            for key, value in content.items()
            if key != "prediction_answer_sha256"
        },
        "target_answer_sha256": hashlib.sha256(
            normalize_exact_answer(str(case.target["answer"])).encode("utf-8")
        ).hexdigest(),
        "prediction_answer_sha256": None,
        "error": _safe_error(error),
    }


def _run_case(
    case: HeldoutCase,
    config: OllamaCLIConfig,
    runner: CommandRunner,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    prompt = render_ollama_prompt(case)
    if len(prompt.encode("utf-8")) > config.max_prompt_bytes:
        raise OllamaCLIBenchmarkError("le prompt depasse la limite locale")
    started = time.perf_counter()
    strict_output: Mapping[str, Any] | None = None
    content_output: Mapping[str, Any] | None = None
    error_value = None
    try:
        result = runner.run(
            (
                config.executable,
                "run",
                config.model,
                "--format",
                "json",
                "--nowordwrap",
            ),
            input_text=prompt,
            timeout_seconds=timeout_seconds,
            stdout_limit=config.max_stdout_bytes,
            stderr_limit=config.max_stderr_bytes,
        )
        if result.returncode != 0:
            raise OllamaCLIBenchmarkError(
                f"ollama run a retourne le code {result.returncode}"
            )
        content_output = _first_json_mapping(result.stdout)
        try:
            strict_output = _validate_raw_generated_answer(
                result.stdout,
                case.capsule,
            )
        except MATLMInferenceError as error:
            error_value = _safe_error(error)
        status = "ok" if strict_output is not None else "invalid_contract"
    except Exception as error:
        status = "execution_error"
        error_value = _safe_error(error)

    scored_output = strict_output if strict_output is not None else content_output
    content = _content_score(scored_output, case.target, case)
    target_hash = hashlib.sha256(
        normalize_exact_answer(str(case.target["answer"])).encode("utf-8")
    ).hexdigest()
    return {
        "case_sha256": _case_key(case),
        "task_type": case.task_type,
        "status": status,
        "elapsed_ms": round((time.perf_counter() - started) * 1_000, 3),
        "contract_valid": strict_output is not None,
        "content_metrics": {
            key: value for key, value in content.items() if key != "prediction_answer_sha256"
        },
        "target_answer_sha256": target_hash,
        "prediction_answer_sha256": content["prediction_answer_sha256"],
        "error": error_value,
    }


def _preflight(
    config: OllamaCLIConfig,
    runner: CommandRunner,
) -> dict[str, Any]:
    version = runner.run(
        (config.executable, "--version"),
        input_text=None,
        timeout_seconds=config.preflight_timeout_seconds,
        stdout_limit=32_768,
        stderr_limit=config.max_stderr_bytes,
    )
    if version.returncode != 0:
        raise OllamaCLIBenchmarkError("ollama --version a echoue")
    shown = runner.run(
        (config.executable, "show", config.model),
        input_text=None,
        timeout_seconds=config.preflight_timeout_seconds,
        stdout_limit=config.max_stdout_bytes,
        stderr_limit=config.max_stderr_bytes,
    )
    if shown.returncode != 0:
        raise OllamaCLIBenchmarkError(
            "le modele demande n'est pas installe dans Ollama local; aucun pull automatique"
        )
    listed = runner.run(
        (config.executable, "list"),
        input_text=None,
        timeout_seconds=config.preflight_timeout_seconds,
        stdout_limit=config.max_stdout_bytes,
        stderr_limit=config.max_stderr_bytes,
    )
    if listed.returncode != 0:
        raise OllamaCLIBenchmarkError("ollama list a echoue")
    manifest_id = None
    for line in listed.stdout.splitlines():
        fields = line.split()
        if (
            len(fields) >= 2
            and fields[0] == config.model
            and _MODEL_MANIFEST_ID.fullmatch(fields[1])
        ):
            manifest_id = fields[1].lower()
            break
    if manifest_id is None:
        raise OllamaCLIBenchmarkError(
            "le tag exact n'a pas d'identifiant de manifeste dans ollama list"
        )
    if (
        config.expected_manifest_id is not None
        and manifest_id != config.expected_manifest_id
    ):
        raise OllamaCLIBenchmarkError(
            "l'identifiant du manifeste local ne correspond pas a celui attendu"
        )
    return {
        "version_output_sha256": hashlib.sha256(
            version.stdout.encode("utf-8")
        ).hexdigest(),
        "model_show_sha256": hashlib.sha256(shown.stdout.encode("utf-8")).hexdigest(),
        "manifest_id": manifest_id,
    }


def _release_model(
    config: OllamaCLIConfig,
    runner: CommandRunner,
) -> bool:
    try:
        stopped = runner.run(
            (config.executable, "stop", config.model),
            input_text=None,
            timeout_seconds=config.stop_timeout_seconds,
            stdout_limit=32_768,
            stderr_limit=config.max_stderr_bytes,
        )
        return stopped.returncode == 0
    except OllamaCLIBenchmarkError:
        return False


def benchmark_plan(
    selection: HeldoutSelection,
    config: OllamaCLIConfig,
) -> dict[str, Any]:
    """Produit un plan complet sans chercher Ollama et sans lancer de modele."""

    if not isinstance(selection, HeldoutSelection):
        raise TypeError("selection doit etre une HeldoutSelection")
    clean = validate_ollama_cli_config(config, resolve_executable=False)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "dry-run",
        "network": "offline",
        "transport": "ollama-cli-stdin",
        "model": clean.model,
        "expected_manifest_id": clean.expected_manifest_id,
        "executable": Path(clean.executable).name,
        "selected_count": len(selection.cases),
        "selection_sha256": selection.selection_sha256,
        "task_type_counts": dict(selection.task_counts),
        "limits": {
            "case_timeout_seconds": clean.case_timeout_seconds,
            "total_timeout_seconds": clean.total_timeout_seconds,
            "max_prompt_bytes": clean.max_prompt_bytes,
            "max_stdout_bytes": clean.max_stdout_bytes,
            "max_stderr_bytes": clean.max_stderr_bytes,
        },
        "safety": {
            "external_api": False,
            "http_client": False,
            "model_download": False,
            "one_model_tag": True,
            "raw_inputs_stored": False,
            "raw_outputs_stored": False,
        },
    }


def run_ollama_cli_benchmark(
    selection: HeldoutSelection,
    config: OllamaCLIConfig,
    *,
    runner: CommandRunner | None = None,
    resolve_executable: bool = True,
) -> dict[str, Any]:
    """Execute Qwen/Ollama sequentiellement sur la selection MAT-LM exacte."""

    if not isinstance(selection, HeldoutSelection):
        raise TypeError("selection doit etre une HeldoutSelection")
    clean = validate_ollama_cli_config(
        config,
        resolve_executable=resolve_executable,
    )
    active_runner = runner or SubprocessCommandRunner()
    preflight = _preflight(clean, active_runner)
    started = time.perf_counter()
    deadline = started + clean.total_timeout_seconds
    results: list[dict[str, Any]] = []
    release_succeeded = False
    try:
        for index, case in enumerate(selection.cases):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                error = OllamaCLITimeoutError("limite totale du benchmark depassee")
                results.extend(
                    _failed_case(pending, error)
                    for pending in selection.cases[index:]
                )
                break
            results.append(
                _run_case(
                    case,
                    clean,
                    active_runner,
                    timeout_seconds=min(clean.case_timeout_seconds, remaining),
                )
            )
    finally:
        release_succeeded = _release_model(clean, active_runner)

    global_counter = _new_counter()
    task_counters = {task: _new_counter() for task in SYNTHETIC_TASK_ORDER}
    for result in results:
        _record(global_counter, result)
        _record(task_counters[result["task_type"]], result)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete"
        if all(result["status"] != "execution_error" for result in results)
        else "complete_with_failures",
        "network": "offline",
        "transport": "ollama-cli-stdin",
        "dataset": {
            "sha256": selection.source_sha256,
            "available_count": selection.available_count,
            "selected_count": len(selection.cases),
            "selection_method": "deterministic-balanced-prefix-v1",
            "selection_sha256": selection.selection_sha256,
            "task_type_order": list(SYNTHETIC_TASK_ORDER),
            "task_type_counts": dict(selection.task_counts),
        },
        "model": {
            "tag": clean.model,
            "executable": Path(clean.executable).name,
            "manifest_id_verified": clean.expected_manifest_id is not None,
            **preflight,
        },
        "global_metrics": _finish_counter(global_counter),
        "metrics_by_task_type": {
            task: _finish_counter(task_counters[task])
            for task in SYNTHETIC_TASK_ORDER
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1_000, 3),
        "model_release_succeeded": release_succeeded,
        "cases": results,
        "metric_definitions": {
            "contract_valid": "sortie brute acceptee sans reparation par memory-native-answer-v1",
            "answer_exact_normalized": "prose cible exacte apres NFKC, casefold et espaces reduits",
            "answer_anchor_recall": "rappel des identifiants, valeurs fictives, dates, ages, abstention et decision extraits de la cible",
            "answer_anchors_all": "toutes les ancres factuelles cible sont presentes; ce n'est pas une equivalence semantique",
            "content_core_exact": "prose normalisee, preuves et abstention exactes, meme si le contrat echoue",
            "content_all_exact": "content_core_exact et calculs exacts",
            "invented_evidence_ids": "identifiants cites mais absents de la capsule",
        },
        "limits": {
            "case_timeout_seconds": clean.case_timeout_seconds,
            "total_timeout_seconds": clean.total_timeout_seconds,
            "max_prompt_bytes": clean.max_prompt_bytes,
            "max_stdout_bytes": clean.max_stdout_bytes,
            "max_stderr_bytes": clean.max_stderr_bytes,
        },
        "safety": {
            "external_api": False,
            "http_client": False,
            "model_download": False,
            "same_selection_as_matlm": True,
            "one_model_tag": True,
            "sequential_cases": True,
            "raw_inputs_stored": False,
            "raw_targets_stored": False,
            "raw_outputs_stored": False,
            "case_identifiers_hashed": True,
            "errors_redacted": True,
        },
    }


__all__ = [
    "CommandResult",
    "CommandRunner",
    "DEFAULT_MODEL",
    "OllamaCLIBenchmarkError",
    "OllamaCLIConfig",
    "OllamaCLIOutputLimitError",
    "OllamaCLITimeoutError",
    "PLAN_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "SubprocessCommandRunner",
    "answer_anchor_metrics",
    "benchmark_plan",
    "render_ollama_prompt",
    "run_ollama_cli_benchmark",
    "validate_ollama_cli_config",
]
