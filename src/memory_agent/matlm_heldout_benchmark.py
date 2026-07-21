"""Banc d'essai held-out local pour Granite nu et Granite + LoRA.

Le lot est sélectionné une seule fois, puis chaque bras garde exactement une
session ouverte pendant tout le lot. Les sorties brutes ne sont jamais écrites
dans le rapport: seules des métriques, des empreintes et des erreurs bornées le
sont.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import unicodedata
from uuid import uuid4

from .matlm_inference import (
    InferenceConfig,
    MATLMInferenceError,
    MATLMInferenceSession,
    validate_inference_config,
)
from .matlm_training import MATLMTrainingError, load_training_jsonl
from .memory_native_curriculum import SYNTHETIC_TASK_ORDER
from .native_llm_contract import (
    ANSWER_SCHEMA_VERSION,
    ContractValidationError,
    validate_answer,
    validate_capsule,
)


REPORT_SCHEMA_VERSION = "matlm-heldout-benchmark-v1"
DEFAULT_LIMIT = 27
MAX_LIMIT = 50_000
MAX_ERROR_CHARACTERS = 400
MAX_REPORT_BYTES = 20_000_000
_ARM_NAMES = frozenset({"base", "adapter"})
_SAFE_ARM_NAME = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


class MATLMHeldoutBenchmarkError(RuntimeError):
    """Le benchmark ne peut pas produire un résultat fiable et borné."""


class BenchmarkSession(Protocol):
    """Surface minimale d'une session locale chargée une seule fois."""

    effective_mode: str | None

    def ask(self, capsule: Mapping[str, Any]) -> Mapping[str, Any] | str | bytes:
        """Répond à une capsule avec le contrat natif."""


SessionFactory = Callable[
    ["BenchmarkArm"], AbstractContextManager[BenchmarkSession]
]


@dataclass(frozen=True, slots=True)
class BenchmarkArm:
    """Un seul candidat local, Granite nu ou Granite + adaptateur."""

    name: str
    inference: InferenceConfig

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SAFE_ARM_NAME.fullmatch(self.name):
            raise ValueError("name doit être un identifiant de bras stable")
        if self.name not in _ARM_NAMES:
            raise ValueError("name doit valoir base ou adapter")
        if not isinstance(self.inference, InferenceConfig):
            raise TypeError("inference doit être une InferenceConfig")
        if self.inference.allow_model_download:
            raise ValueError("le benchmark held-out interdit tout téléchargement")
        if self.name == "base" and self.inference.adapter_path is not None:
            raise ValueError("le bras base ne doit pas charger d'adaptateur")
        if self.name == "adapter" and self.inference.adapter_path is None:
            raise ValueError("le bras adapter exige un adaptateur local")


@dataclass(frozen=True, slots=True)
class HeldoutCase:
    example_id: str
    task_type: str
    capsule: Mapping[str, Any]
    target: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class HeldoutSelection:
    source_path: Path
    source_sha256: str
    available_count: int
    cases: tuple[HeldoutCase, ...]
    task_counts: Mapping[str, int]
    selection_sha256: str


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def _selection_digest(cases: Sequence[HeldoutCase]) -> str:
    payload = [case.example_id for case in cases]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def normalize_exact_answer(value: Any) -> str:
    """Normalise Unicode, casse et espaces sans modifier la ponctuation."""

    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise MATLMHeldoutBenchmarkError("limit doit être un entier")
    task_count = len(SYNTHETIC_TASK_ORDER)
    if not task_count <= limit <= MAX_LIMIT:
        raise MATLMHeldoutBenchmarkError(
            f"limit doit être compris entre {task_count} et {MAX_LIMIT}"
        )
    return limit


def load_balanced_heldout(
    path: str | Path,
    *,
    limit: int = DEFAULT_LIMIT,
) -> HeldoutSelection:
    """Valide le JSONL et choisit un quota déterministe pour les neuf tâches."""

    clean_limit = _validate_limit(limit)
    try:
        loaded = load_training_jsonl(path)
    except MATLMTrainingError as error:
        raise MATLMHeldoutBenchmarkError(f"jeu held-out invalide: {error}") from error
    if clean_limit > len(loaded.rows):
        raise MATLMHeldoutBenchmarkError(
            f"limit={clean_limit} dépasse les {len(loaded.rows)} exemples disponibles"
        )

    grouped: dict[str, list[tuple[int, HeldoutCase]]] = {
        task: [] for task in SYNTHETIC_TASK_ORDER
    }
    for index, row in enumerate(loaded.rows):
        task = row.get("task")
        if task not in grouped:
            raise MATLMHeldoutBenchmarkError(
                f"tâche held-out inconnue à la ligne {index + 1}: {task!r}"
            )
        try:
            capsule = validate_capsule(row["memory_capsule"])
            target = validate_answer(row["target"], capsule)
        except (KeyError, ContractValidationError) as error:
            raise MATLMHeldoutBenchmarkError(
                f"contrat held-out invalide à la ligne {index + 1}: {error}"
            ) from error
        example_id = row.get("example_id")
        if example_id != capsule["request_id"]:
            raise MATLMHeldoutBenchmarkError(
                f"example_id et request_id diffèrent à la ligne {index + 1}"
            )
        grouped[task].append(
            (
                index,
                HeldoutCase(
                    example_id=example_id,
                    task_type=task,
                    capsule=capsule,
                    target=target,
                ),
            )
        )

    quotient, remainder = divmod(clean_limit, len(SYNTHETIC_TASK_ORDER))
    quotas = {
        task: quotient + (1 if index < remainder else 0)
        for index, task in enumerate(SYNTHETIC_TASK_ORDER)
    }
    selected_with_positions: list[tuple[int, HeldoutCase]] = []
    for task in SYNTHETIC_TASK_ORDER:
        available = grouped[task]
        quota = quotas[task]
        if len(available) < quota:
            raise MATLMHeldoutBenchmarkError(
                f"tâche {task}: {len(available)} exemple(s), quota requis {quota}"
            )
        selected_with_positions.extend(available[:quota])
    # Préserve l'ordre original, mais le même tuple immuable est réutilisé pour
    # chaque bras: aucun candidat ne bénéficie d'un autre lot.
    selected_with_positions.sort(key=lambda item: item[0])
    cases = tuple(case for _, case in selected_with_positions)
    counts = {
        task: sum(case.task_type == task for case in cases)
        for task in SYNTHETIC_TASK_ORDER
    }
    return HeldoutSelection(
        source_path=loaded.summary.path,
        source_sha256=loaded.summary.sha256,
        available_count=len(loaded.rows),
        cases=cases,
        task_counts=counts,
        selection_sha256=_selection_digest(cases),
    )


def build_benchmark_arms(
    *,
    compare: str,
    base_model: str,
    adapter_path: str | Path | None,
    load_mode: str = "auto",
    device_index: int = 0,
    max_input_tokens: int = 4_096,
    max_new_tokens: int = 768,
    seed: int = 20_260_721,
) -> tuple[BenchmarkArm, ...]:
    """Construit et valide un plan strictement hors ligne."""

    if compare not in {"base", "adapter", "both"}:
        raise MATLMHeldoutBenchmarkError("compare doit valoir base, adapter ou both")
    common = {
        "base_model": base_model,
        "load_mode": load_mode,
        "device_index": device_index,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "allow_model_download": False,
    }
    arms: list[BenchmarkArm] = []
    if compare in {"base", "both"}:
        try:
            config = validate_inference_config(
                InferenceConfig(adapter_path=None, **common)
            )
        except MATLMInferenceError as error:
            raise MATLMHeldoutBenchmarkError(f"bras base invalide: {error}") from error
        arms.append(BenchmarkArm(name="base", inference=config))
    if compare in {"adapter", "both"}:
        if adapter_path is None:
            raise MATLMHeldoutBenchmarkError(
                "adapter_path est requis pour le bras adapter"
            )
        try:
            config = validate_inference_config(
                InferenceConfig(adapter_path=Path(adapter_path), **common)
            )
        except MATLMInferenceError as error:
            raise MATLMHeldoutBenchmarkError(f"bras adapter invalide: {error}") from error
        arms.append(BenchmarkArm(name="adapter", inference=config))
    return tuple(arms)


def _safe_error(error: BaseException) -> dict[str, str]:
    error_type = re.sub(r"[^A-Za-z0-9_.-]", "", type(error).__name__)[:80] or "Error"
    message = " ".join(str(error).replace("\x00", " ").split())
    if not message:
        message = error_type
    return {"type": error_type, "message": message[:MAX_ERROR_CHARACTERS]}


def _error_status(error: BaseException) -> str:
    if isinstance(error, ContractValidationError):
        return "invalid_output"
    if isinstance(error, MATLMInferenceError):
        lowered = str(error).casefold()
        markers = (
            "sortie mat-lm invalide",
            "sortie du modèle",
            "objet json",
            "texte interdit",
            "accolades json",
            "clé json répétée",
            "nombre json non fini",
        )
        if any(marker in lowered for marker in markers):
            return "invalid_output"
    return "model_error"


_METRIC_NAMES = (
    "contract_valid",
    "answer_exact_normalized",
    "evidence_ids_exact",
    "abstention_correct",
    "request_id_correct",
    "schema_version_correct",
    "calculations_exact",
    "all_required_exact",
    "full_target_exact",
)


def _new_counter() -> dict[str, int]:
    return {
        "attempted": 0,
        "invalid_output": 0,
        "model_error": 0,
        **{name: 0 for name in _METRIC_NAMES},
    }


def _finish_counter(counter: Mapping[str, int]) -> dict[str, Any]:
    attempted = int(counter["attempted"])
    result: dict[str, Any] = dict(counter)
    result["rates"] = {
        name: round(counter[name] / attempted, 6) if attempted else 0.0
        for name in _METRIC_NAMES
    }
    return result


def _answer_hash(answer: str) -> str:
    return hashlib.sha256(normalize_exact_answer(answer).encode("utf-8")).hexdigest()


def _score_valid_answer(
    output: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, bool]:
    flags = {
        "contract_valid": True,
        "answer_exact_normalized": (
            normalize_exact_answer(output["answer"])
            == normalize_exact_answer(target["answer"])
        ),
        "evidence_ids_exact": output["evidence_ids"] == target["evidence_ids"],
        "abstention_correct": output["abstention"] == target["abstention"],
        "request_id_correct": output["request_id"] == target["request_id"],
        "schema_version_correct": (
            output["schema_version"] == ANSWER_SCHEMA_VERSION
            and output["schema_version"] == target["schema_version"]
        ),
        "calculations_exact": output["calculations"] == target["calculations"],
        "full_target_exact": output == target,
    }
    flags["all_required_exact"] = all(
        flags[name]
        for name in (
            "contract_valid",
            "answer_exact_normalized",
            "evidence_ids_exact",
            "abstention_correct",
            "request_id_correct",
            "schema_version_correct",
        )
    )
    return flags


def _record_counter(counter: dict[str, int], result: Mapping[str, Any]) -> None:
    counter["attempted"] += 1
    status = result["status"]
    if status in {"invalid_output", "model_error"}:
        counter[status] += 1
    metrics = result["metrics"]
    for name in _METRIC_NAMES:
        counter[name] += int(bool(metrics[name]))


def _default_session_factory(
    arm: BenchmarkArm,
) -> AbstractContextManager[BenchmarkSession]:
    return MATLMInferenceSession(arm.inference)


def _failed_case(case: HeldoutCase, error: BaseException) -> dict[str, Any]:
    return {
        "example_id": case.example_id,
        "task_type": case.task_type,
        "status": _error_status(error),
        "elapsed_ms": 0.0,
        "metrics": {name: False for name in _METRIC_NAMES},
        "target_answer_sha256": _answer_hash(str(case.target["answer"])),
        "prediction_answer_sha256": None,
        "error": _safe_error(error),
    }


def _run_case(session: BenchmarkSession, case: HeldoutCase) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        raw_output = session.ask(case.capsule)
        output = validate_answer(raw_output, case.capsule)
        metrics = _score_valid_answer(output, case.target)
        status = "ok"
        prediction_hash = _answer_hash(output["answer"])
        error_value = None
    except Exception as error:
        metrics = {name: False for name in _METRIC_NAMES}
        status = _error_status(error)
        prediction_hash = None
        error_value = _safe_error(error)
    elapsed_ms = round((time.perf_counter() - started) * 1_000, 3)
    return {
        "example_id": case.example_id,
        "task_type": case.task_type,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "metrics": metrics,
        "target_answer_sha256": _answer_hash(str(case.target["answer"])),
        "prediction_answer_sha256": prediction_hash,
        "error": error_value,
    }


def _arm_model_description(arm: BenchmarkArm) -> dict[str, Any]:
    return {
        "base_model": _portable_path_label(arm.inference.base_model),
        "adapter": (
            _portable_path_label(arm.inference.adapter_path)
            if arm.inference.adapter_path is not None
            else None
        ),
        "adapter_enabled": arm.inference.adapter_path is not None,
        "requested_load_mode": arm.inference.load_mode,
        "device": f"xpu:{arm.inference.device_index}",
    }


def _portable_path_label(value: str | Path) -> str:
    """Évite de publier le profil local tout en identifiant l'artefact."""

    raw = str(value)
    path = Path(raw)
    if path.is_absolute():
        return f"{path.parent.name}/{path.name}"
    return raw.replace("\\", "/")


def _run_arm(
    arm: BenchmarkArm,
    selection: HeldoutSelection,
    session_factory: SessionFactory,
) -> dict[str, Any]:
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    arm_error = None
    effective_mode = None
    session_opened = False
    try:
        # Une seule ouverture pour tout le lot de ce bras.
        with session_factory(arm) as session:
            session_opened = True
            effective_mode = getattr(session, "effective_mode", None)
            for case in selection.cases:
                cases.append(_run_case(session, case))
    except Exception as error:
        arm_error = _safe_error(error)
        completed = len(cases)
        for case in selection.cases[completed:]:
            cases.append(_failed_case(case, error))

    global_counter = _new_counter()
    task_counters = {task: _new_counter() for task in SYNTHETIC_TASK_ORDER}
    for result in cases:
        _record_counter(global_counter, result)
        _record_counter(task_counters[result["task_type"]], result)
    successful_generation = sum(case["status"] == "ok" for case in cases)
    if arm_error is not None:
        status = "partial" if successful_generation else "load_error"
    elif successful_generation == len(cases):
        status = "complete"
    else:
        # Une sortie invalide est un résultat de benchmark, pas une panne du lot.
        status = "complete_with_failures"
    return {
        "name": arm.name,
        "status": status,
        "model": _arm_model_description(arm),
        "session_open_count": 1 if session_opened else 0,
        "effective_load_mode": effective_mode,
        "elapsed_ms": round((time.perf_counter() - started) * 1_000, 3),
        "global_metrics": _finish_counter(global_counter),
        "metrics_by_task_type": {
            task: _finish_counter(task_counters[task])
            for task in SYNTHETIC_TASK_ORDER
        },
        "arm_error": arm_error,
        "cases": cases,
    }


def run_heldout_benchmark(
    selection: HeldoutSelection,
    arms: Sequence[BenchmarkArm],
    *,
    session_factory: SessionFactory | None = None,
) -> dict[str, Any]:
    """Exécute les bras séquentiellement sur exactement la même sélection."""

    if not isinstance(selection, HeldoutSelection):
        raise TypeError("selection doit être une HeldoutSelection")
    clean_arms = tuple(arms)
    if not clean_arms:
        raise MATLMHeldoutBenchmarkError("au moins un bras est requis")
    if len(clean_arms) > 2 or len({arm.name for arm in clean_arms}) != len(clean_arms):
        raise MATLMHeldoutBenchmarkError("les bras base/adapter doivent être uniques")
    factory = session_factory or _default_session_factory
    run_started = time.perf_counter()
    arm_reports = [
        _run_arm(arm, selection, factory)
        for arm in clean_arms
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "network": "offline",
        "dataset": {
            "path": _portable_path_label(selection.source_path),
            "sha256": selection.source_sha256,
            "available_count": selection.available_count,
            "selected_count": len(selection.cases),
            "selection_method": "deterministic-balanced-prefix-v1",
            "selection_sha256": selection.selection_sha256,
            "task_type_order": list(SYNTHETIC_TASK_ORDER),
            "task_type_counts": dict(selection.task_counts),
        },
        "arms": arm_reports,
        "elapsed_ms": round((time.perf_counter() - run_started) * 1_000, 3),
        "metric_definitions": {
            "answer_exact_normalized": "Unicode NFKC + casefold + espaces réduits; ponctuation conservée",
            "evidence_ids_exact": "liste et ordre identiques à la cible",
            "abstention_correct": "objet abstention identique à la cible",
            "all_required_exact": "contrat, réponse, preuves, abstention, request_id et schéma corrects",
        },
        "safety": {
            "external_api": False,
            "model_download": False,
            "same_selection_for_every_arm": True,
            "one_session_per_arm": True,
            "one_model_loaded_at_a_time": True,
            "raw_model_output_stored": False,
            "max_error_characters": MAX_ERROR_CHARACTERS,
        },
    }


def write_atomic_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    replace: bool = False,
) -> Path:
    """Écrit un rapport JSON borné via un fichier temporaire voisin."""

    output = Path(path).expanduser().resolve()
    if output.suffix.casefold() != ".json":
        raise MATLMHeldoutBenchmarkError("le rapport doit avoir l'extension .json")
    if output.exists() and not replace:
        raise MATLMHeldoutBenchmarkError(f"le rapport existe déjà: {output}")
    try:
        encoded = (_canonical_json(dict(report), indent=2) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MATLMHeldoutBenchmarkError("le rapport contient du JSON invalide") from error
    if len(encoded) > MAX_REPORT_BYTES:
        raise MATLMHeldoutBenchmarkError(
            f"le rapport dépasse la limite de {MAX_REPORT_BYTES} octets"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except OSError as error:
        raise MATLMHeldoutBenchmarkError(
            f"impossible d'écrire le rapport atomique: {output}"
        ) from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return output


def benchmark_plan(
    selection: HeldoutSelection,
    arms: Sequence[BenchmarkArm],
) -> dict[str, Any]:
    """Plan compact sans charger Torch ni aucun modèle."""

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "dry-run",
        "network": "offline",
        "selection_sha256": selection.selection_sha256,
        "selected_count": len(selection.cases),
        "task_type_counts": dict(selection.task_counts),
        "arms": [
            {"name": arm.name, "model": _arm_model_description(arm)}
            for arm in arms
        ],
    }


__all__ = [
    "BenchmarkArm",
    "BenchmarkSession",
    "DEFAULT_LIMIT",
    "HeldoutCase",
    "HeldoutSelection",
    "MATLMHeldoutBenchmarkError",
    "REPORT_SCHEMA_VERSION",
    "benchmark_plan",
    "build_benchmark_arms",
    "load_balanced_heldout",
    "normalize_exact_answer",
    "run_heldout_benchmark",
    "write_atomic_report",
]
