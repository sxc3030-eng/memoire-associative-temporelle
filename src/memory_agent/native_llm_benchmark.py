"""Squelette séquentiel pour comparer un modèle avec et sans mémoire."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
import json
import re
import threading
import time
from typing import Any, Mapping, Protocol, Sequence

from .native_llm_contract import (
    ANSWER_JSON_SCHEMA,
    ContractValidationError,
    validate_answer,
    validate_capsule,
)


_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")


class ComparisonMode(str, Enum):
    BASELINE = "baseline"
    MEMORY = "memory"
    SPECIALIZED = "specialized"


class LocalModelSession(Protocol):
    """Adaptateur minimal qu'un exécuteur local devra fournir."""

    def generate_json(
        self,
        capsule: Mapping[str, Any],
        *,
        output_schema: Mapping[str, Any],
        mode: ComparisonMode,
    ) -> Mapping[str, Any] | str | bytes:
        """Produit uniquement l'objet de réponse demandé par le contrat."""


class LocalModelProvider(Protocol):
    """Ouvre une session et garantit sa libération à la sortie du contexte."""

    def open(self, model_id: str) -> AbstractContextManager[LocalModelSession]:
        """Charge au plus un modèle pour la durée du bloc ``with``."""


@dataclass(frozen=True, slots=True)
class ComparisonArm:
    mode: ComparisonMode
    model_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ComparisonMode):
            raise TypeError("mode doit être un ComparisonMode")
        if not isinstance(self.model_id, str) or not _NAME_PATTERN.fullmatch(self.model_id):
            raise ValueError("model_id doit être un identifiant local stable")

    @property
    def uses_memory(self) -> bool:
        return self.mode is not ComparisonMode.BASELINE


@dataclass(frozen=True, slots=True)
class ComparisonCase:
    case_id: str
    capsule: Mapping[str, Any]
    benchmark: str = "local-held-out"

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not _NAME_PATTERN.fullmatch(self.case_id):
            raise ValueError("case_id doit être un identifiant stable")
        if not isinstance(self.benchmark, str) or not _NAME_PATTERN.fullmatch(self.benchmark):
            raise ValueError("benchmark doit être un identifiant stable")
        object.__setattr__(self, "capsule", validate_capsule(self.capsule))


@dataclass(frozen=True, slots=True)
class ComparisonRunResult:
    case_id: str
    benchmark: str
    mode: ComparisonMode
    model_id: str
    used_memory: bool
    status: str
    elapsed_ms: float
    output: Mapping[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "benchmark": self.benchmark,
            "mode": self.mode.value,
            "model_id": self.model_id,
            "used_memory": self.used_memory,
            "status": self.status,
            "elapsed_ms": self.elapsed_ms,
            "output": self.output,
            "error": self.error,
        }


def default_comparison_plan(
    *, general_model_id: str, specialized_model_id: str
) -> tuple[ComparisonArm, ...]:
    """Plan équitable: même modèle général, puis modèle spécialisé."""

    return (
        ComparisonArm(ComparisonMode.BASELINE, general_model_id),
        ComparisonArm(ComparisonMode.MEMORY, general_model_id),
        ComparisonArm(ComparisonMode.SPECIALIZED, specialized_model_id),
    )


def _validated_plan(plan: Sequence[ComparisonArm]) -> tuple[ComparisonArm, ...]:
    clean = tuple(plan)
    expected = (
        ComparisonMode.BASELINE,
        ComparisonMode.MEMORY,
        ComparisonMode.SPECIALIZED,
    )
    if tuple(arm.mode for arm in clean) != expected:
        raise ValueError("le plan doit suivre baseline, memory, specialized exactement une fois")
    if clean[0].model_id != clean[1].model_id:
        raise ValueError("baseline et memory doivent utiliser le même modèle général")
    return clean


def _capsule_for_arm(
    capsule: Mapping[str, Any], arm: ComparisonArm
) -> dict[str, Any]:
    clean = validate_capsule(capsule)
    if arm.uses_memory:
        return clean
    baseline = json.loads(json.dumps(clean, ensure_ascii=False, allow_nan=False))
    baseline["evidence"] = []
    baseline["constraints"]["evidence_required"] = False
    baseline["constraints"]["max_evidence_ids"] = 0
    return validate_capsule(baseline)


def _safe_error(error: BaseException) -> str:
    text = " ".join(str(error).split()) or error.__class__.__name__
    return text[:500]


class SequentialComparisonHarness:
    """Exécute trois bras sans jamais garder deux modèles ouverts ensemble."""

    def __init__(self, provider: LocalModelProvider) -> None:
        if provider is None:
            raise TypeError("provider est requis")
        self.provider = provider
        self._run_lock = threading.Lock()

    def run(
        self,
        case: ComparisonCase,
        plan: Sequence[ComparisonArm],
    ) -> list[ComparisonRunResult]:
        if not isinstance(case, ComparisonCase):
            raise TypeError("case doit être un ComparisonCase")
        clean_plan = _validated_plan(plan)
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("un benchmark est déjà en cours sur ce harnais")
        try:
            results: list[ComparisonRunResult] = []
            for arm in clean_plan:
                capsule = _capsule_for_arm(case.capsule, arm)
                started = time.perf_counter()
                try:
                    # Le contexte doit fermer/décharger ce modèle avant le bras suivant.
                    with self.provider.open(arm.model_id) as session:
                        raw_output = session.generate_json(
                            capsule,
                            output_schema=ANSWER_JSON_SCHEMA,
                            mode=arm.mode,
                        )
                    output = validate_answer(raw_output, capsule)
                    status = "ok"
                    error_text = None
                except ContractValidationError as error:
                    output = None
                    status = "invalid_output"
                    error_text = _safe_error(error)
                except Exception as error:  # L'échec d'un modèle ne masque pas les bras suivants.
                    output = None
                    status = "model_error"
                    error_text = _safe_error(error)
                elapsed_ms = round((time.perf_counter() - started) * 1_000, 3)
                results.append(
                    ComparisonRunResult(
                        case_id=case.case_id,
                        benchmark=case.benchmark,
                        mode=arm.mode,
                        model_id=arm.model_id,
                        used_memory=arm.uses_memory,
                        status=status,
                        elapsed_ms=elapsed_ms,
                        output=output,
                        error=error_text,
                    )
                )
            return results
        finally:
            self._run_lock.release()


__all__ = [
    "ComparisonArm",
    "ComparisonCase",
    "ComparisonMode",
    "ComparisonRunResult",
    "LocalModelProvider",
    "LocalModelSession",
    "SequentialComparisonHarness",
    "default_comparison_plan",
]
