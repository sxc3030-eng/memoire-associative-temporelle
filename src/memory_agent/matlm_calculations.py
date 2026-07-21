"""Réexécution locale et bornée des calculs calendaires demandés par MAT-LM.

Ce module ne contient volontairement aucun interpréteur général. Une expression
n'est acceptée que si elle correspond entièrement à la grammaire fermée de
``calendar_age`` définie ci-dessous.
"""

from __future__ import annotations

import calendar
from datetime import date
import hashlib
import re
from typing import Any, Mapping

from .native_llm_contract import validate_answer, validate_capsule


_CALENDAR_AGE_RE = re.compile(
    r"calendar_age\("
    r"(?P<birth>[0-9]{4}-[0-9]{2}-[0-9]{2}),"
    r"(?P<event>[0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2})?)?),"
    r"precision=(?P<precision>day|month|year)"
    r"\)"
)
_EVENT_LENGTH_BY_PRECISION = {"day": 10, "month": 7, "year": 4}
_ANSWER_CALCULATION_RE = re.compile(
    r"(?P<prefix>\ble calcul donne[ \t]+)"
    r"(?P<value>(?:entre[ \t]+[0-9]{1,4}[ \t]+et[ \t]+[0-9]{1,4}|[0-9]{1,4}))"
    r"(?P<suffix>[ \t]+ans\b)",
    flags=re.IGNORECASE,
)


class MATLMCalculationError(ValueError):
    """Une demande de calcul MAT-LM n'appartient pas au langage sûr supporté."""


def _calculation_error(index: int, message: str) -> MATLMCalculationError:
    return MATLMCalculationError(f"$.calculations[{index}].expression: {message}")


def _parse_day(value: str, *, index: int, label: str) -> date:
    try:
        year, month, day = (int(part) for part in value.split("-"))
        return date(year, month, day)
    except (TypeError, ValueError) as error:
        raise _calculation_error(index, f"{label} n'est pas une date ISO valide") from error


def _event_interval(
    value: str,
    precision: str,
    *,
    index: int,
) -> tuple[date, date]:
    if len(value) != _EVENT_LENGTH_BY_PRECISION[precision]:
        raise _calculation_error(index, "la date ne correspond pas à precision")

    try:
        if precision == "day":
            parsed = _parse_day(value, index=index, label="event_date")
            return parsed, parsed
        if precision == "month":
            year_text, month_text = value.split("-")
            year, month = int(year_text), int(month_text)
            last_day = calendar.monthrange(year, month)[1]
            return date(year, month, 1), date(year, month, last_day)
        year = int(value)
        return date(year, 1, 1), date(year, 12, 31)
    except (TypeError, ValueError) as error:
        raise _calculation_error(index, "event_date n'est pas une date ISO valide") from error


def _age_on(birth: date, event: date) -> int:
    return event.year - birth.year - ((event.month, event.day) < (birth.month, birth.day))


def _calendar_age(expression: str, *, index: int) -> tuple[str, str]:
    match = _CALENDAR_AGE_RE.fullmatch(expression)
    if match is None:
        raise _calculation_error(index, "seul calendar_age au format strict est autorisé")

    birth = _parse_day(match.group("birth"), index=index, label="birth_date")
    event_start, event_end = _event_interval(
        match.group("event"),
        match.group("precision"),
        index=index,
    )
    if event_start < birth:
        raise _calculation_error(index, "l'événement ne peut pas précéder la naissance")

    minimum = _age_on(birth, event_start)
    maximum = _age_on(birth, event_end)
    if minimum == maximum:
        value = str(minimum)
        return value, value
    value = f"{minimum}..{maximum}"
    return value, f"entre {minimum} et {maximum}"


def _deterministic_calculation_id(request_id: str, index: int) -> str:
    # Le cas courant (un calcul) reste bit pour bit identique au curriculum.
    material = request_id if index == 0 else f"{request_id}:{index}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"calc:{digest}"


def _correct_answer_text(answer: str, rendered_results: list[str]) -> str:
    """Corrige seulement des formules françaises non ambiguës, dans leur ordre."""

    matches = list(_ANSWER_CALCULATION_RE.finditer(answer))
    if not rendered_results or len(matches) != len(rendered_results):
        return answer

    pieces: list[str] = []
    cursor = 0
    for match, result in zip(matches, rendered_results, strict=True):
        pieces.append(answer[cursor : match.start()])
        pieces.append(match.group("prefix"))
        pieces.append(result)
        pieces.append(match.group("suffix"))
        cursor = match.end()
    pieces.append(answer[cursor:])
    return "".join(pieces)


def reexecute_matlm_calculations(
    response: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    """Recalcule les ``calendar_age`` d'une réponse validée et revalide la copie.

    ``reported_result``, ``unit`` et ``calculation_id`` viennent uniquement du
    moteur déterministe. L'expression et les ``evidence_ids`` déjà validés sont
    conservés. L'objet fourni par l'appelant n'est jamais modifié.
    """

    trusted_capsule = validate_capsule(capsule)
    corrected = validate_answer(response, trusted_capsule)
    rendered_results: list[str] = []
    calculation_ids: set[str] = set()

    for index, calculation in enumerate(corrected["calculations"]):
        reported_result, rendered_result = _calendar_age(
            calculation["expression"],
            index=index,
        )
        calculation_id = _deterministic_calculation_id(corrected["request_id"], index)
        if calculation_id in calculation_ids:  # Défense déterministe contre une collision.
            raise MATLMCalculationError("les calculation_id déterministes ne sont pas uniques")
        calculation_ids.add(calculation_id)
        calculation["calculation_id"] = calculation_id
        calculation["reported_result"] = reported_result
        calculation["unit"] = "ans"
        rendered_results.append(rendered_result)

    corrected_answer = _correct_answer_text(corrected["answer"], rendered_results)
    maximum = trusted_capsule["constraints"]["max_answer_characters"]
    if len(corrected_answer) > maximum:
        raise MATLMCalculationError("la correction de answer dépasserait la borne autorisée")
    corrected["answer"] = corrected_answer
    return validate_answer(corrected, trusted_capsule)


__all__ = ["MATLMCalculationError", "reexecute_matlm_calculations"]
