"""Contrat JSON strict entre la mémoire et un petit modèle local dédié."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence


CAPSULE_SCHEMA_VERSION = "memory-native-capsule-v1"
ANSWER_SCHEMA_VERSION = "memory-native-answer-v1"
MAX_CAPSULE_BYTES = 1_000_000
MAX_ANSWER_BYTES = 65_536
MAX_EVIDENCE_ITEMS = 64

_ID_PATTERN_TEXT = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"
_ID_PATTERN = re.compile(_ID_PATTERN_TEXT)
_SPACES = frozenset({"private", "shared", "reference"})
_EVIDENCE_STATUSES = frozenset(
    {"confirmed", "verified", "observed", "executed", "derived", "unverified"}
)
_ABSTENTION_REASONS = frozenset(
    {
        "none",
        "insufficient_evidence",
        "contradictory_evidence",
        "out_of_scope",
        "unsafe_request",
        "invalid_capsule",
    }
)


CAPSULE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://memoire-locale.invalid/schema/memory-native-capsule-v1.json",
    "title": "Memory-native input capsule",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "request_id", "question", "evidence", "constraints"],
    "properties": {
        "schema_version": {"const": CAPSULE_SCHEMA_VERSION},
        "request_id": {"type": "string", "pattern": _ID_PATTERN_TEXT},
        "question": {"type": "string", "minLength": 1, "maxLength": 8_000},
        "evidence": {
            "type": "array",
            "maxItems": MAX_EVIDENCE_ITEMS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "evidence_id",
                    "text",
                    "space",
                    "status",
                    "confidence",
                    "temporal_context",
                    "tags",
                ],
                "properties": {
                    "evidence_id": {"type": "string", "pattern": _ID_PATTERN_TEXT},
                    "text": {"type": "string", "minLength": 1, "maxLength": 4_000},
                    "space": {"enum": sorted(_SPACES)},
                    "status": {"enum": sorted(_EVIDENCE_STATUSES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "temporal_context": {
                        "type": ["string", "null"],
                        "maxLength": 256,
                    },
                    "tags": {
                        "type": "array",
                        "maxItems": 16,
                        "items": {"type": "string", "minLength": 1, "maxLength": 64},
                    },
                },
            },
        },
        "constraints": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "evidence_required",
                "allow_calculations",
                "max_answer_characters",
                "max_evidence_ids",
                "max_calculations",
            ],
            "properties": {
                "evidence_required": {"type": "boolean"},
                "allow_calculations": {"type": "boolean"},
                "max_answer_characters": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20_000,
                },
                "max_evidence_ids": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_EVIDENCE_ITEMS,
                },
                "max_calculations": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 32,
                },
            },
        },
    },
}


ANSWER_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://memoire-locale.invalid/schema/memory-native-answer-v1.json",
    "title": "Memory-native validated answer",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "request_id",
        "answer",
        "confidence",
        "evidence_ids",
        "calculations",
        "abstention",
    ],
    "properties": {
        "schema_version": {"const": ANSWER_SCHEMA_VERSION},
        "request_id": {"type": "string", "pattern": _ID_PATTERN_TEXT},
        "answer": {"type": "string", "maxLength": 20_000},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_ids": {
            "type": "array",
            "maxItems": MAX_EVIDENCE_ITEMS,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": _ID_PATTERN_TEXT},
        },
        "calculations": {
            "type": "array",
            "maxItems": 32,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "calculation_id",
                    "expression",
                    "reported_result",
                    "unit",
                    "evidence_ids",
                ],
                "properties": {
                    "calculation_id": {"type": "string", "pattern": _ID_PATTERN_TEXT},
                    "expression": {"type": "string", "minLength": 1, "maxLength": 512},
                    "reported_result": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                    "unit": {"type": ["string", "null"], "maxLength": 64},
                    "evidence_ids": {
                        "type": "array",
                        "maxItems": MAX_EVIDENCE_ITEMS,
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": _ID_PATTERN_TEXT},
                    },
                },
            },
        },
        "abstention": {
            "type": "object",
            "additionalProperties": False,
            "required": ["abstained", "reason", "missing_information"],
            "properties": {
                "abstained": {"type": "boolean"},
                "reason": {"enum": sorted(_ABSTENTION_REASONS)},
                "missing_information": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 256},
                },
            },
        },
    },
}


def _model_schema_projection(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Retire les annotations/bornes que le validateur applique déjà après génération."""

    projected: dict[str, Any] = {}
    for key in ("type", "const", "enum", "additionalProperties", "required"):
        if key in schema:
            value = schema[key]
            projected[key] = list(value) if isinstance(value, list) else value
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        projected["properties"] = {
            str(key): _model_schema_projection(value)
            for key, value in properties.items()
            if isinstance(value, Mapping)
        }
    items = schema.get("items")
    if isinstance(items, Mapping):
        projected["items"] = _model_schema_projection(items)
    return projected


# Projection compacte destinée au prompt. ANSWER_JSON_SCHEMA et validate_answer
# restent l'autorité complète pour toute sortie du modèle.
MODEL_ANSWER_JSON_SCHEMA: dict[str, Any] = _model_schema_projection(ANSWER_JSON_SCHEMA)

MODEL_ANSWER_JSON_TEMPLATE: dict[str, Any] = {
    "schema_version": ANSWER_SCHEMA_VERSION,
    "request_id": "<request_id>",
    "answer": "<answer>",
    "confidence": 0.0,
    "evidence_ids": ["<evidence_id>"],
    "calculations": [
        {
            "calculation_id": "calc:<id>",
            "expression": "<expression>",
            "reported_result": "<result>",
            "unit": None,
            "evidence_ids": ["<evidence_id>"],
        }
    ],
    "abstention": {
        "abstained": False,
        "reason": "none",
        "missing_information": [],
    },
}


class ContractValidationError(ValueError):
    """Une entrée ou sortie ne respecte pas le contrat public."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _fail(path: str, message: str) -> None:
    raise ContractValidationError(path, message)


def _json_object(value: Any, *, label: str, max_bytes: int) -> dict[str, Any]:
    if isinstance(value, bytes):
        if len(value) > max_bytes:
            _fail(label, f"dépasse {max_bytes} octets")
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ContractValidationError(label, "doit être encodé en UTF-8") from error
    if isinstance(value, str):
        if len(value.encode("utf-8")) > max_bytes:
            _fail(label, f"dépasse {max_bytes} octets")

        def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    _fail(label, f"clé JSON répétée: {key}")
                result[key] = item
            return result

        try:
            value = json.loads(
                value,
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda constant: _fail(
                    label, f"nombre JSON non fini interdit: {constant}"
                ),
            )
        except ContractValidationError:
            raise
        except json.JSONDecodeError as error:
            raise ContractValidationError(label, "JSON invalide") from error
    elif isinstance(value, Mapping):
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ContractValidationError(label, "doit contenir uniquement du JSON") from error
        if len(encoded.encode("utf-8")) > max_bytes:
            _fail(label, f"dépasse {max_bytes} octets")
        value = json.loads(encoded)
    else:
        _fail(label, "doit être un objet JSON")
    if not isinstance(value, dict):
        _fail(label, "doit être un objet JSON")
    return value


def _exact_object(
    value: Any,
    *,
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "doit être un objet")
    allowed = required | (optional or set())
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        _fail(path, "champs requis absents: " + ", ".join(missing))
    if unknown:
        _fail(path, "champs inconnus: " + ", ".join(unknown))
    return value


def _string(
    value: Any,
    *,
    path: str,
    minimum: int = 0,
    maximum: int,
    identifier: bool = False,
) -> str:
    if not isinstance(value, str):
        _fail(path, "doit être une chaîne")
    if value != value.strip():
        _fail(path, "ne doit pas contenir d'espace au début ou à la fin")
    if not minimum <= len(value) <= maximum:
        _fail(path, f"longueur attendue entre {minimum} et {maximum}")
    if "\x00" in value:
        _fail(path, "ne doit pas contenir de caractère nul")
    if identifier and not _ID_PATTERN.fullmatch(value):
        _fail(path, "identifiant invalide")
    return value


def _boolean(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "doit être un booléen")
    return value


def _integer(value: Any, *, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "doit être un entier")
    if not minimum <= value <= maximum:
        _fail(path, f"doit être compris entre {minimum} et {maximum}")
    return value


def _number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "doit être un nombre")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        _fail(path, "doit être un nombre fini entre 0 et 1")
    return number


def _string_list(
    value: Any,
    *,
    path: str,
    maximum_items: int,
    maximum_length: int,
    identifiers: bool = False,
    unique: bool = True,
) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "doit être une liste")
    if len(value) > maximum_items:
        _fail(path, f"ne peut pas dépasser {maximum_items} éléments")
    clean = [
        _string(
            item,
            path=f"{path}[{index}]",
            minimum=1,
            maximum=maximum_length,
            identifier=identifiers,
        )
        for index, item in enumerate(value)
    ]
    if unique and len(set(clean)) != len(clean):
        _fail(path, "ne doit pas contenir de doublons")
    return clean


def validate_capsule(value: Mapping[str, Any] | str | bytes) -> dict[str, Any]:
    """Valide et retourne une copie JSON indépendante d'une capsule."""

    capsule = _json_object(value, label="$", max_bytes=MAX_CAPSULE_BYTES)
    _exact_object(
        capsule,
        path="$",
        required={"schema_version", "request_id", "question", "evidence", "constraints"},
    )
    if capsule["schema_version"] != CAPSULE_SCHEMA_VERSION:
        _fail("$.schema_version", f"doit valoir {CAPSULE_SCHEMA_VERSION}")
    _string(capsule["request_id"], path="$.request_id", minimum=1, maximum=128, identifier=True)
    _string(capsule["question"], path="$.question", minimum=1, maximum=8_000)

    evidence = capsule["evidence"]
    if not isinstance(evidence, list):
        _fail("$.evidence", "doit être une liste")
    if len(evidence) > MAX_EVIDENCE_ITEMS:
        _fail("$.evidence", f"ne peut pas dépasser {MAX_EVIDENCE_ITEMS} éléments")
    evidence_ids: list[str] = []
    for index, raw_item in enumerate(evidence):
        path = f"$.evidence[{index}]"
        item = _exact_object(
            raw_item,
            path=path,
            required={
                "evidence_id",
                "text",
                "space",
                "status",
                "confidence",
                "temporal_context",
                "tags",
            },
        )
        evidence_ids.append(
            _string(
                item["evidence_id"],
                path=f"{path}.evidence_id",
                minimum=1,
                maximum=128,
                identifier=True,
            )
        )
        _string(item["text"], path=f"{path}.text", minimum=1, maximum=4_000)
        if item["space"] not in _SPACES:
            _fail(f"{path}.space", "valeur inconnue")
        if item["status"] not in _EVIDENCE_STATUSES:
            _fail(f"{path}.status", "valeur inconnue")
        _number(item["confidence"], path=f"{path}.confidence")
        if item["temporal_context"] is not None:
            _string(
                item["temporal_context"],
                path=f"{path}.temporal_context",
                minimum=1,
                maximum=256,
            )
        _string_list(
            item["tags"],
            path=f"{path}.tags",
            maximum_items=16,
            maximum_length=64,
        )
    if len(set(evidence_ids)) != len(evidence_ids):
        _fail("$.evidence", "evidence_id doit être unique")

    constraints = _exact_object(
        capsule["constraints"],
        path="$.constraints",
        required={
            "evidence_required",
            "allow_calculations",
            "max_answer_characters",
            "max_evidence_ids",
            "max_calculations",
        },
    )
    _boolean(constraints["evidence_required"], path="$.constraints.evidence_required")
    _boolean(constraints["allow_calculations"], path="$.constraints.allow_calculations")
    _integer(
        constraints["max_answer_characters"],
        path="$.constraints.max_answer_characters",
        minimum=1,
        maximum=20_000,
    )
    _integer(
        constraints["max_evidence_ids"],
        path="$.constraints.max_evidence_ids",
        minimum=0,
        maximum=MAX_EVIDENCE_ITEMS,
    )
    _integer(
        constraints["max_calculations"],
        path="$.constraints.max_calculations",
        minimum=0,
        maximum=32,
    )
    return capsule


def validate_answer(
    value: Mapping[str, Any] | str | bytes,
    capsule: Mapping[str, Any] | str | bytes,
) -> dict[str, Any]:
    """Valide syntaxe, bornes et références d'une réponse du modèle."""

    trusted_capsule = validate_capsule(capsule)
    answer = _json_object(value, label="$", max_bytes=MAX_ANSWER_BYTES)
    _exact_object(
        answer,
        path="$",
        required={
            "schema_version",
            "request_id",
            "answer",
            "confidence",
            "evidence_ids",
            "calculations",
            "abstention",
        },
    )
    if answer["schema_version"] != ANSWER_SCHEMA_VERSION:
        _fail("$.schema_version", f"doit valoir {ANSWER_SCHEMA_VERSION}")
    request_id = _string(
        answer["request_id"], path="$.request_id", minimum=1, maximum=128, identifier=True
    )
    if request_id != trusted_capsule["request_id"]:
        _fail("$.request_id", "ne correspond pas à la capsule")
    constraints = trusted_capsule["constraints"]
    text = _string(
        answer["answer"],
        path="$.answer",
        minimum=0,
        maximum=constraints["max_answer_characters"],
    )
    confidence = _number(answer["confidence"], path="$.confidence")
    evidence_ids = _string_list(
        answer["evidence_ids"],
        path="$.evidence_ids",
        maximum_items=constraints["max_evidence_ids"],
        maximum_length=128,
        identifiers=True,
    )
    available_ids = {item["evidence_id"] for item in trusted_capsule["evidence"]}
    unknown_ids = sorted(set(evidence_ids) - available_ids)
    if unknown_ids:
        _fail("$.evidence_ids", "preuves absentes de la capsule: " + ", ".join(unknown_ids))

    calculations = answer["calculations"]
    if not isinstance(calculations, list):
        _fail("$.calculations", "doit être une liste")
    if len(calculations) > constraints["max_calculations"]:
        _fail(
            "$.calculations",
            f"ne peut pas dépasser {constraints['max_calculations']} éléments",
        )
    if calculations and not constraints["allow_calculations"]:
        _fail("$.calculations", "les calculs sont interdits par la capsule")
    calculation_ids: list[str] = []
    for index, raw_calculation in enumerate(calculations):
        path = f"$.calculations[{index}]"
        calculation = _exact_object(
            raw_calculation,
            path=path,
            required={
                "calculation_id",
                "expression",
                "reported_result",
                "unit",
                "evidence_ids",
            },
        )
        calculation_ids.append(
            _string(
                calculation["calculation_id"],
                path=f"{path}.calculation_id",
                minimum=1,
                maximum=128,
                identifier=True,
            )
        )
        _string(calculation["expression"], path=f"{path}.expression", minimum=1, maximum=512)
        # Cette valeur vient du modèle. La validation est uniquement structurelle;
        # seul le moteur mathématique borné peut ensuite confirmer le résultat.
        _string(
            calculation["reported_result"],
            path=f"{path}.reported_result",
            minimum=1,
            maximum=512,
        )
        if calculation["unit"] is not None:
            _string(calculation["unit"], path=f"{path}.unit", minimum=1, maximum=64)
        calculation_evidence = _string_list(
            calculation["evidence_ids"],
            path=f"{path}.evidence_ids",
            maximum_items=constraints["max_evidence_ids"],
            maximum_length=128,
            identifiers=True,
        )
        if not set(calculation_evidence) <= set(evidence_ids):
            _fail(f"{path}.evidence_ids", "doit être inclus dans $.evidence_ids")
    if len(set(calculation_ids)) != len(calculation_ids):
        _fail("$.calculations", "calculation_id doit être unique")

    abstention = _exact_object(
        answer["abstention"],
        path="$.abstention",
        required={"abstained", "reason", "missing_information"},
    )
    abstained = _boolean(abstention["abstained"], path="$.abstention.abstained")
    reason = abstention["reason"]
    if reason not in _ABSTENTION_REASONS:
        _fail("$.abstention.reason", "raison inconnue")
    missing = _string_list(
        abstention["missing_information"],
        path="$.abstention.missing_information",
        maximum_items=8,
        maximum_length=256,
    )
    if abstained:
        if reason == "none":
            _fail("$.abstention.reason", "une abstention exige une raison")
        if confidence != 0.0:
            _fail("$.confidence", "doit valoir 0 lors d'une abstention")
    else:
        if reason != "none":
            _fail("$.abstention.reason", "doit valoir none sans abstention")
        if missing:
            _fail("$.abstention.missing_information", "doit être vide sans abstention")
        if not text:
            _fail("$.answer", "ne peut pas être vide sans abstention")
        if constraints["evidence_required"] and not evidence_ids:
            _fail("$.evidence_ids", "au moins une preuve est requise")
    return answer


def build_capsule(
    *,
    request_id: str,
    question: str,
    evidence: Sequence[Mapping[str, Any]] = (),
    evidence_required: bool = True,
    allow_calculations: bool = True,
    max_answer_characters: int = 4_000,
    max_evidence_ids: int = 8,
    max_calculations: int = 4,
) -> dict[str, Any]:
    """Construit puis valide une capsule sans dépendre d'un moteur de modèle."""

    return validate_capsule(
        {
            "schema_version": CAPSULE_SCHEMA_VERSION,
            "request_id": request_id,
            "question": question,
            "evidence": list(evidence),
            "constraints": {
                "evidence_required": evidence_required,
                "allow_calculations": allow_calculations,
                "max_answer_characters": max_answer_characters,
                "max_evidence_ids": max_evidence_ids,
                "max_calculations": max_calculations,
            },
        }
    )


__all__ = [
    "ANSWER_JSON_SCHEMA",
    "ANSWER_SCHEMA_VERSION",
    "CAPSULE_JSON_SCHEMA",
    "CAPSULE_SCHEMA_VERSION",
    "ContractValidationError",
    "MODEL_ANSWER_JSON_SCHEMA",
    "MODEL_ANSWER_JSON_TEMPLATE",
    "build_capsule",
    "validate_answer",
    "validate_capsule",
]
