"""Build deterministic SFT examples for a model backed by external memory.

The builder deliberately projects the input onto ``sources``, ``entities`` and
``claims`` before generating anything.  Benchmark questions and answer keys
are accepted only as an ignored top-level compatibility field; their contents
are never inspected, validated, hashed or copied into training examples.

Every example teaches the same contract: the model receives a bounded,
facts-only capsule and must either ground its answer in claim/source IDs or
abstain.  Date arithmetic is represented as an explicit tool call so a model
does not have to learn calendar calculations in its weights.
"""

from __future__ import annotations

import calendar
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .matlm_bridge import strict_chat_messages
from .native_llm_contract import (
    ANSWER_SCHEMA_VERSION,
    CAPSULE_SCHEMA_VERSION,
    build_capsule as build_native_capsule,
    validate_answer as validate_native_answer,
)


CURRICULUM_SCHEMA = "memory-native-curriculum-v1"
EXAMPLE_SCHEMA = "memory-native-sft-example-v1"
CAPSULE_SCHEMA = "facts-only-capsule-v1"
ABSTENTION_MARKER = "JE_NE_SAIS_PAS"

TASK_ORDER = (
    "direct_recall_with_citations",
    "multi_hop_relation",
    "date_age_arithmetic",
    "role_distinction",
    "causal_uncertainty",
    "unsupported_abstention",
)

SYNTHETIC_TASK_ORDER = (
    "direct_recall_with_citations",
    "multi_hop_relation",
    "date_age_arithmetic",
    "role_distinction",
    "causal_uncertainty",
    "contradiction_resolution",
    "distractor_rejection",
    "unsupported_abstention",
    "provenance_selection",
)
SYNTHETIC_GENERATOR_SCHEMA = "memory-native-synthetic-generator-v8"
SYNTHETIC_TEMPLATE_VERSION = "memory-native-contract-templates-v8"

_MAX_INPUT_BYTES = 5_000_000
_MAX_ROWS = 50_000
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_YEAR_RE = re.compile(r"^(\d{4})$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_BENCHMARK_KEYS = frozenset(
    {
        "evaluation_questions",
        "expected_answer_fragments",
        "forbidden_answer_fragments",
        "supporting_claim_ids",
        "answer_status",
        "evaluation_kind",
    }
)
_SYSTEM_PROMPT = (
    "Tu es un modèle relié à une mémoire externe. Utilise uniquement la capsule "
    "fournie. Cite chaque fait avec [claim-id] puis ses [source-id]. Si la capsule "
    f"ne suffit pas, réponds {ABSTENTION_MARKER} sans inventer."
)
_UNSUPPORTED_REQUESTS = (
    ("favorite-color", "Quelle était la couleur préférée de {name} ?", "la couleur préférée"),
    ("shoe-size", "Quelle était la pointure de {name} ?", "la pointure"),
    ("breakfast", "Que mangeait {name} au petit déjeuner ?", "le petit déjeuner"),
    ("home-street", "Dans quelle rue privée habitait {name} ?", "l'adresse privée"),
    ("favorite-song", "Quelle était la chanson préférée de {name} ?", "la chanson préférée"),
)


class MemoryNativeCurriculumError(ValueError):
    """The facts-only training input violates its bounded schema."""


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MemoryNativeCurriculumError(f"clé JSON dupliquée: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise MemoryNativeCurriculumError(f"constante JSON interdite: {value}")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _text(value: Any, field: str, maximum: int = 4_000) -> str:
    if not isinstance(value, str):
        raise MemoryNativeCurriculumError(f"{field} doit être une chaîne")
    clean = value.strip()
    if not clean or len(clean) > maximum:
        raise MemoryNativeCurriculumError(f"{field} est vide ou trop long")
    return clean


def _identifier(value: Any, field: str) -> str:
    clean = _text(value, field, 128).replace("_", "-")
    if not _ID_RE.fullmatch(clean):
        raise MemoryNativeCurriculumError(f"{field} n'est pas un identifiant stable")
    return clean


def _rows(value: Any, field: str, maximum: int = _MAX_ROWS) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise MemoryNativeCurriculumError(f"{field} doit être une liste non vide et bornée")
    if not all(isinstance(row, Mapping) for row in value):
        raise MemoryNativeCurriculumError(f"{field} doit contenir uniquement des objets")
    return list(value)


def _unique_identifiers(rows: Sequence[Mapping[str, Any]], field: str) -> set[str]:
    found: set[str] = set()
    for index, row in enumerate(rows):
        identifier = _identifier(row.get("id"), f"{field}[{index}].id")
        if identifier in found:
            raise MemoryNativeCurriculumError(f"identifiant dupliqué: {identifier}")
        found.add(identifier)
    return found


def _source_ids(value: Any, field: str, known: set[str]) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 100:
        raise MemoryNativeCurriculumError(f"{field} doit être une liste non vide et bornée")
    identifiers = [_identifier(item, f"{field}[]") for item in value]
    if len(set(identifiers)) != len(identifiers):
        raise MemoryNativeCurriculumError(f"{field} contient un doublon")
    if not set(identifiers) <= known:
        raise MemoryNativeCurriculumError(f"{field} référence une source inconnue")
    return sorted(identifiers)


def load_facts_only_corpus(path: str | Path) -> dict[str, Any]:
    """Load only source/entity/claim material from a science corpus.

    ``evaluation_questions`` may be present for compatibility with the v1
    corpus, but this function intentionally does not index or validate its
    value.  Consequently, changing the complete benchmark payload cannot
    affect the returned projection or its fingerprint.
    """

    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise MemoryNativeCurriculumError(f"corpus introuvable: {corpus_path}")
    if corpus_path.stat().st_size > _MAX_INPUT_BYTES:
        raise MemoryNativeCurriculumError("le corpus dépasse 5 000 000 octets")
    try:
        document = json.loads(
            corpus_path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MemoryNativeCurriculumError("le corpus n'est pas un JSON UTF-8 valide") from error
    if not isinstance(document, Mapping):
        raise MemoryNativeCurriculumError("la racine du corpus doit être un objet")

    required = {"schema_version", "sources", "entities", "claims"}
    allowed = required | {"evaluation_questions"}
    if not required <= set(document) or not set(document) <= allowed:
        raise MemoryNativeCurriculumError("les champs racine du corpus sont incomplets ou inconnus")
    schema_version = _text(document["schema_version"], "schema_version", 100)
    if schema_version != "science-biographies-v1":
        raise MemoryNativeCurriculumError("schema_version non pris en charge")

    raw_sources = _rows(document["sources"], "sources", 1_000)
    raw_entities = _rows(document["entities"], "entities", 10_000)
    raw_claims = _rows(document["claims"], "claims", 50_000)
    source_ids = _unique_identifiers(raw_sources, "sources")
    entity_ids = _unique_identifiers(raw_entities, "entities")
    _unique_identifiers(raw_claims, "claims")

    sources: list[dict[str, str]] = []
    for index, source in enumerate(raw_sources):
        if set(source) != {"id", "title", "url", "publisher"}:
            raise MemoryNativeCurriculumError(f"sources[{index}] contient des champs invalides")
        url = _text(source["url"], f"sources[{index}].url", 2_048)
        if not url.startswith(("https://", "http://")):
            raise MemoryNativeCurriculumError(f"sources[{index}].url doit être HTTP(S)")
        sources.append(
            {
                "id": _identifier(source["id"], f"sources[{index}].id"),
                "title": _text(source["title"], f"sources[{index}].title", 500),
                "url": url,
                "publisher": _text(source["publisher"], f"sources[{index}].publisher", 500),
            }
        )

    entities: list[dict[str, str]] = []
    for index, entity in enumerate(raw_entities):
        if set(entity) != {"id", "type", "name", "dossier"}:
            raise MemoryNativeCurriculumError(f"entities[{index}] contient des champs invalides")
        entities.append(
            {
                "id": _identifier(entity["id"], f"entities[{index}].id"),
                "type": _identifier(entity["type"], f"entities[{index}].type"),
                "name": _text(entity["name"], f"entities[{index}].name", 500),
                "dossier": _identifier(entity["dossier"], f"entities[{index}].dossier"),
            }
        )

    claims: list[dict[str, Any]] = []
    required_claim = {"id", "subject", "predicate", "object", "date", "source_ids", "status"}
    for index, claim in enumerate(raw_claims):
        if not required_claim <= set(claim) or not set(claim) <= required_claim | {"statement_fr"}:
            raise MemoryNativeCurriculumError(f"claims[{index}] contient des champs invalides")
        subject = _identifier(claim["subject"], f"claims[{index}].subject")
        if subject not in entity_ids:
            raise MemoryNativeCurriculumError(f"claims[{index}] référence un sujet inconnu")
        raw_date = claim["date"]
        if not isinstance(raw_date, Mapping) or set(raw_date) != {"value", "precision"}:
            raise MemoryNativeCurriculumError(f"claims[{index}].date est invalide")
        clean: dict[str, Any] = {
            "id": _identifier(claim["id"], f"claims[{index}].id"),
            "subject": subject,
            "predicate": _identifier(claim["predicate"], f"claims[{index}].predicate"),
            "object": _text(claim["object"], f"claims[{index}].object"),
            "date": {
                "value": _text(raw_date["value"], f"claims[{index}].date.value", 100),
                "precision": _identifier(
                    raw_date["precision"], f"claims[{index}].date.precision"
                ),
            },
            "source_ids": _source_ids(
                claim["source_ids"], f"claims[{index}].source_ids", source_ids
            ),
            "status": _identifier(claim["status"], f"claims[{index}].status"),
        }
        if "statement_fr" in claim:
            clean["statement"] = _text(
                claim["statement_fr"], f"claims[{index}].statement_fr"
            )
        claims.append(clean)

    projection = {
        "schema_version": schema_version,
        "sources": sorted(sources, key=lambda row: row["id"]),
        "entities": sorted(entities, key=lambda row: row["id"]),
        "claims": sorted(claims, key=lambda row: row["id"]),
    }
    projection["facts_sha256"] = hashlib.sha256(
        _canonical(projection).encode("utf-8")
    ).hexdigest()
    return projection


def _claim_statement(claim: Mapping[str, Any], entities: Mapping[str, Mapping[str, str]]) -> str:
    if claim.get("statement"):
        return str(claim["statement"])
    subject = entities[str(claim["subject"])]["name"]
    raw_object = str(claim["object"])
    object_label = entities.get(raw_object, {}).get("name", raw_object)
    predicate = str(claim["predicate"]).replace("-", " ")
    return f"{subject} — {predicate} — {object_label}."


def _citations(claims: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str], str]:
    claim_ids = sorted({str(claim["id"]) for claim in claims})
    source_ids = sorted(
        {str(source_id) for claim in claims for source_id in claim["source_ids"]}
    )
    suffix = " ".join(f"[{identifier}]" for identifier in [*claim_ids, *source_ids])
    return claim_ids, source_ids, suffix


def _fact_row(claim: Mapping[str, Any], entities: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    raw_object = str(claim["object"])
    return {
        "claim_id": claim["id"],
        "subject_id": claim["subject"],
        "subject": entities[str(claim["subject"])]["name"],
        "predicate": claim["predicate"],
        "object": raw_object,
        "object_label": entities.get(raw_object, {}).get("name", raw_object),
        "statement": _claim_statement(claim, entities),
        "date": dict(claim["date"]),
        "status": claim["status"],
        "source_ids": list(claim["source_ids"]),
    }


def _capsule(
    claims: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    ordered_claims = sorted(claims, key=lambda row: str(row["id"]))
    source_ids = sorted(
        {str(source_id) for claim in ordered_claims for source_id in claim["source_ids"]}
    )
    body = {
        "schema_version": CAPSULE_SCHEMA,
        "facts": [_fact_row(claim, entities) for claim in ordered_claims],
        "sources": [dict(sources[source_id]) for source_id in source_ids],
    }
    body["capsule_sha256"] = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
    return body


def _user_message(capsule: Mapping[str, Any], question: str) -> str:
    return f"CAPSULE_MÉMOIRE={_canonical(capsule)}\nQUESTION={question}"


def _example(
    *,
    example_id: str,
    task: str,
    question: str,
    answer: str,
    claims: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, Mapping[str, str]],
    facts_sha256: str,
    grounding: str = "fully_supported",
    messages: list[dict[str, Any]] | None = None,
    calculation: Mapping[str, Any] | None = None,
    citation_claims: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    memory_capsule = _capsule(claims, entities, sources)
    cited = claims if citation_claims is None else citation_claims
    claim_ids, source_ids, _ = _citations(cited)
    context_claim_ids = sorted(str(claim["id"]) for claim in claims)
    conversation = messages or [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_message(memory_capsule, question)},
        {"role": "assistant", "content": answer},
    ]
    target: dict[str, Any] = {
        "answer": answer,
        "grounding": grounding,
        "citations": {"claim_ids": claim_ids, "source_ids": source_ids},
    }
    if calculation is not None:
        target["calculation"] = dict(calculation)
    return {
        "schema_version": EXAMPLE_SCHEMA,
        "example_id": example_id,
        "task": task,
        "memory_capsule": memory_capsule,
        "messages": conversation,
        "target": target,
        "provenance": {
            "facts_sha256": facts_sha256,
            "claim_ids": claim_ids,
            "context_claim_ids": context_claim_ids,
            "template_version": "memory-native-templates-v1",
        },
    }


def _date_interval(value: str, precision: str) -> tuple[date, date] | None:
    try:
        if precision == "day":
            match = _DAY_RE.fullmatch(value)
            if match:
                parsed = date(*(int(part) for part in match.groups()))
                return parsed, parsed
        if precision == "month":
            match = _MONTH_RE.fullmatch(value)
            if match:
                year, month = (int(part) for part in match.groups())
                return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
        if precision == "year":
            match = _YEAR_RE.fullmatch(value)
            if match:
                year = int(match.group(1))
                return date(year, 1, 1), date(year, 12, 31)
    except ValueError:
        return None
    return None


def _age_on(birth: date, event: date) -> int:
    return event.year - birth.year - ((event.month, event.day) < (birth.month, birth.day))


def _age_calculation(birth_claim: Mapping[str, Any], event_claim: Mapping[str, Any]) -> dict[str, Any] | None:
    birth_interval = _date_interval(
        str(birth_claim["date"]["value"]), str(birth_claim["date"]["precision"])
    )
    event_interval = _date_interval(
        str(event_claim["date"]["value"]), str(event_claim["date"]["precision"])
    )
    if birth_interval is None or event_interval is None or birth_interval[0] != birth_interval[1]:
        return None
    birth = birth_interval[0]
    if event_interval[0] < birth:
        return None
    minimum = _age_on(birth, event_interval[0])
    maximum = _age_on(birth, event_interval[1])
    result: dict[str, Any] = {
        "operation": "calendar_age",
        "birth_date": birth.isoformat(),
        "event_date": str(event_claim["date"]["value"]),
        "event_precision": str(event_claim["date"]["precision"]),
        "formula": "event_year - birth_year - birthday_not_reached",
    }
    if minimum == maximum:
        result.update({"result_kind": "exact", "years": minimum})
    else:
        result.update(
            {"result_kind": "range", "minimum_years": minimum, "maximum_years": maximum}
        )
    return result


def _direct_examples(
    claims: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, Mapping[str, str]],
    facts_sha256: str,
) -> list[dict[str, Any]]:
    examples = []
    for claim in claims:
        subject = entities[str(claim["subject"])]["name"]
        predicate = str(claim["predicate"]).replace("-", " ")
        _, _, citation_text = _citations([claim])
        question = f"Quel fait la mémoire documente-t-elle pour {subject} sous la relation « {predicate} » ?"
        answer = f"{_claim_statement(claim, entities)} {citation_text}"
        examples.append(
            _example(
                example_id=f"direct-recall--{claim['id']}",
                task=TASK_ORDER[0],
                question=question,
                answer=answer,
                claims=[claim],
                entities=entities,
                sources=sources,
                facts_sha256=facts_sha256,
            )
        )
    return examples


def _multi_hop_examples(
    claims: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, Mapping[str, str]],
    facts_sha256: str,
) -> list[dict[str, Any]]:
    by_subject: dict[str, list[Mapping[str, Any]]] = {}
    for claim in claims:
        by_subject.setdefault(str(claim["subject"]), []).append(claim)
    examples = []
    for first in claims:
        middle_id = str(first["object"])
        if middle_id not in entities:
            continue
        for second in sorted(by_subject.get(middle_id, []), key=lambda row: str(row["id"])):
            if second["id"] == first["id"] or str(second["object"]) == str(first["subject"]):
                continue
            first_entity = entities[str(first["subject"])]["name"]
            middle_entity = entities[middle_id]["name"]
            raw_last = str(second["object"])
            last = entities.get(raw_last, {}).get("name", raw_last)
            selected = [first, second]
            _, _, citation_text = _citations(selected)
            question = (
                f"Relie {first_entity}, {middle_entity} et {last} en exactement deux "
                "relations documentées par la capsule."
            )
            answer = (
                f"Étape 1 : {_claim_statement(first, entities)} "
                f"Étape 2 : {_claim_statement(second, entities)} {citation_text}"
            )
            examples.append(
                _example(
                    example_id=f"multi-hop--{first['id']}--{second['id']}",
                    task=TASK_ORDER[1],
                    question=question,
                    answer=answer,
                    claims=selected,
                    entities=entities,
                    sources=sources,
                    facts_sha256=facts_sha256,
                )
            )
    return examples


def _age_examples(
    claims: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, Mapping[str, str]],
    facts_sha256: str,
) -> list[dict[str, Any]]:
    births = {
        str(claim["subject"]): claim
        for claim in claims
        if claim["predicate"] == "born-on"
    }
    examples = []
    for event in claims:
        birth = births.get(str(event["subject"]))
        if birth is None or event["id"] == birth["id"]:
            continue
        calculation = _age_calculation(birth, event)
        if calculation is None:
            continue
        person = entities[str(event["subject"])]["name"]
        selected = [birth, event]
        _, _, citation_text = _citations(selected)
        if calculation["result_kind"] == "exact":
            result_text = f"{calculation['years']} ans"
        else:
            result_text = (
                f"entre {calculation['minimum_years']} et {calculation['maximum_years']} ans"
            )
        question = (
            f"À partir des deux dates de la capsule, calcule l'âge civil de {person} "
            f"au moment du fait {event['id']}. Appelle l'outil calendar_age."
        )
        answer = (
            f"Le calcul civil donne {result_text}. La précision suit celle de la date "
            f"du fait. {citation_text}"
        )
        capsule = _capsule(selected, entities, sources)
        call_id = f"calendar-age--{birth['id']}--{event['id']}"
        arguments = {
            "birth_date": calculation["birth_date"],
            "event_date": calculation["event_date"],
            "event_precision": calculation["event_precision"],
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(capsule, question)},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "calendar_age",
                            "arguments": _canonical(arguments),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "calendar_age",
                "content": _canonical(calculation),
            },
            {"role": "assistant", "content": answer},
        ]
        examples.append(
            _example(
                example_id=f"date-age--{birth['id']}--{event['id']}",
                task=TASK_ORDER[2],
                question=question,
                answer=answer,
                claims=selected,
                entities=entities,
                sources=sources,
                facts_sha256=facts_sha256,
                messages=messages,
                calculation=calculation,
            )
        )
    return examples


def _role_examples(
    claims: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, Mapping[str, str]],
    facts_sha256: str,
) -> list[dict[str, Any]]:
    by_dossier: dict[str, list[Mapping[str, Any]]] = {}
    for claim in claims:
        if "role" in str(claim["status"]):
            dossier = entities[str(claim["subject"])]["dossier"]
            by_dossier.setdefault(dossier, []).append(claim)
    examples = []
    for dossier, dossier_claims in sorted(by_dossier.items()):
        subjects = sorted({str(claim["subject"]) for claim in dossier_claims})
        if len(subjects) < 2:
            continue
        selected = sorted(dossier_claims, key=lambda row: str(row["id"]))
        names = [entities[subject]["name"] for subject in subjects]
        question = (
            f"Distingue les rôles documentés de {', '.join(names)} sans attribuer à l'un "
            "le travail d'un autre."
        )
        parts = []
        for subject in subjects:
            subject_claims = [claim for claim in selected if claim["subject"] == subject]
            _, _, subject_citations = _citations(subject_claims)
            statements = " ".join(_claim_statement(claim, entities) for claim in subject_claims)
            parts.append(f"{entities[subject]['name']} : {statements} {subject_citations}")
        examples.append(
            _example(
                example_id=f"role-distinction--{dossier}",
                task=TASK_ORDER[3],
                question=question,
                answer=" ".join(parts),
                claims=selected,
                entities=entities,
                sources=sources,
                facts_sha256=facts_sha256,
            )
        )
    return examples


def _causal_examples(
    claims: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, Mapping[str, str]],
    facts_sha256: str,
) -> list[dict[str, Any]]:
    examples = []
    for claim in claims:
        status = str(claim["status"])
        if not any(marker in status for marker in ("not-established", "nonexclusive", "not-exclusive")):
            continue
        subject = entities[str(claim["subject"])]["name"]
        _, _, citation_text = _citations([claim])
        if "not-established" in status:
            limit = "Non. La mémoire signale une possibilité chronologique, pas une cause certaine."
        else:
            limit = "Non. La mémoire documente une contribution ou conséquence non exclusive."
        answer = f"{limit} {_claim_statement(claim, entities)} {citation_text}"
        examples.append(
            _example(
                example_id=f"causal-uncertainty--{claim['id']}",
                task=TASK_ORDER[4],
                question=(
                    f"La capsule autorise-t-elle à présenter le rôle de {subject} dans ce "
                    "fait comme une cause certaine, unique et exclusive ?"
                ),
                answer=answer,
                claims=[claim],
                entities=entities,
                sources=sources,
                facts_sha256=facts_sha256,
                grounding="supported_with_causal_limit",
            )
        )
    return examples


def _abstention_examples(
    claims: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, Mapping[str, str]],
    facts_sha256: str,
) -> list[dict[str, Any]]:
    claims_by_dossier: dict[str, list[Mapping[str, Any]]] = {}
    persons_by_dossier: dict[str, list[Mapping[str, str]]] = {}
    for claim in claims:
        dossier = entities[str(claim["subject"])]["dossier"]
        claims_by_dossier.setdefault(dossier, []).append(claim)
    for entity in entities.values():
        if entity["type"] == "person":
            persons_by_dossier.setdefault(entity["dossier"], []).append(entity)

    examples = []
    for index, dossier in enumerate(sorted(persons_by_dossier)):
        people = sorted(persons_by_dossier[dossier], key=lambda row: row["id"])
        person = people[0]
        predicate, template, missing_label = _UNSUPPORTED_REQUESTS[index % len(_UNSUPPORTED_REQUESTS)]
        selected = sorted(
            [claim for claim in claims_by_dossier.get(dossier, []) if claim["subject"] == person["id"]],
            key=lambda row: str(row["id"]),
        )[:3]
        if not selected:
            selected = sorted(claims_by_dossier[dossier], key=lambda row: str(row["id"]))[:3]
        question = template.format(name=person["name"])
        answer = (
            f"{ABSTENTION_MARKER} : la capsule ne documente pas {missing_label} "
            f"de {person['name']}."
        )
        example = _example(
            example_id=f"unsupported-abstention--{dossier}--{predicate}",
            task=TASK_ORDER[5],
            question=question,
            answer=answer,
            claims=selected,
            entities=entities,
            sources=sources,
            facts_sha256=facts_sha256,
            grounding="unsupported",
        )
        example["target"]["citations"] = {"claim_ids": [], "source_ids": []}
        example["provenance"]["unsupported_relation"] = predicate
        examples.append(example)
    return examples


def build_memory_native_curriculum(
    corpus_path: str | Path,
    *,
    maximum_per_task: int | None = None,
) -> dict[str, Any]:
    """Create a deterministic curriculum without consulting benchmark data."""

    if maximum_per_task is not None and (
        isinstance(maximum_per_task, bool)
        or not isinstance(maximum_per_task, int)
        or not 1 <= maximum_per_task <= 10_000
    ):
        raise ValueError("maximum_per_task doit être compris entre 1 et 10000")

    corpus = load_facts_only_corpus(corpus_path)
    entities = {row["id"]: row for row in corpus["entities"]}
    sources = {row["id"]: row for row in corpus["sources"]}
    claims = list(corpus["claims"])
    builders = (
        _direct_examples,
        _multi_hop_examples,
        _age_examples,
        _role_examples,
        _causal_examples,
        _abstention_examples,
    )
    examples: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    for task, builder in zip(TASK_ORDER, builders, strict=True):
        task_examples = sorted(
            builder(claims, entities, sources, corpus["facts_sha256"]),
            key=lambda row: row["example_id"],
        )
        if maximum_per_task is not None:
            task_examples = task_examples[:maximum_per_task]
        if not task_examples:
            raise MemoryNativeCurriculumError(f"aucun exemple généré pour la tâche {task}")
        task_counts[task] = len(task_examples)
        examples.extend(task_examples)

    result = {
        "schema_version": CURRICULUM_SCHEMA,
        "source_schema_version": corpus["schema_version"],
        "facts_sha256": corpus["facts_sha256"],
        "training_contract": {
            "input_projection": ["schema_version", "sources", "entities", "claims"],
            "benchmark_payload": "ignored",
            "answer_policy": "cite_claims_and_sources_or_abstain",
            "calculation_policy": "explicit_tool_call",
        },
        "synthetic": False,
        "evaluation_eligibility": "invalid_against_the_source_corpus",
        "warning": (
            "Démonstrateur de format seulement : ces exemples reprennent les faits "
            "du corpus et ne doivent jamais entraîner un modèle évalué sur ce corpus."
        ),
        "task_counts": task_counts,
        "example_count": len(examples),
        "examples": examples,
    }
    encoded = _canonical(result)
    for forbidden_key in _BENCHMARK_KEYS:
        if f'"{forbidden_key}"' in encoded:
            raise AssertionError(f"fuite d'un champ d'évaluation: {forbidden_key}")
    return result


def _derived_number(seed: int, index: int, label: str, modulus: int) -> int:
    payload = f"{SYNTHETIC_GENERATOR_SCHEMA}\0{seed}\0{index}\0{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus


def _synthetic_tag(seed: int, index: int) -> str:
    payload = f"{SYNTHETIC_GENERATOR_SCHEMA}\0{seed}\0{index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _synthetic_source(tag: str, suffix: str) -> dict[str, str]:
    return {
        "id": f"syn-source-{tag}-{suffix}",
        "title": f"Archive fictive SYN-{tag.upper()}-{suffix.upper()}",
        "url": f"https://memory-{tag}.example.invalid/{suffix}",
        "publisher": f"Institut fictif SYN-{tag.upper()}",
    }


def _synthetic_claim(
    tag: str,
    suffix: str,
    *,
    subject: str,
    predicate: str,
    object_value: str,
    date_value: str,
    precision: str,
    source_suffix: str,
    status: str,
    statement: str,
) -> dict[str, Any]:
    return {
        "id": f"syn-claim-{tag}-{suffix}",
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "date": {"value": date_value, "precision": precision},
        "source_ids": [f"syn-source-{tag}-{source_suffix}"],
        "status": status,
        "statement": statement,
    }


def _synthetic_world(seed: int, index: int) -> tuple[
    str,
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    str,
]:
    tag = _synthetic_tag(seed, index)
    dossier = f"syn-dossier-{tag}"
    raw_entities = (
        ("person-a", "person", f"Personne fictive SYN-{tag.upper()}-A"),
        ("person-b", "person", f"Personne fictive SYN-{tag.upper()}-B"),
        ("person-c", "person", f"Personne fictive SYN-{tag.upper()}-C"),
        ("artifact", "artifact", f"Objet fictif SYN-{tag.upper()}-X"),
        ("location", "location", f"Lieu fictif SYN-{tag.upper()}-L"),
        ("signal", "signal", f"Signal fictif SYN-{tag.upper()}-Q"),
    )
    entities = {
        f"syn-{kind}-{tag}": {
            "id": f"syn-{kind}-{tag}",
            "type": entity_type,
            "name": name,
            "dossier": dossier,
        }
        for kind, entity_type, name in raw_entities
    }
    sources = {
        row["id"]: row
        for row in (
            _synthetic_source(tag, "a"),
            _synthetic_source(tag, "b"),
            _synthetic_source(tag, "c"),
        )
    }
    return tag, entities, sources, dossier


def _synthetic_date(seed: int, index: int, offset: int = 0) -> str:
    year = 2110 + _derived_number(seed, index, "year", 180) + offset
    month = 1 + _derived_number(seed, index, f"month-{offset}", 12)
    day = 1 + _derived_number(seed, index, f"day-{offset}", 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _synthetic_example(
    seed: int,
    index: int,
    task: str,
    generator_sha256: str,
) -> dict[str, Any]:
    tag, entities, sources, dossier = _synthetic_world(seed, index)
    pa = entities[f"syn-person-a-{tag}"]
    pb = entities[f"syn-person-b-{tag}"]
    pc = entities[f"syn-person-c-{tag}"]
    artifact = entities[f"syn-artifact-{tag}"]
    location = entities[f"syn-location-{tag}"]
    signal = entities[f"syn-signal-{tag}"]
    first_date = _synthetic_date(seed, index)
    next_date = _synthetic_date(seed, index, 1)
    prefix = f"synthetic-{index:06d}-{tag}"

    if task == "direct_recall_with_citations":
        state = f"état-fictif-{_derived_number(seed, index, 'state', 10_000):04d}"
        claim = _synthetic_claim(
            tag,
            "a",
            subject=pa["id"],
            predicate="recorded-state",
            object_value=state,
            date_value=first_date,
            precision="day",
            source_suffix="a",
            status="documented",
            statement=f"Le registre fictif associe {pa['name']} à {state} le {first_date}.",
        )
        _, _, cites = _citations([claim])
        return _example(
            example_id=f"{prefix}--direct",
            task=task,
            question=f"Quel état le registre attribue-t-il à {pa['name']} ?",
            answer=f"Le registre attribue {state} à {pa['name']}. {cites}",
            claims=[claim],
            entities=entities,
            sources=sources,
            facts_sha256=generator_sha256,
        )

    if task == "multi_hop_relation":
        first = _synthetic_claim(
            tag,
            "a",
            subject=pa["id"],
            predicate="transferred-responsibility-to",
            object_value=pb["id"],
            date_value=first_date,
            precision="day",
            source_suffix="a",
            status="documented",
            statement=f"{pa['name']} a transmis la responsabilité de {artifact['name']} à {pb['name']}.",
        )
        second = _synthetic_claim(
            tag,
            "b",
            subject=pb["id"],
            predicate="stored-at",
            object_value=location["id"],
            date_value=next_date,
            precision="day",
            source_suffix="b",
            status="documented",
            statement=f"{pb['name']} a ensuite placé {artifact['name']} dans {location['name']}.",
        )
        _, _, cites = _citations([first, second])
        return _example(
            example_id=f"{prefix}--multi-hop",
            task=task,
            question=(
                f"Relie {pa['name']} à {location['name']} en passant par {pb['name']} "
                "et conserve les deux étapes."
            ),
            answer=(
                f"Étape 1 : {pa['name']} a transmis la responsabilité de "
                f"{artifact['name']} à {pb['name']}. Étape 2 : {pb['name']} "
                f"l'a placé dans {location['name']}. {cites}"
            ),
            claims=[first, second],
            entities=entities,
            sources=sources,
            facts_sha256=generator_sha256,
        )

    if task == "date_age_arithmetic":
        birth_year = 2040 + _derived_number(seed, index, "birth-year", 45)
        birth_month = 1 + _derived_number(seed, index, "birth-month", 12)
        birth_day = 1 + _derived_number(seed, index, "birth-day", 28)
        birth_date = f"{birth_year:04d}-{birth_month:02d}-{birth_day:02d}"
        event_year = birth_year + 20 + _derived_number(seed, index, "age", 45)
        if index % 2:
            event_value = f"{event_year:04d}"
            event_precision = "year"
            event_wording = f"au cours de l'année fictive {event_year}"
        else:
            event_month = 1 + _derived_number(seed, index, "event-month", 12)
            event_day = 1 + _derived_number(seed, index, "event-day", 28)
            event_value = f"{event_year:04d}-{event_month:02d}-{event_day:02d}"
            event_precision = "day"
            event_wording = f"le {event_value}"
        birth = _synthetic_claim(
            tag,
            "a",
            subject=pa["id"],
            predicate="born-on",
            object_value=birth_date,
            date_value=birth_date,
            precision="day",
            source_suffix="a",
            status="documented-synthetic",
            statement=f"{pa['name']} est née dans le scénario fictif le {birth_date}.",
        )
        event = _synthetic_claim(
            tag,
            "b",
            subject=pa["id"],
            predicate="completed-artifact",
            object_value=artifact["id"],
            date_value=event_value,
            precision=event_precision,
            source_suffix="b",
            status="documented-synthetic",
            statement=f"{pa['name']} a terminé {artifact['name']} {event_wording}.",
        )
        calculation = _age_calculation(birth, event)
        if calculation is None:  # pragma: no cover - guarded by the generated date ranges
            raise AssertionError("dates synthétiques non calculables")
        if calculation["result_kind"] == "exact":
            age_text = f"{calculation['years']} ans"
        else:
            age_text = (
                f"entre {calculation['minimum_years']} et {calculation['maximum_years']} ans"
            )
        selected = [birth, event]
        capsule = _capsule(selected, entities, sources)
        _, _, cites = _citations(selected)
        question = (
            f"Calcule l'âge civil de {pa['name']} lors de l'achèvement fictif. "
            "Appelle calendar_age."
        )
        answer = (
            f"Pour {pa['name']} à la date fictive {event_value}, le calcul donne {age_text}, "
            f"selon la précision disponible. {cites}"
        )
        call_id = f"calendar-age--{tag}"
        arguments = {
            "birth_date": birth_date,
            "event_date": event_value,
            "event_precision": event_precision,
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(capsule, question)},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "calendar_age",
                            "arguments": _canonical(arguments),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "calendar_age",
                "content": _canonical(calculation),
            },
            {"role": "assistant", "content": answer},
        ]
        return _example(
            example_id=f"{prefix}--date-age",
            task=task,
            question=question,
            answer=answer,
            claims=selected,
            entities=entities,
            sources=sources,
            facts_sha256=generator_sha256,
            messages=messages,
            calculation=calculation,
        )

    if task == "role_distinction":
        roles = (
            ("a", pa, "designed", "documented-design-role", "a", "a conçu"),
            ("b", pb, "verified", "documented-verification-role", "b", "a vérifié"),
            ("c", pc, "deployed", "documented-deployment-role", "c", "a déployé"),
        )
        claims = [
            _synthetic_claim(
                tag,
                suffix,
                subject=person["id"],
                predicate=predicate,
                object_value=artifact["id"],
                date_value=first_date,
                precision="day",
                source_suffix=source_suffix,
                status=status,
                statement=f"{person['name']} {wording} {artifact['name']} dans le scénario fictif.",
            )
            for suffix, person, predicate, status, source_suffix, wording in roles
        ]
        _, _, cites = _citations(claims)
        answer = (
            f"Conçu par {pa['name']}; vérifié par {pb['name']}; "
            f"déployé par {pc['name']}. {cites}"
        )
        return _example(
            example_id=f"{prefix}--roles",
            task=task,
            question="Distingue précisément conception, vérification et déploiement sans fusionner les rôles.",
            answer=answer,
            claims=claims,
            entities=entities,
            sources=sources,
            facts_sha256=generator_sha256,
        )

    if task == "causal_uncertainty":
        chronology = _synthetic_claim(
            tag,
            "a",
            subject=signal["id"],
            predicate="preceded",
            object_value=artifact["id"],
            date_value=first_date,
            precision="day",
            source_suffix="a",
            status="documented-chronology",
            statement=f"{signal['name']} a précédé l'activation de {artifact['name']}.",
        )
        possible = _synthetic_claim(
            tag,
            "b",
            subject=signal["id"],
            predicate="possibly-encouraged",
            object_value=artifact["id"],
            date_value=next_date,
            precision="day",
            source_suffix="b",
            status="possible-not-established-causality",
            statement=(
                f"Le dossier fictif envisage que {signal['name']} ait favorisé {artifact['name']}, "
                "sans établir une cause unique."
            ),
        )
        selected = [chronology, possible]
        _, _, cites = _citations(selected)
        return _example(
            example_id=f"{prefix}--causal-limit",
            task=task,
            question="La succession temporelle suffit-elle à prouver une cause certaine et exclusive ?",
            answer=(
                f"Non. Pour {signal['name']} et {artifact['name']}, l'ordre temporel est "
                "documenté, mais le lien causal reste possible et non établi comme unique. "
                f"{cites}"
            ),
            claims=selected,
            entities=entities,
            sources=sources,
            facts_sha256=generator_sha256,
            grounding="supported_with_causal_limit",
        )

    if task == "contradiction_resolution":
        old_state = f"état-fictif-{_derived_number(seed, index, 'old-state', 10_000):04d}"
        new_state = f"état-fictif-{_derived_number(seed, index, 'new-state', 10_000):04d}"
        initial = _synthetic_claim(
            tag,
            "a",
            subject=artifact["id"],
            predicate="recorded-state",
            object_value=old_state,
            date_value=first_date,
            precision="day",
            source_suffix="a",
            status="superseded-report",
            statement=f"Un relevé fictif initial attribuait {old_state} à {artifact['name']}.",
        )
        correction = _synthetic_claim(
            tag,
            "b",
            subject=artifact["id"],
            predicate="recorded-state",
            object_value=new_state,
            date_value=next_date,
            precision="day",
            source_suffix="b",
            status="documented-correction",
            statement=f"Une correction fictive documentée attribue {new_state} à {artifact['name']}.",
        )
        selected = [initial, correction]
        _, _, cites = _citations(selected)
        return _example(
            example_id=f"{prefix}--contradiction",
            task=task,
            question="Les deux états se contredisent. Lequel rapporter et comment conserver la trace ?",
            answer=(
                f"Pour {artifact['name']}, je signale les deux versions et retiens {new_state}, "
                f"explicitement marqué comme correction; {old_state} reste une version "
                f"remplacée. {cites}"
            ),
            claims=selected,
            entities=entities,
            sources=sources,
            facts_sha256=generator_sha256,
            grounding="resolved_documented_correction",
        )

    if task == "distractor_rejection":
        code = f"code-fictif-{_derived_number(seed, index, 'code', 10_000):04d}"
        relevant = _synthetic_claim(
            tag,
            "a",
            subject=pa["id"],
            predicate="assigned-code",
            object_value=code,
            date_value=first_date,
            precision="day",
            source_suffix="a",
            status="documented",
            statement=f"{pa['name']} a reçu le {code}.",
        )
        distractor_b = _synthetic_claim(
            tag,
            "b",
            subject=pb["id"],
            predicate="visited",
            object_value=location["id"],
            date_value=next_date,
            precision="day",
            source_suffix="b",
            status="documented",
            statement=f"{pb['name']} a visité {location['name']}.",
        )
        distractor_c = _synthetic_claim(
            tag,
            "c",
            subject=pc["id"],
            predicate="inspected",
            object_value=artifact["id"],
            date_value=next_date,
            precision="day",
            source_suffix="c",
            status="documented",
            statement=f"{pc['name']} a inspecté {artifact['name']}.",
        )
        _, _, cites = _citations([relevant])
        return _example(
            example_id=f"{prefix}--distractor",
            task=task,
            question=f"Quel code a été attribué à {pa['name']} ? Ignore les faits sans rapport.",
            answer=f"Pour {pa['name']}, le code attribué est {code}. {cites}",
            claims=[relevant, distractor_b, distractor_c],
            citation_claims=[relevant],
            entities=entities,
            sources=sources,
            facts_sha256=generator_sha256,
        )

    if task == "unsupported_abstention":
        first = _synthetic_claim(
            tag,
            "a",
            subject=pa["id"],
            predicate="inspected",
            object_value=artifact["id"],
            date_value=first_date,
            precision="day",
            source_suffix="a",
            status="documented",
            statement=f"{pa['name']} a inspecté {artifact['name']}.",
        )
        second = _synthetic_claim(
            tag,
            "b",
            subject=pb["id"],
            predicate="visited",
            object_value=location["id"],
            date_value=next_date,
            precision="day",
            source_suffix="b",
            status="documented",
            statement=f"{pb['name']} a visité {location['name']}.",
        )
        return _example(
            example_id=f"{prefix}--abstention",
            task=task,
            question=f"Quel instrument de musique préférait {pa['name']} ?",
            answer=(
                f"{ABSTENTION_MARKER} : la capsule ne documente aucun instrument préféré "
                f"pour {pa['name']}."
            ),
            claims=[first, second],
            citation_claims=[],
            entities=entities,
            sources=sources,
            facts_sha256=generator_sha256,
            grounding="unsupported",
        )

    if task == "provenance_selection":
        documented_code = f"code-fictif-{_derived_number(seed, index, 'doc-code', 10_000):04d}"
        unverified_code = f"code-fictif-{_derived_number(seed, index, 'rumor-code', 10_000):04d}"
        documented = _synthetic_claim(
            tag,
            "a",
            subject=artifact["id"],
            predicate="assigned-code",
            object_value=documented_code,
            date_value=first_date,
            precision="day",
            source_suffix="a",
            status="documented-primary-record",
            statement=f"L'archive fictive principale donne {documented_code} à {artifact['name']}.",
        )
        unverified = _synthetic_claim(
            tag,
            "b",
            subject=artifact["id"],
            predicate="assigned-code",
            object_value=unverified_code,
            date_value=first_date,
            precision="day",
            source_suffix="b",
            status="unverified-secondary-report",
            statement=f"Un relevé fictif secondaire non vérifié donne {unverified_code} à {artifact['name']}.",
        )
        selected = [documented, unverified]
        _, _, cites = _citations(selected)
        return _example(
            example_id=f"{prefix}--provenance",
            task=task,
            question="Deux sources proposent des codes différents. Quelle version est la mieux étayée ?",
            answer=(
                f"Pour {artifact['name']}, je privilégie {documented_code}, issu du relevé "
                f"marqué documenté et principal. Je conserve {unverified_code} comme "
                f"désaccord non vérifié. {cites}"
            ),
            claims=selected,
            entities=entities,
            sources=sources,
            facts_sha256=generator_sha256,
            grounding="provenance_ranked",
        )

    raise AssertionError(f"tâche synthétique inconnue: {task}")


def _opaque_evidence_id(request_id: str, claim_id: str) -> str:
    digest = hashlib.sha256(f"{request_id}\0{claim_id}".encode("utf-8")).hexdigest()[:20]
    return f"ev:{digest}"


def _native_evidence_status(raw_status: str) -> tuple[str, float]:
    if any(marker in raw_status for marker in ("unverified", "superseded", "possible")):
        return "unverified", 0.4
    if "derived" in raw_status:
        return "derived", 0.85
    return "verified", 1.0


def _without_legacy_citations(answer: str) -> str:
    return re.sub(
        r"\s*\[syn-(?:claim|source)-[^\]]+\]",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()


def _align_synthetic_example_with_native_contract(
    legacy: Mapping[str, Any],
    generator_sha256: str,
) -> dict[str, Any]:
    """Convert an internal template row into the exact MAT-LM public contract."""

    request_id = str(legacy["example_id"])
    task = str(legacy["task"])
    user_messages = [
        message
        for message in legacy["messages"]
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]
    if len(user_messages) != 1 or "\nQUESTION=" not in str(user_messages[0].get("content", "")):
        raise AssertionError("question synthétique interne introuvable")
    question = str(user_messages[0]["content"]).split("\nQUESTION=", 1)[1]

    legacy_facts = list(legacy["memory_capsule"]["facts"])
    evidence: list[dict[str, Any]] = []
    claim_to_evidence: dict[str, str] = {}
    for index, fact in enumerate(legacy_facts):
        claim_id = str(fact["claim_id"])
        evidence_id = _opaque_evidence_id(request_id, claim_id)
        claim_to_evidence[claim_id] = evidence_id
        status, confidence = _native_evidence_status(str(fact["status"]))
        temporal = fact.get("date", {}).get("value")
        evidence.append(
            {
                "evidence_id": evidence_id,
                "text": str(fact["statement"]),
                "space": "reference",
                "status": status,
                "confidence": confidence,
                "temporal_context": str(temporal) if temporal else None,
                "tags": ["synthetic", task, f"fact-{index:02d}"],
            }
        )

    calculation_source = legacy["target"].get("calculation")
    allow_calculations = calculation_source is not None
    capsule = build_native_capsule(
        request_id=request_id,
        question=question,
        evidence=evidence,
        evidence_required=True,
        allow_calculations=allow_calculations,
        max_answer_characters=4_000,
        max_evidence_ids=len(evidence),
        max_calculations=1 if allow_calculations else 0,
    )

    cited_claim_ids = list(legacy["target"]["citations"]["claim_ids"])
    cited_evidence_ids = [claim_to_evidence[claim_id] for claim_id in cited_claim_ids]
    grounding = str(legacy["target"]["grounding"])
    abstained = grounding == "unsupported"
    calculations: list[dict[str, Any]] = []
    if calculation_source is not None:
        if calculation_source["result_kind"] == "exact":
            reported_result = str(calculation_source["years"])
        else:
            reported_result = (
                f"{calculation_source['minimum_years']}.."
                f"{calculation_source['maximum_years']}"
            )
        calculations.append(
            {
                "calculation_id": f"calc:{hashlib.sha256(request_id.encode('utf-8')).hexdigest()[:20]}",
                "expression": (
                    f"calendar_age({calculation_source['birth_date']},"
                    f"{calculation_source['event_date']},"
                    f"precision={calculation_source['event_precision']})"
                ),
                "reported_result": reported_result,
                "unit": "ans",
                "evidence_ids": cited_evidence_ids,
            }
        )

    confidence_by_grounding = {
        "fully_supported": 0.98,
        "supported_with_causal_limit": 0.82,
        "resolved_documented_correction": 0.92,
        "provenance_ranked": 0.9,
    }
    answer = {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "request_id": request_id,
        "answer": _without_legacy_citations(str(legacy["target"]["answer"])),
        "confidence": 0.0 if abstained else confidence_by_grounding.get(grounding, 0.95),
        "evidence_ids": [] if abstained else cited_evidence_ids,
        "calculations": calculations,
        "abstention": {
            "abstained": abstained,
            "reason": "insufficient_evidence" if abstained else "none",
            "missing_information": ["La préférence demandée"] if abstained else [],
        },
    }
    trusted_answer = validate_native_answer(answer, capsule)
    world_match = re.match(r"^synthetic-\d{6}-([a-f0-9]{12})--", request_id)
    if world_match is None:  # pragma: no cover - generated IDs are internal
        raise AssertionError("identifiant de monde synthétique invalide")
    world_tag = world_match.group(1)
    return {
        "schema_version": EXAMPLE_SCHEMA,
        "example_id": request_id,
        "task": task,
        "memory_capsule": capsule,
        "messages": [
            *strict_chat_messages(capsule),
            {"role": "assistant", "content": _canonical(trusted_answer)},
        ],
        "target": trusted_answer,
        "provenance": {
            "generator_sha256": generator_sha256,
            "synthetic_world": world_tag,
            "evidence_ids": [item["evidence_id"] for item in evidence],
            "template_version": SYNTHETIC_TEMPLATE_VERSION,
        },
    }


def build_synthetic_memory_curriculum(
    *,
    seed: int = 20_260_721,
    count: int = 2_250,
) -> dict[str, Any]:
    """Generate an exact number of corpus-independent, deterministic examples."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not -(2**63) <= seed < 2**63:
        raise ValueError("seed doit être un entier signé sur 64 bits")
    if isinstance(count, bool) or not isinstance(count, int) or not 9 <= count <= 100_000:
        raise ValueError("count doit être compris entre 9 et 100000")
    generator_projection = {
        "schema_version": SYNTHETIC_GENERATOR_SCHEMA,
        "template_version": SYNTHETIC_TEMPLATE_VERSION,
        "capsule_schema": CAPSULE_SCHEMA_VERSION,
        "answer_schema": ANSWER_SCHEMA_VERSION,
        "seed": seed,
        "count": count,
        "tasks": list(SYNTHETIC_TASK_ORDER),
    }
    generator_sha256 = hashlib.sha256(
        _canonical(generator_projection).encode("utf-8")
    ).hexdigest()
    examples = [
        _align_synthetic_example_with_native_contract(
            _synthetic_example(
                seed,
                index,
                SYNTHETIC_TASK_ORDER[index % len(SYNTHETIC_TASK_ORDER)],
                generator_sha256,
            ),
            generator_sha256,
        )
        for index in range(count)
    ]
    task_counts = {
        task: sum(example["task"] == task for example in examples)
        for task in SYNTHETIC_TASK_ORDER
    }
    return {
        "schema_version": CURRICULUM_SCHEMA,
        "generator_schema_version": SYNTHETIC_GENERATOR_SCHEMA,
        "generator_sha256": generator_sha256,
        "seed": seed,
        "synthetic": True,
        "evaluation_eligibility": "independent_of_external_reference_facts",
        "training_contract": {
            "fact_origin": "deterministic_fiction",
            "capsule_schema": CAPSULE_SCHEMA_VERSION,
            "answer_schema": ANSWER_SCHEMA_VERSION,
            "answer_policy": "cite_opaque_evidence_ids_or_abstain",
            "calculation_policy": "structured_expression_and_reported_result",
            "external_corpus_used": False,
        },
        "task_counts": task_counts,
        "example_count": len(examples),
        "examples": examples,
    }


def audit_curriculum_isolation(
    curriculum: Mapping[str, Any],
    forbidden_corpus_path: str | Path,
) -> dict[str, Any]:
    """Prove that generated rows contain no identity/provenance from a corpus."""

    if curriculum.get("synthetic") is not True:
        raise MemoryNativeCurriculumError("l'audit d'isolation exige un curriculum synthétique")
    examples = curriculum.get("examples")
    if not isinstance(examples, list):
        raise MemoryNativeCurriculumError("curriculum.examples doit être une liste")
    forbidden = load_facts_only_corpus(forbidden_corpus_path)
    tokens: set[tuple[str, str]] = set()
    for entity in forbidden["entities"]:
        tokens.add(("entity_id", str(entity["id"])))
        tokens.add(("entity_name", str(entity["name"])))
    for source in forbidden["sources"]:
        for field in ("id", "title", "url", "publisher"):
            tokens.add((f"source_{field}", str(source[field])))
    for claim in forbidden["claims"]:
        tokens.add(("claim_id", str(claim["id"])))
        if claim.get("statement"):
            tokens.add(("claim_statement", str(claim["statement"])))

    encoded = _canonical(examples).casefold()
    collisions = [
        {"kind": kind, "value": value}
        for kind, value in sorted(tokens)
        if len(value.strip()) >= 4 and value.casefold() in encoded
    ]
    return {
        "schema_version": "memory-native-isolation-audit-v1",
        "passed": not collisions,
        "collision_count": len(collisions),
        "checked_token_count": len(tokens),
        "forbidden_facts_sha256": forbidden["facts_sha256"],
        "collisions": collisions,
    }


def encode_training_jsonl(examples: Iterable[Mapping[str, Any]]) -> str:
    """Encode one strict, canonical JSON object per line."""

    rows = [_canonical(dict(example)) for example in examples]
    return "" if not rows else "\n".join(rows) + "\n"


def write_training_jsonl(path: str | Path, examples: Iterable[Mapping[str, Any]]) -> Path:
    """Write a deterministic UTF-8 JSONL artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encode_training_jsonl(examples), encoding="utf-8", newline="\n")
    return output_path


__all__ = [
    "ABSTENTION_MARKER",
    "CAPSULE_SCHEMA",
    "CURRICULUM_SCHEMA",
    "EXAMPLE_SCHEMA",
    "MemoryNativeCurriculumError",
    "SYNTHETIC_GENERATOR_SCHEMA",
    "SYNTHETIC_TASK_ORDER",
    "TASK_ORDER",
    "audit_curriculum_isolation",
    "build_memory_native_curriculum",
    "build_synthetic_memory_curriculum",
    "encode_training_jsonl",
    "load_facts_only_corpus",
    "write_training_jsonl",
]
