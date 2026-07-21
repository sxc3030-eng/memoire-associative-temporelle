"""Construit hors ligne des capsules de rappel depuis un corpus scientifique validé."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from .memory import MemoryEngine
from .memory_hub import MemoryHub, SpacePolicy


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_MAX_DATASET_BYTES = 5_000_000
_EVALUATION_KEYS = frozenset(
    {
        "evaluation_questions",
        "expected_answer_fragments",
        "forbidden_answer_fragments",
        "supporting_claim_ids",
        "answer_status",
        "evaluation_kind",
    }
)


class ScienceCurriculumError(ValueError):
    """Le corpus scientifique ne respecte pas le contrat attendu."""


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScienceCurriculumError(f"clé JSON dupliquée: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ScienceCurriculumError(f"constante JSON interdite: {value}")


def _text(value: Any, field: str, maximum: int = 4_000) -> str:
    if not isinstance(value, str):
        raise ScienceCurriculumError(f"{field} doit être une chaîne")
    clean = value.strip()
    if not clean or len(clean) > maximum:
        raise ScienceCurriculumError(f"{field} est vide ou trop long")
    return clean


def _identifier(value: Any, field: str) -> str:
    clean = _text(value, field, 128)
    if not _ID_RE.fullmatch(clean):
        raise ScienceCurriculumError(f"{field} n'est pas un identifiant stable")
    return clean


def _slug_identifier(value: Any, field: str) -> str:
    return _identifier(_text(value, field, 128).replace("_", "-"), field)


def _string_list(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty) or len(value) > 100:
        raise ScienceCurriculumError(f"{field} doit être une liste bornée")
    result = [_text(item, f"{field}[]", 2_000) for item in value]
    if len(set(result)) != len(result):
        raise ScienceCurriculumError(f"{field} contient un doublon")
    return result


def _objects(value: Any, field: str, maximum: int) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise ScienceCurriculumError(f"{field} doit être une liste non vide et bornée")
    if not all(isinstance(item, Mapping) for item in value):
        raise ScienceCurriculumError(f"{field} doit contenir uniquement des objets")
    return list(value)


def _unique_ids(rows: Sequence[Mapping[str, Any]], field: str) -> set[str]:
    identifiers: set[str] = set()
    for index, row in enumerate(rows):
        identifier = _identifier(row.get("id"), f"{field}[{index}].id")
        if identifier in identifiers:
            raise ScienceCurriculumError(f"identifiant dupliqué: {identifier}")
        identifiers.add(identifier)
    return identifiers


def load_science_dataset(path: str | Path) -> dict[str, Any]:
    """Charge et valide le corpus sans effectuer d'écriture mémoire."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise ScienceCurriculumError(f"corpus introuvable: {dataset_path}")
    if dataset_path.stat().st_size > _MAX_DATASET_BYTES:
        raise ScienceCurriculumError("le corpus dépasse 5 000 000 octets")
    try:
        document = json.loads(
            dataset_path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceCurriculumError("le corpus n'est pas un JSON UTF-8 valide") from error
    if not isinstance(document, Mapping):
        raise ScienceCurriculumError("la racine du corpus doit être un objet")
    required = {"schema_version", "sources", "entities", "claims", "evaluation_questions"}
    if set(document) != required:
        raise ScienceCurriculumError("les champs racine du corpus sont incomplets ou inconnus")
    schema_version = _text(document["schema_version"], "schema_version", 100)
    if schema_version != "science-biographies-v1":
        raise ScienceCurriculumError("schema_version non pris en charge")

    sources = _objects(document["sources"], "sources", 1_000)
    entities = _objects(document["entities"], "entities", 10_000)
    claims = _objects(document["claims"], "claims", 50_000)
    questions = _objects(document["evaluation_questions"], "evaluation_questions", 5_000)
    source_ids = _unique_ids(sources, "sources")
    entity_ids = _unique_ids(entities, "entities")
    claim_ids = _unique_ids(claims, "claims")
    _unique_ids(questions, "evaluation_questions")

    clean_sources: list[dict[str, str]] = []
    for index, source in enumerate(sources):
        if set(source) != {"id", "title", "url", "publisher"}:
            raise ScienceCurriculumError(f"sources[{index}] contient des champs invalides")
        url = _text(source["url"], f"sources[{index}].url", 2_048)
        if not url.startswith(("https://", "http://")):
            raise ScienceCurriculumError(f"sources[{index}].url doit être HTTP(S)")
        clean_sources.append(
            {
                "id": _identifier(source["id"], f"sources[{index}].id"),
                "title": _text(source["title"], f"sources[{index}].title", 500),
                "url": url,
                "publisher": _text(source["publisher"], f"sources[{index}].publisher", 500),
            }
        )

    clean_entities: list[dict[str, str]] = []
    for index, entity in enumerate(entities):
        if set(entity) != {"id", "type", "name", "dossier"}:
            raise ScienceCurriculumError(f"entities[{index}] contient des champs invalides")
        clean_entities.append(
            {
                "id": _identifier(entity["id"], f"entities[{index}].id"),
                "type": _slug_identifier(entity["type"], f"entities[{index}].type"),
                "name": _text(entity["name"], f"entities[{index}].name", 500),
                "dossier": _identifier(entity["dossier"], f"entities[{index}].dossier"),
            }
        )

    clean_claims: list[dict[str, Any]] = []
    allowed_claim = {"id", "subject", "predicate", "object", "date", "source_ids", "status", "statement_fr"}
    for index, claim in enumerate(claims):
        if not set(claim) <= allowed_claim or not (allowed_claim - {"statement_fr"}) <= set(claim):
            raise ScienceCurriculumError(f"claims[{index}] contient des champs invalides")
        subject = _identifier(claim["subject"], f"claims[{index}].subject")
        if subject not in entity_ids:
            raise ScienceCurriculumError(f"claims[{index}] référence un sujet inconnu")
        referenced_sources = _string_list(claim["source_ids"], f"claims[{index}].source_ids")
        if not set(referenced_sources) <= source_ids:
            raise ScienceCurriculumError(f"claims[{index}] référence une source inconnue")
        date = claim["date"]
        if not isinstance(date, Mapping) or set(date) != {"value", "precision"}:
            raise ScienceCurriculumError(f"claims[{index}].date est invalide")
        clean = {
            "id": _identifier(claim["id"], f"claims[{index}].id"),
            "subject": subject,
            "predicate": _slug_identifier(claim["predicate"], f"claims[{index}].predicate"),
            "object": _text(claim["object"], f"claims[{index}].object"),
            "date": {
                "value": _text(date["value"], f"claims[{index}].date.value", 100),
                "precision": _slug_identifier(
                    date["precision"], f"claims[{index}].date.precision"
                ),
            },
            "source_ids": referenced_sources,
            "status": _slug_identifier(claim["status"], f"claims[{index}].status"),
        }
        if "statement_fr" in claim:
            clean["statement_fr"] = _text(claim["statement_fr"], f"claims[{index}].statement_fr")
        clean_claims.append(clean)

    clean_questions: list[dict[str, Any]] = []
    question_fields = {
        "id", "question", "expected_answer_fragments", "forbidden_answer_fragments",
        "answer_status", "evaluation_kind", "supporting_claim_ids",
    }
    for index, question in enumerate(questions):
        if set(question) != question_fields:
            raise ScienceCurriculumError(f"evaluation_questions[{index}] contient des champs invalides")
        supporting = _string_list(
            question["supporting_claim_ids"],
            f"evaluation_questions[{index}].supporting_claim_ids",
            allow_empty=True,
        )
        if not set(supporting) <= claim_ids:
            raise ScienceCurriculumError(
                f"evaluation_questions[{index}] référence une affirmation inconnue"
            )
        answer_status = _slug_identifier(
            question["answer_status"],
            f"evaluation_questions[{index}].answer_status",
        )
        expected = _string_list(
            question["expected_answer_fragments"],
            f"evaluation_questions[{index}].expected_answer_fragments",
            allow_empty=True,
        )
        if answer_status not in {"answerable", "unanswerable"}:
            raise ScienceCurriculumError(
                f"evaluation_questions[{index}].answer_status est inconnu"
            )
        if answer_status == "answerable" and not expected:
            raise ScienceCurriculumError(
                f"evaluation_questions[{index}] exige un fragment attendu"
            )
        if answer_status == "unanswerable" and expected:
            raise ScienceCurriculumError(
                f"evaluation_questions[{index}] non répondable ne doit pas avoir de réponse attendue"
            )
        clean_questions.append(
            {
                "id": _identifier(question["id"], f"evaluation_questions[{index}].id"),
                "question": _text(question["question"], f"evaluation_questions[{index}].question", 10_000),
                "expected_answer_fragments": expected,
                "forbidden_answer_fragments": _string_list(
                    question["forbidden_answer_fragments"],
                    f"evaluation_questions[{index}].forbidden_answer_fragments",
                    allow_empty=True,
                ),
                "answer_status": answer_status,
                "evaluation_kind": _slug_identifier(
                    question["evaluation_kind"],
                    f"evaluation_questions[{index}].evaluation_kind",
                ),
                "supporting_claim_ids": supporting,
            }
        )

    return {
        "schema_version": schema_version,
        "sources": clean_sources,
        "entities": clean_entities,
        "claims": clean_claims,
        "evaluation_questions": clean_questions,
    }


def _claim_text(claim: Mapping[str, Any], entities: Mapping[str, Mapping[str, str]]) -> str:
    if claim.get("statement_fr"):
        return str(claim["statement_fr"])
    subject = entities[str(claim["subject"])]["name"]
    object_id = str(claim["object"])
    object_label = entities[object_id]["name"] if object_id in entities else object_id
    predicate = str(claim["predicate"]).replace("-", " ")
    date = claim["date"]
    return (
        f"{subject} — {predicate} — {object_label}. "
        f"Date : {date['value']} ({date['precision']}). "
        f"Statut documentaire : {claim['status']}."
    )


def _reference_observations(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the only payloads allowed to cross into reference storage.

    This projection deliberately has no access to evaluation questions after
    its inputs are selected.  It therefore cannot accidentally persist a
    question, an expected answer, or the benchmark's supporting-claim key.
    """

    sources = {source["id"]: source for source in dataset["sources"]}
    entities = {entity["id"]: entity for entity in dataset["entities"]}
    claims_by_dossier: dict[str, list[Mapping[str, Any]]] = {}
    for claim in dataset["claims"]:
        entity = entities[claim["subject"]]
        claims_by_dossier.setdefault(entity["dossier"], []).append(claim)

    observations: list[dict[str, Any]] = []
    for dossier, dossier_claims in sorted(claims_by_dossier.items()):
        source_ids = sorted(
            {
                source_id
                for claim in dossier_claims
                for source_id in claim["source_ids"]
            }
        )
        claim_ids = [claim["id"] for claim in dossier_claims]
        text = "\n".join(
            f"[{claim['id']}] {_claim_text(claim, entities)}"
            for claim in dossier_claims
        )
        observations.append(
            {
                "text": text,
                "episode_id": f"science:{dossier}",
                "context": {
                    "origin": "science_curriculum",
                    "dossier": dossier,
                    "claim_ids": claim_ids,
                    "sources": [sources[source_id] for source_id in source_ids],
                    "claim_provenance": [
                        {
                            "claim_id": claim["id"],
                            "subject_id": claim["subject"],
                            "predicate": claim["predicate"],
                            "date": claim["date"],
                            "claim_status": claim["status"],
                            "source_ids": claim["source_ids"],
                            "statement": _claim_text(claim, entities),
                        }
                        for claim in dossier_claims
                    ],
                },
                "source": {
                    "type": "observed",
                    "origin": "science_curriculum",
                    "dossier": dossier,
                    "claim_ids": claim_ids,
                    "source_ids": source_ids,
                },
                "idempotency_key": (
                    f"science-reference:{dataset['schema_version']}:{dossier}"
                ),
            }
        )
    return observations


def import_science_reference(
    dataset_path: str | Path,
    reference_engine: MemoryEngine,
) -> dict[str, Any]:
    """Persist validated scientific claims in an already separate engine.

    ``reference_engine`` is intentionally explicit: callers must create a
    dedicated SQLite-backed engine rather than accidentally selecting the
    agent's writable personal memory. Re-importing the same corpus is
    idempotent. A changed dossier using the same version is rejected by the
    engine's idempotency contract instead of silently mixing revisions.
    """

    if not isinstance(reference_engine, MemoryEngine):
        raise TypeError("reference_engine doit etre un MemoryEngine dedie")
    if reference_engine.db_path == ":memory:":
        raise ScienceCurriculumError(
            "la memoire de reference persistante exige un fichier SQLite"
        )
    dataset = load_science_dataset(dataset_path)
    observations = _reference_observations(dataset)
    created = 0
    duplicates = 0
    for observation in observations:
        result = reference_engine.observe(**observation)
        created += int(bool(result.get("created")))
        duplicates += int(bool(result.get("duplicate")))

    # The fingerprint is computed from reference material only. Changing a
    # benchmark question or answer cannot alter the stored-memory identity.
    reference_projection = {
        key: dataset[key] for key in ("schema_version", "sources", "entities", "claims")
    }
    fingerprint = hashlib.sha256(_encode(reference_projection).encode("utf-8")).hexdigest()
    return {
        "schema_version": "science-reference-import-v1",
        "dataset_schema_version": dataset["schema_version"],
        "reference_sha256": fingerprint,
        "claims_imported": len(dataset["claims"]),
        "dossiers": len(observations),
        "created_dossiers": created,
        "duplicate_dossiers": duplicates,
        "evaluation_items_excluded": len(dataset["evaluation_questions"]),
        "storage_policy": "reference",
        "persistent": True,
    }


def _stable_item(item: Mapping[str, Any]) -> dict[str, Any]:
    context = item.get("context", {})
    context = context if isinstance(context, Mapping) else {}
    claims = context.get("claim_provenance", [])
    facts = []
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, Mapping):
                continue
            facts.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "statement": claim.get("statement"),
                    "date": claim.get("date"),
                    "status": claim.get("claim_status"),
                    "source_ids": claim.get("source_ids", []),
                }
            )
    facts.sort(key=lambda claim: str(claim.get("claim_id", "")))
    sources = context.get("sources", [])
    stable_sources = sorted(
        (dict(source) for source in sources if isinstance(source, Mapping)),
        key=lambda source: str(source.get("id", "")),
    )
    explanation = dict(item.get("explanation", {}))
    components = dict(explanation.get("components", {}))
    components.pop("recency", None)
    explanation["components"] = components
    return {
        "episode_id": item.get("episode_id"),
        "dossier": str(item.get("episode_id", "")).removeprefix("science:"),
        "facts": facts,
        "sources": stable_sources,
        "retrieval": {
            # Counts preserve retrieval diagnostics without reproducing any
            # token from a held-out evaluation question in the capsule.
            "matched_concept_count": len(item.get("matched_concepts", [])),
            "query_concept_count": len(item.get("query_concepts", [])),
            "explanation": explanation,
        },
        "origin": {
            "space": item.get("space"),
            "space_policy": item.get("space_policy"),
        },
    }


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _bounded_capsule(raw: Mapping[str, Any], budget: int) -> dict[str, Any]:
    stable_items = [_stable_item(item) for item in raw.get("items", [])]
    stable_items.sort(
        key=lambda item: (
            -item["retrieval"]["matched_concept_count"],
            -len(item["facts"]),
            item["episode_id"] or "",
        )
    )

    def envelope(items: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            "schema_version": "science-reference-capsule-v1",
            "query_sha256": raw["query_sha256"],
            "items": items,
            "retrieval": {
                "spaces_consulted": raw["retrieval"]["spaces_consulted"],
                "candidates": raw["retrieval"]["candidates"],
                "returned": len(items),
            },
            "budget": {
                "character_limit": budget,
                "characters_used": 0,
                "truncated": len(items) < len(stable_items),
                "measurement": "compact_json_characters",
            },
        }
        for _ in range(4):
            size = len(_encode(result))
            if result["budget"]["characters_used"] == size:
                break
            result["budget"]["characters_used"] = size
        return result

    kept: list[dict[str, Any]] = []
    for item in stable_items:
        candidate = envelope([*kept, item])
        if len(_encode(candidate)) <= budget:
            kept.append(item)
    result = envelope(kept)
    if len(_encode(result)) > budget:
        raise ScienceCurriculumError("budget trop petit pour la capsule neutre")
    return result


def _claim_ids_in_capsule(capsule: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    for item in capsule.get("items", []):
        for fact in item.get("facts", []):
            if isinstance(fact, Mapping) and isinstance(fact.get("claim_id"), str):
                found.add(fact["claim_id"])
    return found


def build_science_capsules(
    dataset_path: str | Path,
    *,
    character_budget: int = 12_000,
    top_k: int = 5,
) -> dict[str, Any]:
    """Importe seulement les affirmations, rappelle les questions et mesure la couverture."""

    if isinstance(character_budget, bool) or not isinstance(character_budget, int):
        raise TypeError("character_budget doit être un entier")
    if not 1_024 <= character_budget <= 200_000:
        raise ValueError("character_budget doit être compris entre 1024 et 200000")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k doit être compris entre 1 et 20")
    dataset = load_science_dataset(dataset_path)
    capsule_rows: dict[str, dict[str, Any]] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="science-curriculum-") as directory:
        engine = MemoryEngine(Path(directory) / "reference.sqlite3")
        try:
            for observation in _reference_observations(dataset):
                engine.observe(**observation)
            hub = MemoryHub(
                {"science-reference": engine},
                {"science-reference": SpacePolicy.reference()},
            )
            for question in dataset["evaluation_questions"]:
                # Seuls l'identifiant et le texte de question traversent la frontière de rappel.
                raw = hub.recall_capsule(
                    "science-curriculum-reader",
                    question["question"],
                    space_names=["science-reference"],
                    top_k=top_k,
                    character_budget=1_000_000,
                )
                capsule = _bounded_capsule(raw, character_budget)
                capsule_rows[question["id"]] = capsule

                # La clé de correction est consultée seulement après construction de la capsule.
                supporting = set(question["supporting_claim_ids"])
                recalled = _claim_ids_in_capsule(capsule)
                present = sorted(supporting & recalled)
                missing = sorted(supporting - recalled)
                diagnostic_rows.append(
                    {
                        "question_id": question["id"],
                        "supporting_claim_ids": sorted(supporting),
                        "recalled_supporting_claim_ids": present,
                        "missing_supporting_claim_ids": missing,
                        "coverage": round(
                            len(present) / len(supporting) if supporting else 1.0,
                            6,
                        ),
                    }
                )
        finally:
            engine.close()

    total_expected = sum(len(row["supporting_claim_ids"]) for row in diagnostic_rows)
    total_recalled = sum(len(row["recalled_supporting_claim_ids"]) for row in diagnostic_rows)
    output = {
        "schema_version": "science-curriculum-output-v1",
        "capsules": capsule_rows,
        "diagnostics": {
            "dataset_schema_version": dataset["schema_version"],
            "sources_validated": len(dataset["sources"]),
            "entities_validated": len(dataset["entities"]),
            "claims_imported": len(dataset["claims"]),
            "questions_recalled": len(dataset["evaluation_questions"]),
            "reference_space_policy": "reference",
            "temporary_reference_removed": True,
            "supporting_claim_coverage": round(
                total_recalled / total_expected if total_expected else 1.0, 6
            ),
            "questions": diagnostic_rows,
        },
    }
    capsule_json = _encode(output["capsules"])
    for forbidden_key in _EVALUATION_KEYS:
        if f'"{forbidden_key}"' in capsule_json:
            raise AssertionError(f"fuite d'un champ d'évaluation: {forbidden_key}")
    return output


__all__ = [
    "ScienceCurriculumError",
    "build_science_capsules",
    "import_science_reference",
    "load_science_dataset",
]
