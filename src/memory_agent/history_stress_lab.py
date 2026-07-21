"""Laboratoire historique deterministe, local et sans contamination.

Le laboratoire genere un corpus fictif, demande a l'oracle historique public de
normaliser les faits et de produire les calculs derivables, puis injecte le tout
dans deux bases SQLite temporaires.  Les faits source sont ``observed`` et les
calculs de l'oracle sont ``inferred`` : une derivation ne devient donc jamais
une preuve factuelle autonome.

Le moteur de memoire v0.4 ordonne encore les evenements selon leur ingestion.
Le temps historique reste present dans le texte, le contexte et l'oracle, mais
il ne remplace pas silencieusement cet ordre interne.  Le rapport expose cette
limite au lieu de presenter le test comme une requete bitemporelle native.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import tempfile
import time
from typing import Any, Mapping, Sequence

from .history_oracle import (
    build_ground_truth,
    calculation_catalog,
    derive_calculable_data,
    normalize_historical_event,
)
from .pipeline import MemoryPipeline


_SCHEMA_VERSION = "history-stress-v1"
_MIN_EVENTS = 2
_MAX_EVENTS = 10_000
_MAX_EPISODE_SIZE = 64
_MAX_QUERIES = 5_000
_MAX_RATE = 0.75
_MAX_DERIVATIONS_PER_EVENT = 16
_MAX_DERIVATION_TEXT = 8_000
_EARLIEST_YEAR = -2400
_LATEST_YEAR = 2026


@dataclass(frozen=True, slots=True)
class HistoryStressConfig:
    """Configuration bornee du corpus et de son evaluation locale."""

    event_count: int = 100
    seed: int = 20_260_721
    episode_size: int = 4
    duplicate_rate: float = 0.12
    contradiction_rate: float = 0.08
    out_of_order_rate: float = 0.35
    max_queries: int = 200

    def __post_init__(self) -> None:
        _bounded_integer(
            self.event_count,
            name="event_count",
            minimum=_MIN_EVENTS,
            maximum=_MAX_EVENTS,
        )
        _bounded_integer(
            self.seed,
            name="seed",
            minimum=0,
            maximum=2**63 - 1,
        )
        _bounded_integer(
            self.episode_size,
            name="episode_size",
            minimum=1,
            maximum=_MAX_EPISODE_SIZE,
        )
        _bounded_integer(
            self.max_queries,
            name="max_queries",
            minimum=1,
            maximum=_MAX_QUERIES,
        )
        _bounded_rate(self.duplicate_rate, name="duplicate_rate")
        _bounded_rate(self.contradiction_rate, name="contradiction_rate")
        _bounded_rate(self.out_of_order_rate, name="out_of_order_rate")


def _bounded_integer(
    value: Any, *, name: str, minimum: int, maximum: int
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} doit etre un entier")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} doit etre compris entre {minimum} et {maximum}")
    return value


def _bounded_rate(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} doit etre un nombre")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= _MAX_RATE:
        raise ValueError(f"{name} doit etre fini et compris entre 0 et {_MAX_RATE}")
    return numeric


def history_stress_catalog() -> dict[str, Any]:
    """Decrit le laboratoire, ses quotas et les limites de la mesure."""

    oracle_calculations = calculation_catalog()
    if not isinstance(oracle_calculations, Mapping):
        raise TypeError("calculation_catalog doit retourner un objet")
    calculations = oracle_calculations.get("calculations", [])
    if not isinstance(calculations, Sequence) or isinstance(
        calculations, (str, bytes)
    ):
        raise TypeError("calculation_catalog.calculations doit etre une liste")
    return {
        "schema_version": _SCHEMA_VERSION,
        "description": (
            "Corpus historique fictif et reproductible, evalue dans deux bases "
            "SQLite temporaires independantes de la memoire de l'agent."
        ),
        "public_functions": [
            "history_stress_catalog",
            "generate_history_scenario",
            "run_history_stress",
        ],
        "config_bounds": {
            "event_count": {"minimum": _MIN_EVENTS, "maximum": _MAX_EVENTS},
            "seed": {"minimum": 0, "maximum": 2**63 - 1},
            "episode_size": {"minimum": 1, "maximum": _MAX_EPISODE_SIZE},
            "rates": {"minimum": 0.0, "maximum": _MAX_RATE},
            "max_queries": {"minimum": 1, "maximum": _MAX_QUERIES},
            "derivations_per_event": _MAX_DERIVATIONS_PER_EVENT,
        },
        "coverage": [
            "annees civiles de l'Antiquite a 2026, sans annee zero",
            "entites homonymes separees par contexte",
            "mesures, unites, coordonnees et changements temporels",
            "doublons idempotents, contradictions et ingestion hors ordre",
            "rappels par marqueur, calcul derive, contexte et contradiction",
        ],
        "provenance_policy": {
            "source_events": "observed",
            "oracle_derivations": "inferred",
            "free_form_formula_execution": False,
            "automatic_promotion_of_derivations": False,
        },
        "calculations": list(calculations),
        "calculation_count": len(calculations),
        "calculation_catalog": dict(oracle_calculations),
        "limitations": [
            "Le moteur actuel ordonne les episodes par ingestion, pas par valid_from.",
            "L'oracle fournit la verite historique attendue; la memoire teste le rappel lexical et contextuel.",
            "Les resultats sont locaux a la machine et ne s'extrapolent pas directement a un milliard d'evenements.",
        ],
    }


_ENTITY_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "subject": "Alexandrie",
        "entity_key": "alexandrie-port-a",
        "domain": "geographie",
        "region": "mediterranee",
    },
    {
        "subject": "Alexandrie",
        "entity_key": "alexandrie-observatoire-b",
        "domain": "institution",
        "region": "amerique-nord",
    },
    {
        "subject": "Mercure",
        "entity_key": "mercure-monde-c",
        "domain": "astronomie",
        "region": "systeme-fictif",
    },
    {
        "subject": "Mercure",
        "entity_key": "mercure-materiau-d",
        "domain": "materiau",
        "region": "laboratoire-fictif",
    },
    {
        "subject": "Victoria",
        "entity_key": "victoria-cite-e",
        "domain": "geographie",
        "region": "archipel-fictif",
    },
    {
        "subject": "Victoria",
        "entity_key": "victoria-personne-f",
        "domain": "biographie-fictive",
        "region": "europe",
    },
    {
        "subject": "Jordan",
        "entity_key": "jordan-fleuve-g",
        "domain": "hydrologie",
        "region": "orient-fictif",
    },
    {
        "subject": "Jordan",
        "entity_key": "jordan-chercheur-h",
        "domain": "biographie-fictive",
        "region": "amerique-nord",
    },
)

_PREDICATES = (
    "population_estimee",
    "masse_reference",
    "statut_historique",
    "longueur_reference",
)


def _civil_year_at(index: int, total: int) -> int:
    """Interpole dans les annees civiles en sautant explicitement l'annee 0."""

    if total <= 1:
        return _LATEST_YEAR
    negative_count = abs(_EARLIEST_YEAR)
    available_offsets = negative_count + _LATEST_YEAR - 1
    offset = round(index * available_offsets / (total - 1))
    if offset < negative_count:
        return _EARLIEST_YEAR + offset
    return 1 + (offset - negative_count)


def _shift_civil_year(year: int, delta: int) -> int:
    if year == 0:
        raise ValueError("L'annee civile zero est interdite")
    shifted = year + delta
    if year < 0 <= shifted:
        shifted += 1
    elif year > 0 >= shifted:
        shifted -= 1
    return shifted


def _civil_year_label(year: int) -> str:
    if year == 0:
        raise ValueError("L'annee civile zero est interdite")
    return f"{abs(year)} AEC" if year < 0 else f"{year} EC"


def _rate_count(total: int, rate: float) -> int:
    if rate <= 0:
        return 0
    return min(total, max(1, round(total * rate)))


def _base_object(index: int, predicate: str) -> str | int | float:
    if predicate == "population_estimee":
        return 2_000 + (index * 7_919) % 900_000
    if predicate == "masse_reference":
        return round(12.5 + (index * 13.75) % 750, 3)
    if predicate == "longueur_reference":
        return round(18 + (index * 29.5) % 3_000, 3)
    return ("active", "archivee", "reconstruite", "contestee")[index % 4]


def _measurements(index: int) -> list[dict[str, int | float | str]]:
    family = index % 4
    if family == 0:
        return [
            {"name": "distance", "value": 12 + index % 400, "unit": "km"},
            {"name": "duration", "value": 1 + index % 12, "unit": "h"},
        ]
    if family == 1:
        return [{"name": "mass", "value": round(5.25 + index * 0.5, 3), "unit": "kg"}]
    if family == 2:
        return [{"name": "length", "value": 100 + index * 3, "unit": "m"}]
    return [{"name": "temperature", "value": -20 + index % 70, "unit": "C"}]


def _make_base_events(config: HistoryStressConfig) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    latest_for_key: dict[tuple[str, str], str] = {}
    for index in range(config.event_count):
        variant = _ENTITY_VARIANTS[index % len(_ENTITY_VARIANTS)]
        predicate = _PREDICATES[(index // 2) % len(_PREDICATES)]
        year = _civil_year_at(index, config.event_count)
        event_id = f"hist-{config.seed:016x}-{index:05d}"
        key = (variant["entity_key"], predicate)
        event: dict[str, Any] = {
            "id": event_id,
            "subject": variant["subject"],
            "predicate": predicate,
            "object": _base_object(index, predicate),
            "valid_from": {"year": year},
            "recorded_order": index,
            "source": {
                "id": f"archive-fictive-{index % 7}",
                "label": f"Archive fictive {index % 7}",
                "confidence": round(0.55 + (index % 5) * 0.09, 2),
            },
            "context": {
                "dataset": "history-stress-fictional",
                "entity_key": variant["entity_key"],
                "domain": variant["domain"],
                "region": variant["region"],
            },
            "measurements": _measurements(index),
        }
        if index % 3 == 0 and year < _LATEST_YEAR:
            event["valid_to"] = {
                "year": min(_LATEST_YEAR, _shift_civil_year(year, 1 + index % 17))
            }
        if index % 4 == 0:
            event["birth_year"] = _shift_civil_year(year, -(18 + index % 63))
        if index % 5 == 0:
            event["coordinates"] = {
                "latitude": round(-68.0 + (index * 17 % 136) + 0.125, 6),
                "longitude": round(-175.0 + (index * 31 % 350) + 0.25, 6),
            }
        previous_id = latest_for_key.get(key)
        if previous_id is not None and index % 3 == 1:
            event["supersedes"] = [previous_id]
        latest_for_key[key] = event_id
        events.append(event)
    return events


def _changed_object(value: Any, offset: int) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 101 + offset
    if isinstance(value, float):
        return round(value + 7.25 + offset / 100, 6)
    return f"{value} - version concurrente {offset + 1}"


def _make_contradictions(
    base_events: Sequence[Mapping[str, Any]],
    *,
    count: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if count == 0:
        return [], []
    selected = sorted(rng.sample(range(len(base_events)), count))
    contradictions: list[dict[str, Any]] = []
    pairs: list[dict[str, str]] = []
    for offset, index in enumerate(selected):
        original = json.loads(_canonical_json(base_events[index]))
        conflict_id = f"{original['id']}-conflict-{offset:03d}"
        original["id"] = conflict_id
        original["object"] = _changed_object(original["object"], offset)
        original["recorded_order"] = len(base_events) + offset
        original["source"] = {
            "id": f"archive-concurrente-{offset % 5}",
            "label": f"Archive concurrente fictive {offset % 5}",
            "confidence": round(0.51 + (offset % 4) * 0.08, 2),
        }
        original.pop("supersedes", None)
        original.pop("retracts", None)
        if original.get("measurements"):
            first = dict(original["measurements"][0])
            first["value"] = _changed_object(first["value"], offset)
            original["measurements"] = [first, *original["measurements"][1:]]
        contradictions.append(original)
        pairs.append({"left_id": str(base_events[index]["id"]), "right_id": conflict_id})
    return contradictions, pairs


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _marker(prefix: str, identifier: str) -> str:
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}{digest}z"


def _stable_context(event: Mapping[str, Any]) -> dict[str, Any]:
    context = event.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("L'oracle a retourne un contexte historique invalide")
    return {
        "dataset": str(context.get("dataset", "history-stress-fictional")),
        "entity_key": str(context.get("entity_key", "unknown")),
        "domain": str(context.get("domain", "unknown")),
        "region": str(context.get("region", "unknown")),
    }


def _event_text(event: Mapping[str, Any], marker: str) -> str:
    valid_from = event.get("valid_from")
    if not isinstance(valid_from, Mapping):
        raise ValueError("valid_from absent apres normalisation")
    year = int(valid_from["year"])
    components = [
        marker,
        "fait historique fictif",
        f"sujet {event['subject']}",
        f"relation {event['predicate']}",
        f"valeur {event['object']}",
        f"temps evenement {_civil_year_label(year)}",
        f"identifiant {event['id']}",
    ]
    source = event.get("source")
    if isinstance(source, Mapping):
        components.append(f"source historique {source.get('id', 'inconnue')}")
        if source.get("label"):
            components.append(f"archive {source['label']}")
    context = _stable_context(event)
    components.extend(f"contexte {key} {value}" for key, value in sorted(context.items()))
    measurements = event.get("measurements", [])
    if isinstance(measurements, Sequence) and not isinstance(measurements, (str, bytes)):
        for measurement in measurements:
            if isinstance(measurement, Mapping):
                components.append(
                    "mesure "
                    f"{measurement.get('name')} {measurement.get('value')} {measurement.get('unit')}"
                )
    coordinates = event.get("coordinates")
    if isinstance(coordinates, Mapping):
        components.append(
            f"coordonnees {coordinates.get('latitude')} {coordinates.get('longitude')}"
        )
    return ". ".join(str(component) for component in components) + "."


def _episode_assignments(
    events: Sequence[Mapping[str, Any]],
    config: HistoryStressConfig,
    *,
    isolated_event_ids: set[str] | None = None,
) -> dict[str, str]:
    counters: dict[str, int] = {}
    episodes: dict[str, str] = {}
    isolated = isolated_event_ids or set()
    for event in events:
        context = _stable_context(event)
        entity_key = context["entity_key"]
        event_id = str(event["id"])
        if event_id in isolated:
            digest = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:16]
            episodes[event_id] = f"history-{config.seed:x}-independent-{digest}"
            continue
        count = counters.get(entity_key, 0)
        chunk = count // config.episode_size
        episodes[event_id] = f"history-{config.seed:x}-{entity_key}-{chunk:04d}"
        counters[entity_key] = count + 1
    return episodes


def _normalise_calculation_items(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        nested = value.get("calculations")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            value = nested
        else:
            return [dict(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Les calculs de l'oracle doivent former une sequence")
    calculations: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("Un calcul de l'oracle n'est pas un objet")
        calculations.append(dict(item))
    return calculations


def _derive_all(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Appelle l'oracle sur chaque fait et son precedent de meme sujet/relation."""

    previous_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    calculations: list[dict[str, Any]] = []
    for event in events:
        key = (
            str(_stable_context(event)["entity_key"]),
            str(event["predicate"]),
        )
        previous = previous_by_key.get(key)
        produced = derive_calculable_data(
            event,
            previous_event=previous,
            birth_year=event.get("birth_year"),
        )
        calculations.extend(_normalise_calculation_items(produced))
        previous_by_key[key] = event
    return calculations


def _merge_calculations(
    direct: Sequence[Mapping[str, Any]], ground_truth: Mapping[str, Any]
) -> list[dict[str, Any]]:
    candidates = [dict(item) for item in direct]
    candidates.extend(
        _normalise_calculation_items(ground_truth.get("calculations", []))
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        encoded = _canonical_json(item)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(item)
    return unique


def _source_ids_for_calculation(calculation: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "event_id",
        "source_event_id",
        "left_event_id",
        "right_event_id",
        "from_event_id",
        "to_event_id",
    ):
        value = calculation.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    for key in (
        "event_ids",
        "source_event_ids",
        "dependency_event_ids",
        "depends_on",
    ):
        raw = calculation.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values.extend(str(item) for item in raw if str(item))
    return list(dict.fromkeys(values))


def _derived_records(
    calculations: Sequence[Mapping[str, Any]],
    *,
    config: HistoryStressConfig,
) -> list[dict[str, Any]]:
    maximum = config.event_count * _MAX_DERIVATIONS_PER_EVENT
    if len(calculations) > maximum:
        raise ValueError(
            f"L'oracle depasse le quota de {maximum} derivations pour ce scenario"
        )
    records: list[dict[str, Any]] = []
    for index, calculation in enumerate(calculations):
        encoded = _canonical_json(calculation)
        if len(encoded) > _MAX_DERIVATION_TEXT:
            raise ValueError("Une derivation de l'oracle est trop volumineuse")
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        marker = _marker("derivedmarker", digest)
        source_ids = _source_ids_for_calculation(calculation)
        text = (
            f"{marker}. calcul historique derive par oracle borne. "
            f"donnees {_canonical_json(calculation)}."
        )
        records.append(
            {
                "kind": "derived",
                "calculation": dict(calculation),
                "marker": marker,
                "text": text,
                "episode_id": f"history-{config.seed:x}-derived-{index:05d}",
                "context": {
                    "dataset": "history-stress-fictional",
                    "domain": "derived_history",
                    "oracle": "history_oracle",
                },
                "source": {
                    "type": "inferred",
                    "origin": "history_oracle",
                    "source_event_ids": source_ids,
                    "external": False,
                },
                "idempotency_key": f"history-stress:{config.seed}:derived:{digest}",
            }
        )
    return records


def _inversion_count(values: Sequence[int]) -> int:
    """Compte les inversions en O(n log n), sans cout quadratique de benchmark."""

    def merge_count(items: list[int]) -> tuple[list[int], int]:
        if len(items) < 2:
            return items, 0
        middle = len(items) // 2
        left, left_count = merge_count(items[:middle])
        right, right_count = merge_count(items[middle:])
        merged: list[int] = []
        inversions = left_count + right_count
        left_index = right_index = 0
        while left_index < len(left) and right_index < len(right):
            if left[left_index] <= right[right_index]:
                merged.append(left[left_index])
                left_index += 1
            else:
                merged.append(right[right_index])
                right_index += 1
                inversions += len(left) - left_index
        merged.extend(left[left_index:])
        merged.extend(right[right_index:])
        return merged, inversions

    return merge_count(list(values))[1]


def _apply_out_of_order(
    records: Sequence[dict[str, Any]], *, rate: float, rng: random.Random
) -> tuple[list[dict[str, Any]], int]:
    ordered = list(records)
    move_count = _rate_count(len(ordered), rate)
    if move_count == 1 and len(ordered) > 1:
        move_count = 2
    if move_count > 1:
        positions = sorted(rng.sample(range(len(ordered)), move_count))
        moved = [ordered[position] for position in positions][::-1]
        for position, record in zip(positions, moved):
            ordered[position] = record
    recorded_orders = [int(record["event"]["recorded_order"]) for record in ordered]
    return ordered, _inversion_count(recorded_orders)


def _sample_evenly(items: Sequence[Any], count: int) -> list[Any]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[-1]]
    indices = {
        round(index * (len(items) - 1) / (count - 1)) for index in range(count)
    }
    return [items[index] for index in sorted(indices)]


def _build_queries(
    source_records: Sequence[Mapping[str, Any]],
    derived_records: Sequence[Mapping[str, Any]],
    contradiction_pairs: Sequence[Mapping[str, str]],
    *,
    ground_truth: Mapping[str, Any],
    max_queries: int,
) -> list[dict[str, Any]]:
    by_kind: dict[str, list[dict[str, Any]]] = {
        "date": [],
        "latest": [],
        "context": [],
        "contradiction": [],
        "marker": [],
        "derived": [],
    }
    source_by_id = {str(record["event"]["id"]): record for record in source_records}

    def evidence_for(record: Mapping[str, Any]) -> list[str]:
        event = record["event"]
        source = event["source"]
        year = int(event["valid_from"]["year"])
        return [
            f"valeur {event['object']}",
            f"temps evenement {_civil_year_label(year)}",
            f"source historique {source['id']}",
        ]

    # Les marqueurs mesurent seulement le cablage index -> episode. Ils ne
    # participent jamais au score semantique publie.
    marker_budget = max(1, max_queries // 6)
    for record in _sample_evenly(source_records, marker_budget):
        by_kind["marker"].append(
            {
                "score_group": "plumbing",
                "kind": "marker",
                "query": record["marker"],
                "context": None,
                "expected_episode_ids": [record["episode_id"]],
                "expected_markers": [record["marker"]],
                "match_mode": "any_episode",
            }
        )

    derived_budget = min(len(derived_records), max(1, max_queries // 6))
    for record in _sample_evenly(derived_records, derived_budget):
        by_kind["derived"].append(
            {
                "score_group": "plumbing",
                "kind": "derived",
                "query": record["marker"],
                "context": None,
                "expected_episode_ids": [record["episode_id"]],
                "expected_markers": [record["marker"]],
                "match_mode": "any_episode",
            }
        )

    # Questions naturelles datees : aucun identifiant ni marqueur synthetique
    # n'est present dans la requete.
    date_budget = max(1, max_queries // 3)
    for record in _sample_evenly(source_records, date_budget):
        event = record["event"]
        year = int(event["valid_from"]["year"])
        by_kind["date"].append(
            {
                "score_group": "semantic",
                "kind": "date",
                "query": (
                    f"{event['subject']} {event['predicate']} en "
                    f"{_civil_year_label(year)}"
                ),
                "context": None,
                "expected_episode_ids": [record["episode_id"]],
                "expected_markers": [],
                "expected_text_fragments": evidence_for(record),
                "match_mode": "any_episode",
            }
        )

    latest = ground_truth.get("latest_by_subject_predicate", {})
    if not isinstance(latest, Mapping):
        raise TypeError("latest_by_subject_predicate doit etre un objet")
    for key, event_id in sorted(latest.items(), key=lambda item: str(item[0])):
        record = source_by_id.get(str(event_id))
        if record is None:
            continue
        event = record["event"]
        by_kind["latest"].append(
            {
                "score_group": "semantic",
                "kind": "latest",
                "query": f"{event['subject']} {event['predicate']} plus recent",
                "context": None,
                "expected_episode_ids": [record["episode_id"]],
                "expected_markers": [],
                "expected_text_fragments": evidence_for(record),
                "match_mode": "any_episode",
                "oracle_key": str(key),
            }
        )

    for pair in contradiction_pairs:
        left = source_by_id.get(pair["left_id"])
        right = source_by_id.get(pair["right_id"])
        if left is None or right is None:
            continue
        event = left["event"]
        year = int(event["valid_from"]["year"])
        expected_episodes = list(
            dict.fromkeys([str(left["episode_id"]), str(right["episode_id"])])
        )
        if len(expected_episodes) != 2:
            raise AssertionError(
                "Une contradiction doit conserver deux episodes independants"
            )
        by_kind["contradiction"].append(
            {
                "score_group": "semantic",
                "kind": "contradiction",
                "query": (
                    f"{event['subject']} {event['predicate']} en "
                    f"{_civil_year_label(year)}"
                ),
                "context": None,
                "expected_episode_ids": expected_episodes,
                "expected_markers": [],
                "expected_evidence_by_episode": {
                    str(left["episode_id"]): evidence_for(left),
                    str(right["episode_id"]): evidence_for(right),
                },
                "match_mode": "all_episodes",
            }
        )

    by_ambiguous_key: dict[tuple[str, str, str], set[str]] = {}
    for record in source_records:
        event = record["event"]
        context = _stable_context(event)
        key = (str(event["subject"]), str(event["predicate"]), context["entity_key"])
        by_ambiguous_key.setdefault(key, set()).add(str(record["episode_id"]))
    for (subject, predicate, entity_key), episode_ids in sorted(by_ambiguous_key.items()):
        same_name_contexts = {
            key[2] for key in by_ambiguous_key if key[0] == subject and key[1] == predicate
        }
        if len(same_name_contexts) < 2:
            continue
        exemplar = next(
            record
            for record in source_records
            if str(record["event"]["subject"]) == subject
            and str(record["event"]["predicate"]) == predicate
            and _stable_context(record["event"])["entity_key"] == entity_key
        )
        context = _stable_context(exemplar["event"])
        by_kind["context"].append(
            {
                "score_group": "semantic",
                "kind": "context",
                "query": f"{subject} {predicate} dans {context['region']}",
                "context": context,
                "expected_episode_ids": sorted(episode_ids),
                "expected_markers": [],
                "expected_text_fragments": [
                    f"sujet {subject}",
                    f"relation {predicate}",
                    f"contexte entity_key {entity_key}",
                ],
                "match_mode": "any_episode",
            }
        )

    # Preserve chaque categorie disponible, puis completer en tourniquet pour
    # qu'un grand ensemble de questions datees n'ecrase pas les autres tests.
    selected: list[dict[str, Any]] = []
    order = ("date", "latest", "context", "contradiction", "marker", "derived")
    cursors = {kind: 0 for kind in order}
    while len(selected) < max_queries:
        progressed = False
        for kind in order:
            cursor = cursors[kind]
            if cursor >= len(by_kind[kind]):
                continue
            selected.append(by_kind[kind][cursor])
            cursors[kind] = cursor + 1
            progressed = True
            if len(selected) >= max_queries:
                break
        if not progressed:
            break
    return selected


def generate_history_scenario(config: HistoryStressConfig) -> dict[str, Any]:
    """Genere un scenario JSON-compatible sans toucher a une base de donnees."""

    if not isinstance(config, HistoryStressConfig):
        raise TypeError("config doit etre une instance de HistoryStressConfig")
    rng = random.Random(config.seed)
    base_events = _make_base_events(config)
    contradiction_count = _rate_count(config.event_count, config.contradiction_rate)
    conflict_events, contradiction_pairs = _make_contradictions(
        base_events,
        count=contradiction_count,
        rng=rng,
    )
    raw_events = [*base_events, *conflict_events]
    normalised_events = [normalize_historical_event(event) for event in raw_events]
    if any(int(event["valid_from"]["year"]) == 0 for event in normalised_events):
        raise AssertionError("L'oracle a laisse passer une annee civile zero")

    ground_truth = build_ground_truth(normalised_events)
    if not isinstance(ground_truth, Mapping):
        raise TypeError("build_ground_truth doit retourner un objet")
    # L'oracle possède seul la sémantique temporelle et les corrections. Le
    # runner ne recalcule pas une seconde chronologie selon l'ordre d'entrée.
    calculations = _normalise_calculation_items(
        ground_truth.get("calculations", [])
    )
    derived = _derived_records(calculations, config=config)

    contradiction_event_ids = {pair["right_id"] for pair in contradiction_pairs}
    # Une source concurrente conserve son episode propre : retrouver une
    # contradiction exige donc deux preuves independantes, pas un seul paquet
    # rendu artificiellement facile par la generation du benchmark.
    episodes = _episode_assignments(
        normalised_events,
        config,
        isolated_event_ids=contradiction_event_ids,
    )
    source_records: list[dict[str, Any]] = []
    for event in normalised_events:
        event_id = str(event["id"])
        marker = _marker("historymarker", event_id)
        context = _stable_context(event)
        source_records.append(
            {
                "kind": "source",
                "event": event,
                "marker": marker,
                "text": _event_text(event, marker),
                "episode_id": episodes[event_id],
                "context": context,
                "source": {
                    "type": "observed",
                    "origin": "history_stress_fixture",
                    "historical_source": event["source"],
                    "event_time": event["valid_from"],
                    "recorded_order": event.get("recorded_order"),
                    "fictional": True,
                },
                "idempotency_key": f"history-stress:{config.seed}:source:{event_id}",
            }
        )

    source_order, inversion_count = _apply_out_of_order(
        source_records,
        rate=config.out_of_order_rate,
        rng=rng,
    )
    submissions = list(source_order)
    duplicate_count = _rate_count(config.event_count, config.duplicate_rate)
    duplicate_targets = (
        rng.sample(list(source_order), duplicate_count) if duplicate_count else []
    )
    for target in duplicate_targets:
        insertion = rng.randrange(0, len(submissions) + 1)
        duplicate = dict(target)
        duplicate["kind"] = "duplicate"
        submissions.insert(insertion, duplicate)
    submissions.extend(derived)

    queries = _build_queries(
        source_records,
        derived,
        contradiction_pairs,
        ground_truth=ground_truth,
        max_queries=config.max_queries,
    )
    years = [int(event["valid_from"]["year"]) for event in normalised_events]
    scenario = {
        "schema_version": _SCHEMA_VERSION,
        "config": asdict(config),
        "events": normalised_events,
        "source_records": source_records,
        "derived_records": derived,
        "submissions": submissions,
        "queries": queries,
        "ground_truth": dict(ground_truth),
        "calculations": calculations,
        "contradiction_pairs": contradiction_pairs,
        "coverage": {
            "earliest_year": min(years),
            "latest_year": max(years),
            "contains_year_zero": 0 in years,
            "base_events": len(base_events),
            "contradictory_events": len(conflict_events),
            "duplicate_submissions": duplicate_count,
            "derived_calculations": len(calculations),
            "source_ingestion_inversions": inversion_count,
            "out_of_order_requested_rate": config.out_of_order_rate,
        },
    }
    # Enforce a strict JSON contract before any pipeline work begins.
    _canonical_json(scenario)
    return scenario


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean": round(statistics.fmean(values), 6) if values else 0.0,
        "p50": round(_percentile(values, 0.50), 6),
        "p95": round(_percentile(values, 0.95), 6),
        "p99": round(_percentile(values, 0.99), 6),
        "max": round(max(values, default=0.0), 6),
    }


def _score_queries(
    pipeline: MemoryPipeline, queries: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[float]]:
    totals: dict[str, dict[str, int]] = {}
    groups: dict[str, dict[str, int]] = {
        "semantic": {
            "queries": 0,
            "top1_queries": 0,
            "top1_hits": 0,
            "top5_hits": 0,
        },
        "plumbing": {
            "queries": 0,
            "top1_queries": 0,
            "top1_hits": 0,
            "top5_hits": 0,
        },
    }
    latencies: list[float] = []
    first_misses: dict[str, list[dict[str, Any]]] = {
        "semantic": [],
        "plumbing": [],
    }
    for query in queries:
        group = str(query.get("score_group", "semantic"))
        if group not in groups:
            raise ValueError(f"Groupe de requete inconnu: {group}")
        requested_context = query.get("context")
        payload: str | dict[str, Any]
        if isinstance(requested_context, Mapping):
            payload = {"query": str(query["query"]), "context": dict(requested_context)}
        else:
            payload = str(query["query"])
        started = time.perf_counter_ns()
        results = pipeline.reader_engine.recall(payload, top_k=5)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        expected_episodes = {str(value) for value in query["expected_episode_ids"]}
        expected_markers = [str(value) for value in query.get("expected_markers", [])]
        expected_fragments = [
            str(value) for value in query.get("expected_text_fragments", [])
        ]
        raw_evidence = query.get("expected_evidence_by_episode", {})
        expected_evidence = {
            str(episode_id): [str(fragment) for fragment in fragments]
            for episode_id, fragments in raw_evidence.items()
        } if isinstance(raw_evidence, Mapping) else {}
        match_mode = str(query.get("match_mode", "any_episode"))

        def matches(result: Mapping[str, Any]) -> bool:
            if str(result.get("episode_id")) not in expected_episodes:
                return False
            text = str(result.get("text", ""))
            episode_fragments = expected_evidence.get(
                str(result.get("episode_id")), expected_fragments
            )
            return (
                all(marker in text for marker in expected_markers)
                and all(fragment in text for fragment in episode_fragments)
            )

        if match_mode == "all_episodes":
            top1_eligible = False
            hit1 = False
            hit5 = all(
                any(
                    str(result.get("episode_id")) == episode_id
                    and matches(result)
                    for result in results[:5]
                )
                for episode_id in expected_episodes
            )
        elif match_mode == "any_episode":
            top1_eligible = True
            hit1 = bool(results and matches(results[0]))
            hit5 = any(matches(result) for result in results[:5])
        else:
            raise ValueError(f"Mode de correspondance inconnu: {match_mode}")

        group_bucket = groups[group]
        group_bucket["queries"] += 1
        group_bucket["top1_queries"] += int(top1_eligible)
        group_bucket["top1_hits"] += int(top1_eligible and hit1)
        group_bucket["top5_hits"] += int(hit5)
        kind = str(query["kind"])
        bucket = totals.setdefault(
            kind,
            {
                "queries": 0,
                "top1_queries": 0,
                "top1_hits": 0,
                "top5_hits": 0,
            },
        )
        bucket["queries"] += 1
        bucket["top1_queries"] += int(top1_eligible)
        bucket["top1_hits"] += int(top1_eligible and hit1)
        bucket["top5_hits"] += int(hit5)
        misses = first_misses[group]
        if not hit5 and len(misses) < 10:
            misses.append(
                {
                    "kind": kind,
                    "query": str(query["query"]),
                    "expected_episode_ids": sorted(expected_episodes),
                    "returned_episode_ids": [
                        str(result.get("episode_id")) for result in results[:5]
                    ],
                }
            )

    def summarize(bucket: Mapping[str, int]) -> dict[str, Any]:
        query_count = int(bucket["queries"])
        top1_count = int(bucket["top1_queries"])
        return {
            **bucket,
            "top1_percent": (
                round(100.0 * int(bucket["top1_hits"]) / top1_count, 6)
                if top1_count
                else None
            ),
            "top5_percent": (
                round(100.0 * int(bucket["top5_hits"]) / query_count, 6)
                if query_count
                else None
            ),
        }

    by_kind: dict[str, Any] = {}
    for kind, bucket in totals.items():
        by_kind[kind] = summarize(bucket)
    semantic = summarize(groups["semantic"])
    plumbing = summarize(groups["plumbing"])
    return (
        {
            "status": "scored",
            # Champs historiques conserves, mais ils designent maintenant le
            # score naturel uniquement. Les marqueurs sont publies separement.
            "queries": semantic["queries"],
            "total_queries_executed": len(queries),
            "top1_queries": semantic["top1_queries"],
            "top1_hits": semantic["top1_hits"],
            "top5_hits": semantic["top5_hits"],
            "top1_percent": semantic["top1_percent"],
            "top5_percent": semantic["top5_percent"],
            "semantic_top1_percent": semantic["top1_percent"],
            "semantic_top5_percent": semantic["top5_percent"],
            "semantic": {
                **semantic,
                "description": "Questions naturelles date, contexte, etat recent et contradictions.",
                "first_misses": first_misses["semantic"],
            },
            "plumbing_diagnostics": {
                **plumbing,
                "description": (
                    "Marqueurs exacts et calculs derives; diagnostic de routage, "
                    "exclu du score semantique."
                ),
                "first_misses": first_misses["plumbing"],
            },
            "by_kind": by_kind,
            "first_misses": first_misses["semantic"],
        },
        latencies,
    )


def _unscored_queries(
    queries: Sequence[Mapping[str, Any]], reasons: Sequence[str]
) -> dict[str, Any]:
    semantic_count = sum(
        1 for query in queries if query.get("score_group", "semantic") == "semantic"
    )
    plumbing_count = len(queries) - semantic_count
    by_kind: dict[str, dict[str, int]] = {}
    for query in queries:
        kind = str(query["kind"])
        by_kind.setdefault(kind, {"queries": 0})["queries"] += 1
    return {
        "status": "not_scored",
        "incomplete_reasons": list(reasons),
        "queries": semantic_count,
        "total_queries_executed": 0,
        "top1_queries": None,
        "top1_hits": None,
        "top5_hits": None,
        "top1_percent": None,
        "top5_percent": None,
        "semantic_top1_percent": None,
        "semantic_top5_percent": None,
        "semantic": {
            "status": "not_scored",
            "queries": semantic_count,
            "top1_percent": None,
            "top5_percent": None,
        },
        "plumbing_diagnostics": {
            "status": "not_run",
            "queries": plumbing_count,
            "top1_percent": None,
            "top5_percent": None,
        },
        "by_kind": by_kind,
        "first_misses": [],
    }


def run_history_stress(config: HistoryStressConfig) -> dict[str, Any]:
    """Execute le scenario dans des bases temporaires et retourne ses mesures."""

    scenario = generate_history_scenario(config)
    report: dict[str, Any]
    temporary_paths_removed = False
    with tempfile.TemporaryDirectory(prefix="history-stress-") as directory:
        root = Path(directory).resolve()
        memory_path = (root / "memory.sqlite3").resolve()
        queue_path = (root / "injection.sqlite3").resolve()
        if root not in memory_path.parents or root not in queue_path.parents:
            raise AssertionError("Les bases du stress test ne sont pas isolees")
        pipeline = MemoryPipeline(
            memory_path,
            queue_path,
            batch_size=16,
            poll_interval=0.005,
            retry_base_delay=0.01,
            retry_max_delay=0.1,
        )
        enqueue_latencies: list[float] = []
        duplicate_responses = 0
        started_all = time.perf_counter()
        try:
            for record in scenario["submissions"]:
                started = time.perf_counter_ns()
                response = pipeline.enqueue(
                    record["text"],
                    episode_id=record["episode_id"],
                    context=record["context"],
                    source=record["source"],
                    idempotency_key=record["idempotency_key"],
                )
                enqueue_latencies.append(
                    (time.perf_counter_ns() - started) / 1_000_000
                )
                duplicate_responses += int(bool(response.get("duplicate")))
            enqueue_finished = time.perf_counter()
            timeout = max(10.0, min(300.0, len(scenario["submissions"]) * 0.15))
            drained = pipeline.wait_until_idle(timeout=timeout)
            consolidation_finished = time.perf_counter()
            stats = pipeline.stats()
            queue_stats = stats["queue"]
            memory_stats = stats["memory"]
            submissions = len(scenario["submissions"])
            unique_expected = len(scenario["source_records"]) + len(
                scenario["derived_records"]
            )
            incomplete_reasons: list[str] = []
            if not drained:
                incomplete_reasons.append("pipeline_not_drained")
            if int(queue_stats["failed"]) > 0:
                incomplete_reasons.append("pipeline_failed_jobs")
            if int(queue_stats["completed"]) != unique_expected:
                incomplete_reasons.append("completed_count_mismatch")
            if incomplete_reasons:
                query_metrics = _unscored_queries(
                    scenario["queries"], incomplete_reasons
                )
                recall_latencies: list[float] = []
            else:
                query_metrics, recall_latencies = _score_queries(
                    pipeline, scenario["queries"]
                )
            finished = time.perf_counter()
            total_elapsed = finished - started_all
            consolidation_elapsed = consolidation_finished - started_all
            oracle_truth = scenario["ground_truth"]
            report = {
                "schema_version": _SCHEMA_VERSION,
                "status": "incomplete" if incomplete_reasons else "complete",
                "incomplete_reasons": incomplete_reasons,
                "config": asdict(config),
                "scenario": {
                    **scenario["coverage"],
                    "normalized_source_events": len(scenario["events"]),
                    "submissions": submissions,
                    "unique_events_expected": unique_expected,
                    "queries_generated": len(scenario["queries"]),
                },
                "oracle": {
                    "schema_version": oracle_truth.get("schema_version"),
                    "events": len(oracle_truth.get("events", [])),
                    "calculations": len(oracle_truth.get("calculations", [])),
                    "contradictions": len(oracle_truth.get("contradictions", [])),
                    "latest_subject_predicate_keys": len(
                        oracle_truth.get("latest_by_subject_predicate", {})
                    ),
                    "all_calculations_injected_as_inferred": len(
                        scenario["derived_records"]
                    )
                    == len(scenario["calculations"]),
                },
                "pipeline": {
                    "drained": drained,
                    "submitted": submissions,
                    "unique_jobs": int(queue_stats["total"]),
                    "completed": int(queue_stats["completed"]),
                    "completed_count_exact": int(queue_stats["completed"])
                    == unique_expected,
                    "failed": int(queue_stats["failed"]),
                    "deduplicated_requests": int(
                        queue_stats["deduplicated_requests"]
                    ),
                    "duplicate_responses": duplicate_responses,
                    "expected_duplicate_submissions": scenario["coverage"][
                        "duplicate_submissions"
                    ],
                    "deduplication_exact": int(
                        queue_stats["deduplicated_requests"]
                    )
                    == scenario["coverage"]["duplicate_submissions"],
                    "reader_writer_separated": bool(
                        stats["reader_writer_separated"]
                    ),
                    "memory_events": int(memory_stats["events"]),
                    "memory_event_count_exact": int(memory_stats["events"])
                    == unique_expected,
                    "scoring_gate_passed": not incomplete_reasons,
                    "source_counts": memory_stats.get("sources", {}),
                    "memory_size_bytes_including_wal_shm": memory_stats.get(
                        "database_size_bytes"
                    ),
                    "queue_size_bytes_including_wal_shm": queue_stats.get(
                        "database_size_bytes"
                    ),
                },
                "retrieval": query_metrics,
                "performance": {
                    "enqueue_seconds": round(enqueue_finished - started_all, 6),
                    "consolidation_seconds": round(consolidation_elapsed, 6),
                    "total_seconds_including_queries": round(total_elapsed, 6),
                    "enqueue_requests_per_second": round(
                        submissions / max(enqueue_finished - started_all, 1e-12), 2
                    ),
                    "completed_per_second": round(
                        int(queue_stats["completed"])
                        / max(consolidation_elapsed, 1e-12),
                        2,
                    ),
                    "enqueue_latency_ms": _latency_summary(enqueue_latencies),
                    "recall_latency_ms": _latency_summary(recall_latencies),
                },
                "provenance": {
                    "source_events_expected_type": "observed",
                    "derived_events_expected_type": "inferred",
                    "derived_results_auto_promoted": 0,
                    "free_form_formulas_executed": 0,
                    "fictional_corpus": True,
                },
                "temporal_semantics": {
                    "event_time_field": "valid_from.year",
                    "ingestion_order_field": "pipeline sequence / MemoryEngine ingest_order",
                    "event_time_equals_ingestion_time": False,
                    "native_bitemporal_ranking": False,
                    "ground_truth_time_owner": "history_oracle",
                    "note": (
                        "Le moteur actuel rappelle des episodes selon les tokens, "
                        "le contexte et l'ordre d'ingestion. valid_from reste une "
                        "preuve textuelle/contextuelle et n'est pas son horloge de tri."
                    ),
                },
                "isolation": {
                    "temporary_directory_used": True,
                    "memory_and_queue_paths_inside_temporary_directory": True,
                    "production_memory_opened": False,
                    "writes_outside_temporary_directory": 0,
                    "temporary_storage_removed_after_run": False,
                },
            }
        finally:
            pipeline.close()
    temporary_paths_removed = not root.exists()
    report["isolation"]["temporary_storage_removed_after_run"] = temporary_paths_removed
    _canonical_json(report)
    return report


__all__ = [
    "HistoryStressConfig",
    "generate_history_scenario",
    "history_stress_catalog",
    "run_history_stress",
]
