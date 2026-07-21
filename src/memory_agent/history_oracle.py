"""Oracle indépendant pour les faits historiques et leurs valeurs calculables.

Ce module ne dépend ni de :mod:`memory_agent.memory`, ni du pipeline, ni du
classement lexical. Il sert de vérité de référence au laboratoire de stress.
Les calculs disponibles sont inscrits en dur; aucune formule importée n'est
jamais exécutée.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, localcontext
import json
import math
import re
from typing import Any, Final


HISTORY_SCHEMA_VERSION: Final = "history-event-v1"
HISTORY_CALCULATION_VERSION: Final = "history-calc-v1"
_MAX_SAFE_INTEGER: Final = 9_007_199_254_740_991
_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_YEAR: Final = 9_999_999
_MAX_MEASUREMENTS: Final = 64
_EARTH_RADIUS_KM: Final = 6_371.0088


class HistoryValidationError(ValueError):
    """Un fait historique n'est pas conforme au schéma borné."""


def _text(value: Any, *, field: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise HistoryValidationError(f"{field} doit etre une chaine")
    clean = " ".join(value.split())
    if not clean:
        raise HistoryValidationError(f"{field} ne peut pas etre vide")
    if len(clean) > maximum:
        raise HistoryValidationError(f"{field} depasse {maximum} caracteres")
    return clean


def _identifier(value: Any, *, field: str) -> str:
    clean = _text(value, field=field, maximum=128)
    if _ID_RE.fullmatch(clean) is None:
        raise HistoryValidationError(f"{field} contient des caracteres interdits")
    return clean


def _decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise HistoryValidationError(f"{field} doit etre un nombre fini")
    encoded = str(value).strip()
    if not encoded or len(encoded) > 256:
        raise HistoryValidationError(f"{field} depasse la taille numerique permise")
    try:
        number = Decimal(encoded)
    except (InvalidOperation, ValueError):
        raise HistoryValidationError(f"{field} doit etre un nombre fini") from None
    if not number.is_finite():
        raise HistoryValidationError(f"{field} doit etre un nombre fini")
    if abs(number.adjusted()) > 1000 or len(number.as_tuple().digits) > 120:
        raise HistoryValidationError(f"{field} depasse la precision ou l'exposant permis")
    return number


def _number(value: Decimal, *, approximate: bool = False) -> int | float | str:
    if approximate:
        result = float(value)
        if not math.isfinite(result):
            raise HistoryValidationError("resultat calcule non fini")
        return result
    integral = value.to_integral_value()
    if value == integral:
        integer = int(integral)
        return integer if abs(integer) <= _MAX_SAFE_INTEGER else str(integer)
    return format(value.normalize(), "f")


def _safe_json(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        raise HistoryValidationError("context depasse la profondeur maximale")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise HistoryValidationError("context contient un entier hors limite JSON sure")
        return value
    if isinstance(value, str):
        if len(value) > 10_000:
            raise HistoryValidationError("context contient une chaine trop longue")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HistoryValidationError("context contient un nombre non fini")
        return value
    if isinstance(value, Mapping):
        if len(value) > 128:
            raise HistoryValidationError("context contient trop de champs")
        result: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            encoded_key = str(key)
            if encoded_key in result:
                raise HistoryValidationError("context contient des cles qui entrent en collision")
            result[encoded_key] = _safe_json(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 256:
            raise HistoryValidationError("context contient trop d'elements")
        return [_safe_json(item, depth=depth + 1) for item in value]
    raise HistoryValidationError("context contient un type non JSON")


def _year(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HistoryValidationError(f"{field}.year doit etre un entier")
    if value == 0:
        raise HistoryValidationError("le calendrier civil ne contient pas d'annee zero")
    if abs(value) > _MAX_YEAR:
        raise HistoryValidationError(f"{field}.year depasse la limite")
    return value


def _date(value: Any, *, field: str) -> dict[str, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return {"year": _year(value, field=field), "precision": 9}
    if not isinstance(value, Mapping):
        raise HistoryValidationError(f"{field} doit contenir au moins year")
    year = _year(value.get("year"), field=field)
    month = value.get("month")
    day = value.get("day")
    if month is not None and (
        isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12
    ):
        raise HistoryValidationError(f"{field}.month est invalide")
    if day is not None and (
        isinstance(day, bool) or not isinstance(day, int) or not 1 <= day <= 31
    ):
        raise HistoryValidationError(f"{field}.day est invalide")
    if day is not None and month is None:
        raise HistoryValidationError(f"{field}.day exige month")
    if day is not None:
        astronomical_year = year if year > 0 else year + 1
        leap = astronomical_year % 4 == 0 and (
            astronomical_year % 100 != 0 or astronomical_year % 400 == 0
        )
        month_days = (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        if day > month_days[month - 1]:
            raise HistoryValidationError(f"{field}.day n'existe pas dans ce mois")
    result = {"year": year, "precision": 11 if day else 10 if month else 9}
    if month is not None:
        result["month"] = month
    if day is not None:
        result["day"] = day
    return result


def civil_year_ordinal(year: int) -> int:
    """Place les années civiles sur un axe continu sans année zéro."""

    checked = _year(year, field="year")
    return checked if checked > 0 else checked + 1


def _days_from_civil(year: int, month: int, day: int) -> int:
    """Nombre de jours proleptiques, avec année astronomique interne."""

    astronomical_year = year if year > 0 else year + 1
    adjusted_year = astronomical_year - (1 if month <= 2 else 0)
    era = adjusted_year // 400
    year_of_era = adjusted_year - era * 400
    shifted_month = month + (-3 if month > 2 else 9)
    day_of_year = (153 * shifted_month + 2) // 5 + day - 1
    day_of_era = (
        year_of_era * 365
        + year_of_era // 4
        - year_of_era // 100
        + day_of_year
    )
    return era * 146_097 + day_of_era


def _date_start_day(value: Mapping[str, int]) -> int:
    return _days_from_civil(value["year"], value.get("month", 1), value.get("day", 1))


def _date_position(value: Mapping[str, int]) -> tuple[int, Decimal, str]:
    precision = int(value["precision"])
    if precision == 11:
        return precision, Decimal(_date_start_day(value)), "day"
    if precision == 10:
        astronomical_year = value["year"] if value["year"] > 0 else value["year"] + 1
        return precision, Decimal(astronomical_year * 12 + value["month"] - 1), "month"
    return precision, Decimal(civil_year_ordinal(value["year"])), "year"


def _object(value: Any) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) <= _MAX_SAFE_INTEGER else str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HistoryValidationError("object contient un nombre non fini")
        return value
    if isinstance(value, str):
        return _text(value, field="object", maximum=1_000)
    raise HistoryValidationError("object doit etre une valeur scalaire")


def _string_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 128:
        raise HistoryValidationError(f"{field} doit etre une liste bornee")
    return list(dict.fromkeys(_identifier(item, field=field) for item in value))


def normalize_historical_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Valide et canonicalise un événement historique borné."""

    if not isinstance(event, Mapping):
        raise HistoryValidationError("event doit etre un objet")
    event_id = _identifier(event.get("id"), field="id")
    start = _date(event.get("valid_from"), field="valid_from")
    end = _date(event.get("valid_to"), field="valid_to") if event.get("valid_to") is not None else None
    if end is not None and _date_start_day(end) < _date_start_day(start):
        raise HistoryValidationError("valid_to precede valid_from")

    source = event.get("source")
    if not isinstance(source, Mapping):
        raise HistoryValidationError("source doit etre un objet")
    source_id = _identifier(source.get("id"), field="source.id")
    clean_source: dict[str, Any] = {"id": source_id}
    if source.get("label") is not None:
        clean_source["label"] = _text(source["label"], field="source.label", maximum=300)
    if source.get("confidence") is not None:
        confidence = _decimal(source["confidence"], field="source.confidence")
        if not Decimal(0) <= confidence <= Decimal(1):
            raise HistoryValidationError("source.confidence doit etre entre 0 et 1")
        clean_source["confidence"] = float(confidence)
    optional_source_limits = {
        "url": 2_048,
        "reference": 1_000,
        "page": 100,
        "accessed_at": 100,
        "statement_id": 300,
        "revision_id": 300,
        "license": 300,
    }
    allowed_source_fields = {"id", "label", "confidence", "metadata", *optional_source_limits}
    unknown_source_fields = sorted(str(key) for key in source if str(key) not in allowed_source_fields)
    if unknown_source_fields:
        raise HistoryValidationError(
            "source contient des champs inconnus: " + ", ".join(unknown_source_fields)
        )
    for key, maximum in optional_source_limits.items():
        if source.get(key) is not None:
            clean_source[key] = _text(source[key], field=f"source.{key}", maximum=maximum)
    if source.get("metadata") is not None:
        if not isinstance(source["metadata"], Mapping):
            raise HistoryValidationError("source.metadata doit etre un objet")
        clean_source["metadata"] = _safe_json(source["metadata"])

    recorded_order = event.get("recorded_order", 0)
    if isinstance(recorded_order, bool) or not isinstance(recorded_order, int) or recorded_order < 0:
        raise HistoryValidationError("recorded_order doit etre un entier positif ou nul")

    measurements = event.get("measurements", [])
    if not isinstance(measurements, list) or len(measurements) > _MAX_MEASUREMENTS:
        raise HistoryValidationError("measurements doit etre une liste bornee")
    clean_measurements: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, measurement in enumerate(measurements):
        if not isinstance(measurement, Mapping):
            raise HistoryValidationError(f"measurements[{index}] doit etre un objet")
        name = _identifier(measurement.get("name"), field=f"measurements[{index}].name")
        if name in names:
            raise HistoryValidationError(f"mesure dupliquee: {name}")
        names.add(name)
        number = _decimal(measurement.get("value"), field=f"measurements[{index}].value")
        unit = _text(measurement.get("unit"), field=f"measurements[{index}].unit", maximum=40)
        unit_key = unit.casefold()
        supported = unit_key in _UNIT_REGISTRY or unit_key in _TEMPERATURE_UNITS
        clean_measurement = {
            "name": name,
            "value": _number(number),
            "unit": unit,
            "calculable": supported,
        }
        if not supported:
            clean_measurement["skip_reason"] = "unsupported_or_unsourced_unit"
        clean_measurements.append(clean_measurement)

    coordinates = event.get("coordinates")
    clean_coordinates = None
    if coordinates is not None:
        if not isinstance(coordinates, Mapping):
            raise HistoryValidationError("coordinates doit etre un objet")
        latitude = _decimal(coordinates.get("latitude"), field="coordinates.latitude")
        longitude = _decimal(coordinates.get("longitude"), field="coordinates.longitude")
        if not Decimal(-90) <= latitude <= Decimal(90):
            raise HistoryValidationError("latitude doit etre entre -90 et 90")
        if not Decimal(-180) <= longitude <= Decimal(180):
            raise HistoryValidationError("longitude doit etre entre -180 et 180")
        clean_coordinates = {"latitude": float(latitude), "longitude": float(longitude)}

    birth_year = event.get("birth_year")
    if birth_year is not None:
        birth_year = _year(birth_year, field="birth_year")

    clean_context = _safe_json(event.get("context", {}))
    encoded_context = json.dumps(
        clean_context, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded_context) > 100_000:
        raise HistoryValidationError("context depasse 100000 octets")

    result: dict[str, Any] = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "id": event_id,
        "subject": _text(event.get("subject"), field="subject"),
        "predicate": _identifier(event.get("predicate"), field="predicate"),
        "object": _object(event.get("object")),
        "valid_from": start,
        "source": clean_source,
        "recorded_order": recorded_order,
        "context": clean_context,
        "measurements": clean_measurements,
        "supersedes": _string_list(event.get("supersedes"), field="supersedes"),
        "retracts": _string_list(event.get("retracts"), field="retracts"),
    }
    if end is not None:
        result["valid_to"] = end
    if clean_coordinates is not None:
        result["coordinates"] = clean_coordinates
    if birth_year is not None:
        result["birth_year"] = birth_year
    # Vérifie aussi la sérialisabilité et l'absence de NaN.
    json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return result


_UNIT_REGISTRY: Final = {
    "m": ("length", "m", Decimal("1")),
    "meter": ("length", "m", Decimal("1")),
    "metre": ("length", "m", Decimal("1")),
    "km": ("length", "m", Decimal("1000")),
    "cm": ("length", "m", Decimal("0.01")),
    "mm": ("length", "m", Decimal("0.001")),
    "kg": ("mass", "kg", Decimal("1")),
    "g": ("mass", "kg", Decimal("0.001")),
    "mg": ("mass", "kg", Decimal("0.000001")),
    "s": ("time", "s", Decimal("1")),
    "sec": ("time", "s", Decimal("1")),
    "min": ("time", "s", Decimal("60")),
    "h": ("time", "s", Decimal("3600")),
    "day": ("time", "s", Decimal("86400")),
    "jour": ("time", "s", Decimal("86400")),
    "year": ("time", "s", Decimal("31557600")),
    "annee": ("time", "s", Decimal("31557600")),
    "person": ("count", "person", Decimal("1")),
    "personne": ("count", "person", Decimal("1")),
    "count": ("count", "count", Decimal("1")),
    "%": ("percentage", "%", Decimal("1")),
}
_TEMPERATURE_UNITS: Final = {"c", "°c", "celsius", "f", "°f", "fahrenheit", "k", "kelvin"}


def _canonical_measurement(
    measurement: Mapping[str, Any],
) -> tuple[str, str, Decimal, str, bool] | None:
    name = str(measurement["name"])
    unit = str(measurement["unit"]).strip().casefold()
    value = _decimal(measurement["value"], field=f"measurement.{name}")
    if unit in _TEMPERATURE_UNITS:
        with localcontext() as context:
            context.prec = 40
            if unit in {"c", "°c", "celsius"}:
                canonical = value + Decimal("273.15")
                exact = True
            elif unit in {"f", "°f", "fahrenheit"}:
                canonical = (value - Decimal("32")) * Decimal(5) / Decimal(9) + Decimal("273.15")
                exact = False
            else:
                canonical = value
                exact = True
        if canonical < 0:
            raise HistoryValidationError(
                f"measurement.{name} est sous le zero absolu"
            )
        return name, "temperature", canonical, "K", exact
    spec = _UNIT_REGISTRY.get(unit)
    if spec is None:
        return None
    dimension, canonical_unit, factor = spec
    return name, dimension, value * factor, canonical_unit, True


def _derived(
    *,
    event_id: str,
    kind: str,
    value: Decimal,
    unit: str,
    formula: str,
    inputs: Mapping[str, Any],
    dependency_event_ids: Sequence[str],
    exact: bool,
    approximate: bool = False,
    measurement: str | None = None,
) -> dict[str, Any]:
    result = {
        "id": f"calc:{event_id}:{kind}" + (f":{measurement}" if measurement else ""),
        "event_id": event_id,
        "kind": kind,
        "value": _number(value, approximate=approximate),
        "display": format(value, ".12g") if approximate else format(value.normalize(), "f"),
        "unit": unit,
        "formula": formula,
        "inputs": _safe_json(inputs),
        "dependency_event_ids": list(dict.fromkeys(dependency_event_ids)),
        "exact": exact,
        "status": "computed_from_sources",
        "calculation_version": HISTORY_CALCULATION_VERSION,
    }
    if measurement is not None:
        result["measurement"] = measurement
    return result


def derive_calculable_data(
    event: Mapping[str, Any],
    previous_event: Mapping[str, Any] | None = None,
    birth_year: int | None = None,
) -> list[dict[str, Any]]:
    """Calcule toutes les dérivations autorisées dont les entrées existent."""

    current = normalize_historical_event(event)
    previous = normalize_historical_event(previous_event) if previous_event is not None else None
    event_id = current["id"]
    start_precision, start_axis, start_unit = _date_position(current["valid_from"])
    start_year_axis = Decimal(civil_year_ordinal(current["valid_from"]["year"]))
    calculations: list[dict[str, Any]] = []

    if current.get("valid_to") is not None:
        end_precision, end_axis, end_unit = _date_position(current["valid_to"])
        if end_precision == start_precision and end_unit == start_unit:
            duration = end_axis - start_axis
            duration_kind = {
                "year": "duration_years",
                "month": "duration_months",
                "day": "duration_days",
            }[start_unit]
            boundary_exact = start_precision == 11
            calculations.append(_derived(
                event_id=event_id, kind=duration_kind, value=duration, unit=start_unit,
                formula=f"civil_{start_unit}_ordinal(valid_to) - civil_{start_unit}_ordinal(valid_from)",
                inputs={"valid_from": current["valid_from"], "valid_to": current["valid_to"]},
                dependency_event_ids=[event_id], exact=boundary_exact,
            ))
            calculations.append(_derived(
                event_id=event_id, kind="temporal_midpoint", value=(start_axis + end_axis) / Decimal(2),
                unit=f"civil_{start_unit}_ordinal",
                formula=f"(civil_{start_unit}_ordinal(valid_from) + civil_{start_unit}_ordinal(valid_to)) / 2",
                inputs={"valid_from": current["valid_from"], "valid_to": current["valid_to"]},
                dependency_event_ids=[event_id], exact=boundary_exact,
            ))

    effective_birth = birth_year if birth_year is not None else current.get("birth_year")
    if effective_birth is not None:
        birth = _year(effective_birth, field="birth_year")
        age = start_year_axis - Decimal(civil_year_ordinal(birth))
        if age >= 0:
            calculations.append(_derived(
                event_id=event_id, kind="age_at_start", value=age, unit="year",
                formula="civil_ordinal(valid_from) - civil_ordinal(birth_year)",
                inputs={"valid_from": current["valid_from"], "birth_year": birth},
                # Seule l'année de naissance est connue; l'âge civil exact
                # dépendrait du mois et du jour de naissance.
                dependency_event_ids=[event_id], exact=False,
            ))

    current_measures: dict[str, tuple[str, str, Decimal, str, bool]] = {}
    for measurement in current["measurements"]:
        canonical = _canonical_measurement(measurement)
        if canonical is None:
            continue
        name, dimension, value, unit, conversion_exact = canonical
        current_measures[name] = canonical
        calculations.append(_derived(
            event_id=event_id, kind="normalized_measurement", value=value, unit=unit,
            formula="convert_to_canonical_unit(value, unit)",
            inputs={"name": name, "value": measurement["value"], "unit": measurement["unit"], "dimension": dimension},
            dependency_event_ids=[event_id], exact=conversion_exact, measurement=name,
        ))

    if previous is not None:
        previous_precision, previous_axis, previous_unit = _date_position(previous["valid_from"])
        interval: Decimal | None = None
        interval_unit: str | None = None
        if previous_precision == start_precision and previous_unit == start_unit:
            interval = start_axis - previous_axis
            interval_unit = start_unit
            calculations.append(_derived(
                event_id=event_id, kind="interval_from_previous", value=interval, unit=start_unit,
                formula=f"civil_{start_unit}_ordinal(valid_from) - civil_{start_unit}_ordinal(previous.valid_from)",
                inputs={"valid_from": current["valid_from"], "previous_valid_from": previous["valid_from"]},
                dependency_event_ids=[previous["id"], event_id], exact=start_precision == 11,
            ))
        previous_measures = {
            item["name"]: canonical
            for item in previous["measurements"]
            if (canonical := _canonical_measurement(item)) is not None
        }
        for name, (_, dimension, value, unit, current_exact) in current_measures.items():
            old = previous_measures.get(name)
            if old is None or old[1] != dimension or old[3] != unit:
                continue
            old_value = old[2]
            previous_exact = old[4]
            change = value - old_value
            dependencies = [previous["id"], event_id]
            calculations.append(_derived(
                event_id=event_id, kind="absolute_change", value=change, unit=unit,
                formula="current_canonical_value - previous_canonical_value",
                inputs={"measurement": name, "current": _number(value), "previous": _number(old_value)},
                dependency_event_ids=dependencies,
                exact=current_exact and previous_exact,
                measurement=name,
            ))
            if old_value != 0:
                with localcontext() as context:
                    context.prec = 40
                    percent = change / old_value * Decimal(100)
                calculations.append(_derived(
                    event_id=event_id, kind="percent_change", value=percent, unit="%",
                    formula="(current - previous) / previous * 100",
                    inputs={"measurement": name, "current": _number(value), "previous": _number(old_value)},
                    dependency_event_ids=dependencies, exact=False, measurement=name,
                ))
            if interval is not None and interval != 0 and interval_unit == "year":
                with localcontext() as context:
                    context.prec = 40
                    rate = change / interval
                calculations.append(_derived(
                    event_id=event_id, kind="annual_rate", value=rate, unit=f"{unit}/year",
                    formula="(current - previous) / elapsed_years",
                    inputs={"measurement": name, "change": _number(change), "elapsed_years": _number(interval)},
                    dependency_event_ids=dependencies, exact=False, measurement=name,
                ))

        if current.get("coordinates") is not None and previous.get("coordinates") is not None:
            first = previous["coordinates"]
            second = current["coordinates"]
            lat1, lon1, lat2, lon2 = map(math.radians, (
                first["latitude"], first["longitude"], second["latitude"], second["longitude"]
            ))
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1
            a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
            distance = _EARTH_RADIUS_KM * 2 * math.asin(min(1.0, math.sqrt(a)))
            calculations.append(_derived(
                event_id=event_id, kind="distance_from_previous", value=Decimal(str(distance)), unit="km",
                formula="haversine(previous.coordinates, coordinates, earth_radius_km)",
                inputs={"previous": first, "current": second, "earth_radius_km": _EARTH_RADIUS_KM},
                dependency_event_ids=[previous["id"], event_id], exact=False, approximate=True,
            ))

    return calculations


def _entity_identity(event: Mapping[str, Any]) -> str:
    """Sépare les homonymes quand le fait fournit un identifiant de contexte."""

    context = event.get("context")
    if isinstance(context, Mapping):
        for key in ("entity_id", "entity_key", "wikidata_id"):
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().casefold()
    return str(event["subject"]).casefold()


def _validate_corrections(
    events: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Valide le graphe de correction et retourne les faits rétractés."""

    graph: dict[str, list[str]] = {}
    retracted: set[str] = set()
    for event in events:
        event_id = str(event["id"])
        targets = [*event.get("supersedes", []), *event.get("retracts", [])]
        graph[event_id] = []
        for target in targets:
            if target not in by_id:
                raise HistoryValidationError(
                    f"{event_id} reference un fait de correction inconnu: {target}"
                )
            if target == event_id:
                raise HistoryValidationError(f"{event_id} ne peut pas se corriger lui-meme")
            graph[event_id].append(target)
        for target in event.get("supersedes", []):
            replaced = by_id[target]
            if (
                _entity_identity(event) != _entity_identity(replaced)
                or event["predicate"] != replaced["predicate"]
            ):
                raise HistoryValidationError(
                    f"{event_id} ne peut superseder un autre sujet ou predicat"
                )
        retracted.update(str(value) for value in event.get("retracts", []))

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise HistoryValidationError("le graphe de correction contient un cycle")
        if node in visited:
            return
        visiting.add(node)
        for target in graph.get(node, []):
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for event_id in graph:
        visit(event_id)
    return retracted


def build_ground_truth(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Construit un oracle indépendant, les contradictions et les dérivations."""

    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        raise HistoryValidationError("events doit etre une sequence")
    if not 1 <= len(events) <= 100_000:
        raise HistoryValidationError("events doit contenir entre 1 et 100000 faits")
    normalized = [normalize_historical_event(event) for event in events]
    by_id: dict[str, dict[str, Any]] = {}
    for event in normalized:
        if event["id"] in by_id:
            raise HistoryValidationError(f"identifiant d'evenement duplique: {event['id']}")
        by_id[event["id"]] = event
    retracted_ids = _validate_corrections(normalized, by_id)

    ordered = sorted(normalized, key=lambda item: (item["recorded_order"], item["id"]))
    calculations: list[dict[str, Any]] = []
    latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    temporal_groups: dict[
        tuple[str, str], dict[tuple[int, int, Decimal], list[dict[str, Any]]]
    ] = {}
    for event in normalized:
        if event["id"] in retracted_ids:
            continue
        key = (_entity_identity(event), event["predicate"])
        precision, position, _ = _date_position(event["valid_from"])
        temporal_groups.setdefault(key, {}).setdefault(
            (_date_start_day(event["valid_from"]), precision, position), []
        ).append(event)

    contradictions: list[dict[str, Any]] = []
    for key, by_time in sorted(temporal_groups.items()):
        entity_identity, predicate = key
        previous_unique: dict[str, Any] | None = None
        latest_group: list[dict[str, Any]] = []
        for (_, precision, position), raw_group in sorted(by_time.items()):
            group = sorted(raw_group, key=lambda item: (item["recorded_order"], item["id"]))
            superseded_here = {
                target
                for item in group
                for target in item.get("supersedes", [])
            }
            effective_group = [item for item in group if item["id"] not in superseded_here]
            if not effective_group:
                continue
            for event in effective_group:
                calculations.extend(
                    derive_calculable_data(
                        event,
                        previous_unique,
                        event.get("birth_year"),
                    )
                )
            encoded_objects = {
                json.dumps(item["object"], ensure_ascii=False, sort_keys=True)
                for item in effective_group
            }
            if len(effective_group) > 1 and len(encoded_objects) > 1:
                contradictions.append({
                    "subject": effective_group[0]["subject"],
                    "entity_identity": entity_identity,
                    "predicate": predicate,
                    "time_precision": precision,
                    "time_ordinal": _number(position),
                    "event_ids": [item["id"] for item in effective_group],
                    "objects": [item["object"] for item in effective_group],
                    "status": "preserved_not_silently_resolved",
                })
            # Deux sources au même instant ne fournissent pas un précédent
            # unique pour le prochain calcul, même lorsqu'elles s'accordent.
            previous_unique = effective_group[0] if len(effective_group) == 1 else None
            latest_group = effective_group
        if len(latest_group) == 1:
            latest_by_key[key] = latest_group[0]

    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "events": ordered,
        "by_id": by_id,
        "calculations": calculations,
        "contradictions": contradictions,
        "latest_by_subject_predicate": {
            f"{entity_identity}|{predicate}": event["id"]
            for (entity_identity, predicate), event in sorted(latest_by_key.items())
        },
    }


def calculation_catalog() -> dict[str, Any]:
    """Décrit le registre fini de calculs, sans exposer de formule exécutable."""

    calculations = [
        ("duration_years", "Durée historique en années civiles"),
        ("duration_months", "Durée historique en mois civils"),
        ("duration_days", "Durée historique exacte en jours"),
        ("temporal_midpoint", "Milieu d'un intervalle"),
        ("interval_from_previous", "Temps depuis le fait précédent"),
        ("age_at_start", "Âge à la date du fait"),
        ("normalized_measurement", "Conversion vers l'unité canonique"),
        ("absolute_change", "Variation absolue"),
        ("percent_change", "Variation en pourcentage"),
        ("annual_rate", "Taux de variation annuel"),
        ("distance_from_previous", "Distance géographique"),
    ]
    return {
        "version": HISTORY_CALCULATION_VERSION,
        "schema_version": HISTORY_SCHEMA_VERSION,
        "calculations": [{"name": name, "description": description} for name, description in calculations],
        "calculation_count": len(calculations),
        "units": {
            "canonical": ["m", "kg", "s", "K", "person", "count", "%"],
            "currencies": "not_converted_without_dated_sourced_rates",
        },
        "external_api": False,
        "free_form_formulas": False,
    }


__all__ = [
    "HISTORY_CALCULATION_VERSION",
    "HISTORY_SCHEMA_VERSION",
    "HistoryValidationError",
    "build_ground_truth",
    "calculation_catalog",
    "civil_year_ordinal",
    "derive_calculable_data",
    "normalize_historical_event",
]
