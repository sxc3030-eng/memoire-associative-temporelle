"""Import JSON deterministe vers la memoire associative.

Le serveur ne lit jamais un chemin fourni par le client : il recoit une
valeur JSON deja decodee. Chaque feuille scalaire devient un souvenir avec
une provenance explicite. L'identifiant d'import et les cles d'idempotence
sont derives du contenu canonique, ce qui lie l'aperçu aux donnees commises.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Mapping, Protocol


MAX_IMPORT_FILE_BYTES = 1024 * 1024
MAX_IMPORT_DEPTH = 32
MAX_IMPORT_NODES = 10_000
MAX_IMPORT_MEMORIES = 200
MAX_IMPORT_TOTAL_TOKENS = 10_000
MAX_MEMORY_TEXT_CHARS = 4_000
MAX_FILENAME_CHARS = 180
MAX_JSON_KEY_CHARS = 256
MAX_JSON_PATH_CHARS = 2_048
MAX_PREVIEW_ITEMS = 100
IMPORT_ALGORITHM_VERSION = "json-import-v1"

_SIMPLE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_HAS_TOKEN_RE = re.compile(r"[^\W_]", re.UNICODE)
_IMPORT_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)
_GENERIC_SECTION_KEYS = frozenset(
    {"data", "items", "records", "results", "payload", "content", "values", "entries"}
)


class JSONImportError(ValueError):
    """Erreur de validation presentable directement au client local."""


class _MemoryEngine(Protocol):
    def observe(
        self,
        text: str,
        episode_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        source: str | Mapping[str, Any] = "user_confirmed",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class JSONMemoryItem:
    path: str
    friendly_path: str
    value_type: str
    value_preview: str
    text: str
    text_truncated: bool
    token_count: int
    category: str
    category_label: str
    category_reason: str


@dataclass(frozen=True)
class JSONImportPlan:
    data_digest: str
    import_id: str
    filename: str
    canonical_size_bytes: int
    root_type: str
    node_count: int
    max_depth: int
    items: tuple[JSONMemoryItem, ...]


def _normalise_for_matching(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _slug(value: str, *, fallback: str = "general") -> str:
    normalised = _normalise_for_matching(value)
    result = _SLUG_RE.sub("-", normalised).strip("-")
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    if not result:
        return f"{fallback}-{suffix}"
    if len(result) > 48:
        return f"{result[:39].rstrip('-')}-{suffix}"
    return result


def _clean_filename(filename: Any) -> str:
    if filename is None:
        return "import.json"
    if not isinstance(filename, str):
        raise JSONImportError("filename doit etre une chaine")
    # Les deux separateurs sont traites comme des separateurs, quel que soit
    # le systeme. Le nom reste une simple metadonnee et n'est jamais ouvert.
    candidate = unicodedata.normalize("NFC", filename).replace("\\", "/").split("/")[-1]
    candidate = "".join(character for character in candidate if ord(character) >= 32).strip()
    if not candidate:
        candidate = "import.json"
    if len(candidate) > MAX_FILENAME_CHARS:
        candidate = candidate[:MAX_FILENAME_CHARS].rstrip()
    return candidate or "import.json"


def decode_json_import_content(content: Any) -> Any:
    """Decode un document JSON texte sans perdre les grands entiers.

    Le navigateur envoie le texte original plutot qu'un objet issu de
    ``JSON.parse`` : les entiers superieurs a 2**53 restent ainsi exacts.
    """

    if not isinstance(content, str):
        raise JSONImportError("content doit etre une chaine JSON")
    document = content.removeprefix("\ufeff")
    try:
        raw_size = len(document.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise JSONImportError("Le document contient un caractere Unicode invalide") from error
    if raw_size > MAX_IMPORT_FILE_BYTES:
        raise JSONImportError(
            f"Le document depasse la limite de {MAX_IMPORT_FILE_BYTES} octets"
        )
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                clean_key = " ".join(key.split())[:80] or "(cle vide)"
                raise JSONImportError(f"Cle JSON dupliquee: {clean_key}")
            result[key] = value
        return result

    try:
        data = json.loads(
            document,
            parse_constant=lambda value: (_ for _ in ()).throw(
                JSONImportError(f"Constante JSON invalide: {value}")
            ),
            object_pairs_hook=unique_object,
        )
    except JSONImportError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise JSONImportError("Le contenu n'est pas un JSON valide") from error
    if not isinstance(data, (dict, list)):
        raise JSONImportError("Le document doit contenir un objet ou un tableau JSON")
    return data


def _json_path_key(path: str, key: str) -> str:
    if _SIMPLE_KEY_RE.fullmatch(key):
        result = f"{path}.{key}"
    else:
        encoded = json.dumps(key, ensure_ascii=False)
        result = f"{path}[{encoded}]"
    if len(result) > MAX_JSON_PATH_CHARS:
        raise JSONImportError(
            f"Un chemin JSON depasse la limite de {MAX_JSON_PATH_CHARS} caracteres"
        )
    return result


def _friendly_path(segments: tuple[str | int, ...]) -> str:
    parts: list[str] = []
    for item in segments:
        if isinstance(item, int):
            parts.append(f"element {item + 1}")
            continue
        printable = "".join(
            " " if unicodedata.category(character).startswith("C") else character
            for character in item
        )
        parts.append(" ".join(printable.split()) or "cle sans nom")
    return ", ".join(parts) or "racine"


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _display_value(value: Any) -> str:
    if value is None:
        return "valeur nulle"
    if value is True:
        return "vrai"
    if value is False:
        return "faux"
    if isinstance(value, str):
        return value if value else "chaine vide"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


_CATEGORY_RULES: tuple[tuple[str, str, frozenset[str]], ...] = (
    (
        "identite",
        "Identite",
        frozenset(
            {
                "nom", "name", "prenom", "firstname", "lastname", "fullname",
                "username", "utilisateur", "email", "courriel", "telephone", "phone",
                "mobile", "anniversaire", "birthday", "naissance", "birth",
            }
        ),
    ),
    (
        "temps",
        "Temps",
        frozenset(
            {
                "date", "time", "heure", "timestamp", "created", "createdat",
                "updated", "updatedat", "year", "annee", "month", "mois", "day",
                "jour", "deadline", "echeance", "duration", "duree",
            }
        ),
    ),
    (
        "localisation",
        "Localisation",
        frozenset(
            {
                "adresse", "address", "ville", "city", "pays", "country", "province",
                "state", "postal", "zipcode", "zip", "latitude", "longitude", "lat",
                "lon", "lng", "location", "lieu",
            }
        ),
    ),
    (
        "preference",
        "Preference",
        frozenset(
            {
                "preference", "preferences", "favorite", "favourite", "favori",
                "aime", "likes", "like", "dislike", "souhait", "wish", "choix",
            }
        ),
    ),
    (
        "relation",
        "Relation",
        frozenset(
            {
                "relation", "relations", "parent", "parents", "enfant", "children",
                "conjoint", "spouse", "ami", "amis", "friend", "friends", "famille",
                "family", "contact", "contacts",
            }
        ),
    ),
    (
        "finance",
        "Finance",
        frozenset(
            {
                "prix", "price", "cout", "cost", "montant", "amount", "total",
                "salaire", "salary", "revenu", "income", "devise", "currency",
                "budget", "balance", "solde",
            }
        ),
    ),
    (
        "activite",
        "Activite",
        frozenset(
            {
                "activite", "activity", "action", "event", "evenement", "task", "tache",
                "projet", "project", "status", "statut", "objectif", "goal", "message",
                "note", "description",
            }
        ),
    ),
)


def _heuristic_category(segments: tuple[str | int, ...], value: Any) -> tuple[str, str, str]:
    keys: list[str] = []
    for item in reversed(segments):
        if not isinstance(item, str):
            continue
        parts = [
            part
            for part in _SLUG_RE.split(_normalise_for_matching(item))
            if part
        ]
        keys.extend(parts)
        joined = "".join(parts)
        if joined and joined not in parts:
            keys.append(joined)
    for category, label, keywords in _CATEGORY_RULES:
        for key in keys:
            if key in keywords:
                return category, label, f"cle JSON classee comme {label.lower()}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "mesure", "Mesure", "valeur numerique"
    return "general", "General", "aucune regle plus specifique"


def _category(
    section: str | None,
    segments: tuple[str | int, ...],
    value: Any,
) -> tuple[str, str, str]:
    section_key = (
        _SLUG_RE.sub("", _normalise_for_matching(section)) if section is not None else None
    )
    if section is not None and section_key not in _GENERIC_SECTION_KEYS:
        label = " ".join(part for part in re.split(r"[_\-\s]+", section) if part).strip()
        label = (label[:80] or "Section").capitalize()
        return _slug(section, fallback="section"), label, f"section racine {section!r}"
    return _heuristic_category(segments, value)


def _walk_json(data: Any) -> tuple[list[JSONMemoryItem], int, int]:
    if not isinstance(data, (dict, list)):
        raise JSONImportError("data doit contenir un objet ou un tableau JSON")

    items: list[JSONMemoryItem] = []
    node_count = 0
    deepest = 0
    total_tokens = 0
    ancestors: set[int] = set()

    def walk(
        value: Any,
        path: str,
        segments: tuple[str | int, ...],
        depth: int,
        section: str | None,
    ) -> None:
        nonlocal node_count, deepest, total_tokens
        node_count += 1
        if node_count > MAX_IMPORT_NODES:
            raise JSONImportError(
                f"Le document depasse la limite de {MAX_IMPORT_NODES} noeuds JSON"
            )
        deepest = max(deepest, depth)
        if depth > MAX_IMPORT_DEPTH:
            raise JSONImportError(
                f"Le document depasse la profondeur maximale de {MAX_IMPORT_DEPTH}"
            )

        if isinstance(value, dict):
            identity = id(value)
            if identity in ancestors:
                raise JSONImportError("Une reference cyclique n'est pas un JSON valide")
            ancestors.add(identity)
            try:
                for key in sorted(value, key=lambda item: str(item)):
                    if not isinstance(key, str):
                        raise JSONImportError("Toutes les cles JSON doivent etre des chaines")
                    if len(key) > MAX_JSON_KEY_CHARS:
                        raise JSONImportError(
                            f"Une cle JSON depasse la limite de {MAX_JSON_KEY_CHARS} caracteres"
                        )
                    child = value[key]
                    child_section = section
                    if depth == 0 and isinstance(child, (dict, list)):
                        child_section = key
                    walk(
                        child,
                        _json_path_key(path, key),
                        (*segments, key),
                        depth + 1,
                        child_section,
                    )
            finally:
                ancestors.remove(identity)
            return

        if isinstance(value, list):
            identity = id(value)
            if identity in ancestors:
                raise JSONImportError("Une reference cyclique n'est pas un JSON valide")
            ancestors.add(identity)
            try:
                for index, child in enumerate(value):
                    child_path = f"{path}[{index}]"
                    if len(child_path) > MAX_JSON_PATH_CHARS:
                        raise JSONImportError(
                            f"Un chemin JSON depasse la limite de {MAX_JSON_PATH_CHARS} caracteres"
                        )
                    walk(child, child_path, (*segments, index), depth + 1, section)
            finally:
                ancestors.remove(identity)
            return

        if value is None:
            # Une absence de valeur n'est pas transformee en fait positif.
            return
        if isinstance(value, str) and not value.strip():
            # Une chaine vide n'apporte aucun indice rappelable au prototype.
            return
        if not isinstance(value, (str, bool, int, float)):
            raise JSONImportError(f"Type JSON non pris en charge: {type(value).__name__}")
        if isinstance(value, float) and not math.isfinite(value):
            raise JSONImportError("NaN et Infinity ne sont pas des nombres JSON valides")
        if len(items) >= MAX_IMPORT_MEMORIES:
            raise JSONImportError(
                f"Le document depasse la limite de {MAX_IMPORT_MEMORIES} souvenirs"
            )

        friendly = _friendly_path(segments)
        display = _display_value(value)
        prefix = f"{friendly} : "
        if not _HAS_TOKEN_RE.search(prefix + display):
            prefix = f"Valeur JSON, {friendly} : "
        available = max(1, MAX_MEMORY_TEXT_CHARS - len(prefix))
        truncated = len(display) > available
        if truncated:
            raise JSONImportError(
                f"La valeur au chemin {path} depasse la limite de "
                f"{MAX_MEMORY_TEXT_CHARS} caracteres par souvenir"
            )
        text = prefix + display
        token_count = len(_IMPORT_TOKEN_RE.findall(unicodedata.normalize("NFKC", text)))
        total_tokens += token_count
        if total_tokens > MAX_IMPORT_TOTAL_TOKENS:
            raise JSONImportError(
                "Le document depasse le budget total de "
                f"{MAX_IMPORT_TOTAL_TOKENS} concepts textuels"
            )
        category, category_label, reason = _category(section, segments, value)
        items.append(
            JSONMemoryItem(
                path=path,
                friendly_path=friendly,
                value_type=_value_type(value),
                value_preview=(display[:239] + "…" if len(display) > 240 else display),
                text=text,
                text_truncated=truncated,
                token_count=token_count,
                category=category,
                category_label=category_label,
                category_reason=reason,
            )
        )

    walk(data, "$", (), 0, None)
    return items, node_count, deepest


def prepare_json_import(data: Any, *, filename: Any = None) -> JSONImportPlan:
    """Valide et transforme ``data`` sans modifier la memoire."""

    clean_filename = _clean_filename(filename)
    try:
        canonical = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError) as error:
        raise JSONImportError("data ne contient pas un JSON valide") from error
    if len(canonical) > MAX_IMPORT_FILE_BYTES:
        raise JSONImportError(
            f"Le document depasse la limite de {MAX_IMPORT_FILE_BYTES} octets"
        )
    items, node_count, deepest = _walk_json(data)
    digest = hashlib.sha256(canonical).hexdigest()
    filename_digest = hashlib.sha256(clean_filename.encode("utf-8")).hexdigest()[:12]
    return JSONImportPlan(
        data_digest=digest,
        import_id=f"json-v1-{digest}-{filename_digest}",
        filename=clean_filename,
        canonical_size_bytes=len(canonical),
        root_type="object" if isinstance(data, dict) else "array",
        node_count=node_count,
        max_depth=deepest,
        items=tuple(items),
    )


def _category_summary(plan: JSONImportPlan) -> list[dict[str, Any]]:
    categories: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for item in plan.items:
        category = categories.setdefault(
            item.category,
            {
                "id": item.category,
                "label": item.category_label,
                "count": 0,
                "reason": item.category_reason,
                "examples": [],
            },
        )
        category["count"] += 1
        if len(category["examples"]) < 3:
            category["examples"].append(
                {"path": item.path, "value": item.value_preview, "type": item.value_type}
            )
    return sorted(categories.values(), key=lambda item: (-item["count"], item["id"]))


def preview_json_import(plan: JSONImportPlan) -> dict[str, Any]:
    """Retourne un aperçu borne du plan valide."""

    preview_items = [
        {
            "path": item.path,
            "text": item.text,
            "value": item.value_preview,
            "value_type": item.value_type,
            "text_truncated": item.text_truncated,
            "category": item.category,
            "category_label": item.category_label,
            "category_reason": item.category_reason,
        }
        for item in plan.items[:MAX_PREVIEW_ITEMS]
    ]
    return {
        "import_id": plan.import_id,
        "digest": plan.data_digest,
        "filename": plan.filename,
        "summary": {
            "root_type": plan.root_type,
            "canonical_size_bytes": plan.canonical_size_bytes,
            "nodes": plan.node_count,
            "max_depth": plan.max_depth,
            "memories": len(plan.items),
            "tokens": sum(item.token_count for item in plan.items),
            "categories": len({item.category for item in plan.items}),
        },
        "categories": _category_summary(plan),
        "items": preview_items,
        "items_truncated": len(plan.items) > len(preview_items),
        "limits": {
            "file_bytes": MAX_IMPORT_FILE_BYTES,
            "depth": MAX_IMPORT_DEPTH,
            "nodes": MAX_IMPORT_NODES,
            "memories": MAX_IMPORT_MEMORIES,
            "tokens": MAX_IMPORT_TOTAL_TOKENS,
            "memory_text_chars": MAX_MEMORY_TEXT_CHARS,
            "preview_items": MAX_PREVIEW_ITEMS,
        },
        "algorithm_version": IMPORT_ALGORITHM_VERSION,
    }


def commit_json_import(
    engine: _MemoryEngine,
    plan: JSONImportPlan,
    *,
    import_id: Any,
) -> dict[str, Any]:
    """Enregistre le plan; un nouvel appel identique ne cree aucun doublon.

    Les observations sont commises sequentiellement par le moteur existant.
    Une interruption peut donc laisser un import partiel, mais la reprise avec
    le meme document est sure : chaque feuille utilise une cle idempotente.
    """

    if not isinstance(import_id, str) or not import_id.strip():
        raise JSONImportError("import_id retourne par l'aperçu est requis pour confirmer")
    if import_id.strip() != plan.import_id:
        raise JSONImportError("import_id ne correspond pas au contenu JSON courant")

    results: list[dict[str, Any]] = []
    created = 0
    duplicates = 0
    for item in plan.items:
        path_digest = hashlib.sha256(item.path.encode("utf-8")).hexdigest()
        # Une feuille JSON n'est pas une etape temporelle d'une autre feuille.
        # Chaque fait recoit donc son propre episode; la categorie reste dans
        # le contexte et ne cree pas de transitions artificielles entre champs.
        episode_id = f"json-{plan.data_digest[:24]}-{path_digest[:24]}"
        idempotency_key = f"json:{plan.data_digest}:{path_digest}"
        metadata = {
            "origin": "json_import",
            "category": item.category,
            "json_path": item.path,
            "filename": plan.filename,
            "digest": plan.data_digest,
            "json_import": {
                "algorithm_version": IMPORT_ALGORITHM_VERSION,
                "import_id": plan.import_id,
                "digest": plan.data_digest,
                "filename": plan.filename,
                "json_path": item.path,
                "friendly_path": item.friendly_path,
                "category": item.category,
                "category_label": item.category_label,
                "category_reason": item.category_reason,
                "value_type": item.value_type,
                "text_truncated": item.text_truncated,
            },
        }
        source = {
            "type": "user_confirmed",
            "medium": "json_import",
            "algorithm_version": IMPORT_ALGORITHM_VERSION,
            "import_id": plan.import_id,
            "digest": plan.data_digest,
            "filename": plan.filename,
            "json_path": item.path,
            "category": item.category,
        }
        observed = engine.observe(
            item.text,
            episode_id=episode_id,
            context=metadata,
            source=source,
            idempotency_key=idempotency_key,
        )
        duplicate = bool(observed.get("duplicate"))
        duplicates += int(duplicate)
        created += int(not duplicate)
        results.append(
            {
                "event_id": observed.get("event_id"),
                "episode_id": observed.get("episode_id"),
                "path": item.path,
                "category": item.category,
                "created": not duplicate,
                "duplicate": duplicate,
            }
        )

    response = preview_json_import(plan)
    response.update(
        {
            "created": created,
            "duplicates": duplicates,
            "imported": len(results),
            "results": results,
            "commit_atomic": False,
            "resume_safe": True,
        }
    )
    return response
