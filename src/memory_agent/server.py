"""Petit serveur HTTP local pour dialoguer avec :class:`MemoryEngine`.

Le module n'utilise que la bibliotheque standard. Il peut etre lance avec::

    python -m memory_agent.server --port 8765 --db data/memory.sqlite3

L'interface n'est volontairement pas un agent generatif : elle classe des
souvenirs et des continuations deja observes par le moteur.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import re
import socket
import threading
import time
import unicodedata
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import UUID, uuid4

from memory_agent.json_import import JSONImportError, decode_json_import_content
from memory_agent.memory import MemoryEngine


LOGGER = logging.getLogger("memory_agent.server")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_ROOT = PROJECT_ROOT / "web"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "memory.sqlite3"

MAX_BODY_BYTES = 3 * 1024 * 1024
MAX_CHAT_BODY_BYTES = 16 * 1024
MAX_MESSAGE_CHARS = 4_000
MAX_STATIC_BYTES = 2 * 1024 * 1024
MAX_MEMORIES = 100


_OBSERVE_RE = re.compile(
    r"^\s*(?:souviens[\s-]*toi|m[ée]morise)\s+que\s+(.+?)\s*[.!]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_RECALL_RE = re.compile(
    r"^\s*de\s+quoi\s+te\s+souviens[\s-]*tu"
    r"(?:\s+(?:[àa]\s+propos\s+de|au\s+sujet\s+de|sur|de|du|des))?"
    r"\s*(.*?)\s*[?!.]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_KNOW_RE = re.compile(
    r"^\s*que\s+sais[\s-]*tu"
    r"(?:\s+(?:[àa]\s+propos\s+de|au\s+sujet\s+de|sur|de|du|des))?"
    r"\s*(.*?)\s*[?!.]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_REMIND_RE = re.compile(
    r"^\s*rappelle[\s-]*moi"
    r"(?:\s+(?:[àa]\s+propos\s+de|au\s+sujet\s+de|sur|de|du|des))?"
    r"\s*(.*?)\s*[?!.]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_EXPLAIN_RE = re.compile(
    r"^\s*explique\s+pourquoi\s+tu\s+te\s+souviens"
    r"(?:\s+(?:[àa]\s+propos\s+de|au\s+sujet\s+de|sur|de|du|des))?"
    r"\s*(.*?)\s*[?!.]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_PREDICT_RE = re.compile(
    r"^\s*qu(?:['’<]\s*)?est[\s-]*ce\s+qui\s+vient\s+apr[eè]s\s+"
    r"(.+?)\s*[?!.]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FORGET_RE = re.compile(
    r"^\s*oublie\s+(?:(?:l['’<])?[ée]v[ée]nement\s+)?"
    r"([A-Za-z0-9][A-Za-z0-9._:-]{0,127})\s*[?!.]?\s*$",
    re.IGNORECASE,
)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _host_authority(value: str | None) -> tuple[str, int | None] | None:
    """Parse une autorite HTTP et refuse les formes ambigues."""

    if not value:
        return None
    candidate = value.strip()
    if not candidate or "," in candidate or any(character.isspace() for character in candidate):
        return None
    try:
        parsed = urlsplit(f"//{candidate}")
        port = parsed.port
    except ValueError:
        return None
    if (
        not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return None
    hostname = (parsed.hostname or "").rstrip(".").casefold()
    if hostname not in _LOOPBACK_HOSTS:
        return None
    return hostname, port


def _json_default(value: Any) -> Any:
    """Convertit seulement les types usuels et previsibles de la couche DB."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Path, UUID)):
        return str(value)
    if isinstance(value, set):
        return sorted(value, key=str)
    raise TypeError(f"Type non serialisable: {type(value).__name__}")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _items(value: Any, possible_keys: Iterable[str]) -> list[dict[str, Any]]:
    """Normalise les resultats du moteur durant l'evolution du prototype."""

    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    for key in possible_keys:
        nested = value.get(key)
        if isinstance(nested, (list, tuple)):
            return [item for item in nested if isinstance(item, dict)]
    # Certains prototypes retournent directement un souvenir unique.
    if any(key in value for key in ("event_id", "episode_id", "text", "concept")):
        return [value]
    return []


def _clean_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _score_text(value: Any) -> str:
    try:
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return "?"


def _memory_reply(results: Any, *, generic: bool = False) -> str:
    memories = _items(results, ("results", "memories", "ranked_episodes", "episodes"))
    prefix = ""
    if generic:
        prefix = (
            "Je suis un moteur de mémoire associative, pas un modèle de langage. "
            "Je cherche seulement dans ce qui m’a été confié. "
        )
    if not memories:
        return prefix + "Je ne trouve aucun souvenir associé à cette demande."

    lines: list[str] = []
    for index, memory in enumerate(memories[:5], start=1):
        text = _clean_text(memory.get("text") or memory.get("label") or memory.get("concept"))
        if not text:
            events = memory.get("events")
            if isinstance(events, list):
                text = " → ".join(
                    _clean_text(event.get("text") if isinstance(event, dict) else event, 80)
                    for event in events
                )
        if not text:
            text = "Souvenir sans libellé"
        score = memory.get("score")
        suffix = f" (score relatif : {_score_text(score)})" if score is not None else ""
        lines.append(f"{index}. {text}{suffix}")
    return prefix + "Voici ce que ma mémoire retrouve :\n" + "\n".join(lines)


def _prediction_reply(results: Any) -> str:
    predictions = _items(results, ("results", "predictions", "candidates", "continuations"))
    if not predictions:
        return "Je n’ai pas encore observé de suite suffisamment liée à cet historique."

    lines: list[str] = []
    for index, candidate in enumerate(predictions[:5], start=1):
        label = _clean_text(candidate.get("label") or candidate.get("concept") or candidate.get("text"))
        if not label:
            label = "Candidat sans libellé"
        score = candidate.get("score", candidate.get("relative_score"))
        score_part = f" — score relatif {_score_text(score)}" if score is not None else ""
        support = candidate.get("support_count", candidate.get("support"))
        support_part = f", support {support}" if isinstance(support, (int, float)) else ""
        lines.append(f"{index}. {label}{score_part}{support_part}")
    return (
        "D’après les chemins déjà mémorisés, la suite la mieux classée est :\n"
        + "\n".join(lines)
        + "\nCes valeurs sont des scores de classement, pas des probabilités."
    )


def _evidence_ids(candidate: dict[str, Any], limit: int = 20) -> list[Any]:
    """Retourne des references de preuve sans recopier leur contenu."""

    evidence = candidate.get("evidence")
    if not isinstance(evidence, list):
        return []
    identifiers: list[Any] = []
    seen: set[str] = set()
    for proof in evidence:
        if not isinstance(proof, dict):
            continue
        identifier = next(
            (
                proof.get(key)
                for key in ("evidence_id", "occurrence_id", "event_id")
                if proof.get(key) is not None
            ),
            None,
        )
        if identifier is None or str(identifier) in seen:
            continue
        identifiers.append(identifier)
        seen.add(str(identifier))
        if len(identifiers) >= limit:
            break
    return identifiers


def _algorithm_version(candidate: dict[str, Any]) -> Any:
    explanation = candidate.get("explanation")
    if isinstance(explanation, dict):
        return explanation.get("algorithm_version")
    return None


def _compact_recall_details(results: Any, intent: str = "recall") -> dict[str, Any]:
    memories = _items(results, ("results", "memories", "ranked_episodes", "episodes"))
    traces: list[dict[str, Any]] = []
    for memory in memories[:5]:
        path = memory.get("path", memory.get("paths", []))
        if not isinstance(path, list):
            path = []
        traces.append(
            {
                "episode_id": memory.get("episode_id"),
                "score": memory.get("score"),
                "path": path[:20],
                "evidence_ids": _evidence_ids(memory),
                "algorithm_version": _algorithm_version(memory),
            }
        )
    return {"intent": intent, "recalled": len(memories), "traces": traces}


def _compact_prediction_details(results: Any) -> dict[str, Any]:
    predictions = _items(results, ("results", "predictions", "candidates", "continuations"))
    traces: list[dict[str, Any]] = []
    for candidate in predictions[:5]:
        path = candidate.get("path", [])
        if not isinstance(path, list):
            path = []
        traces.append(
            {
                "concept": candidate.get("concept"),
                "score": candidate.get("score", candidate.get("relative_score")),
                "path": path[:20],
                "support_count": candidate.get("support_count"),
                "episode_support_count": candidate.get("episode_support_count"),
                "evidence_ids": _evidence_ids(candidate),
                "algorithm_version": _algorithm_version(candidate),
            }
        )
    return {"intent": "predict", "candidates": len(predictions), "traces": traces}


def _explanation_reply(results: Any) -> str:
    reply = _memory_reply(results)
    memories = _items(results, ("results", "memories", "ranked_episodes", "episodes"))
    if not memories:
        return reply
    explanation = memories[0].get("explanation")
    summary = explanation.get("summary") if isinstance(explanation, dict) else None
    if summary:
        return reply + "\nPourquoi ce résultat arrive en tête : " + _clean_text(summary, 300)
    return reply + "\nCe résultat arrive en tête selon ses correspondances, son support et sa récence."


class MemoryHTTPServer(ThreadingHTTPServer):
    """Serveur partageant un moteur protege par un verrou."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        engine: MemoryEngine,
        web_root: Path = DEFAULT_WEB_ROOT,
    ) -> None:
        host = str(server_address[0]).strip().casefold()
        if host not in _LOOPBACK_HOSTS:
            raise ValueError(
                "Ce prototype sans authentification doit écouter uniquement sur "
                "127.0.0.1, localhost ou ::1."
            )
        # TCPServer consulte cet attribut d'instance lors de la creation de la
        # socket, ce qui permet de prendre correctement en charge ::1.
        self.address_family = socket.AF_INET6 if host == "::1" else socket.AF_INET
        super().__init__(server_address, MemoryRequestHandler)
        self.engine = engine
        self.engine_lock = threading.RLock()
        self.web_root = web_root.resolve()
        self.started_at = time.monotonic()
        # Une execution locale represente une conversation. Tous ses messages
        # confirmes appartiennent au meme episode et apprennent donc aussi les
        # transitions entre deux appels distincts a ``observe``.
        self.conversation_episode_id = str(uuid4())


class MemoryRequestHandler(BaseHTTPRequestHandler):
    """Routes JSON et fichiers statiques du prototype."""

    server: MemoryHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Ne jamais journaliser le corps des messages ou les souvenirs.
        LOGGER.info("%s - %s", self.client_address[0], fmt % args)

    def _send_bytes(
        self,
        status: HTTPStatus | int,
        body: bytes,
        content_type: str,
        *,
        head_only: bool = False,
        cache: str = "no-store",
    ) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'",
        )
        self.end_headers()
        if not head_only and body:
            self.wfile.write(body)

    def _send_json(
        self,
        status: HTTPStatus | int,
        payload: Any,
        *,
        head_only: bool = False,
    ) -> None:
        self._send_bytes(
            status,
            _json_bytes(payload),
            "application/json; charset=utf-8",
            head_only=head_only,
        )

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _same_origin_or_non_browser(self) -> bool:
        host_authority = _host_authority(self.headers.get("Host"))
        if host_authority is None:
            return False
        host_name, host_port = host_authority
        expected_port = int(self.server.server_port)
        if (host_port if host_port is not None else 80) != expected_port:
            return False

        origin = self.headers.get("Origin")
        if origin is None:
            return True
        try:
            parsed = urlsplit(origin)
            origin_port = parsed.port
        except ValueError:
            return False
        origin_name = (parsed.hostname or "").rstrip(".").casefold()
        return (
            parsed.scheme.casefold() == "http"
            and parsed.username is None
            and parsed.password is None
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and origin_name == host_name
            and origin_name in _LOOPBACK_HOSTS
            and (origin_port if origin_port is not None else 80) == expected_port
        )

    def _read_json_object(self, *, max_bytes: int = MAX_BODY_BYTES) -> dict[str, Any] | None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type application/json requis.")
            return None
        if self.headers.get("Transfer-Encoding"):
            self._error(HTTPStatus.BAD_REQUEST, "Le transfert segmenté n’est pas accepté.")
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._error(HTTPStatus.LENGTH_REQUIRED, "Content-Length requis.")
            return None
        try:
            length = int(raw_length)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "Content-Length invalide.")
            return None
        if length < 0 or length > max_bytes:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Requête trop volumineuse.")
            return None
        try:
            raw = self.rfile.read(length)
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"Constante JSON invalide: {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._error(HTTPStatus.BAD_REQUEST, "Corps JSON invalide.")
            return None
        if not isinstance(payload, dict):
            self._error(HTTPStatus.BAD_REQUEST, "Un objet JSON est requis.")
            return None
        return payload

    def do_GET(self) -> None:  # noqa: N802 - API de BaseHTTPRequestHandler
        self._handle_get(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802 - API de BaseHTTPRequestHandler
        self._handle_get(head_only=True)

    def _handle_get(self, *, head_only: bool) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path.startswith("/api/") and not self._same_origin_or_non_browser():
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"ok": False, "error": "Hote ou origine non autorise."},
                head_only=head_only,
            )
            return
        if path == "/favicon.ico":
            self._send_bytes(
                HTTPStatus.NO_CONTENT,
                b"",
                "image/x-icon",
                head_only=head_only,
                cache="public, max-age=86400",
            )
            return
        if path == "/api/health":
            try:
                with self.server.engine_lock:
                    stats = self.server.engine.stats()
            except Exception:
                LOGGER.exception("La verification du moteur a echoue")
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "status": "unavailable"},
                    head_only=head_only,
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "status": "ready",
                    "uptime_seconds": round(time.monotonic() - self.server.started_at, 3),
                    "engine": "memory",
                    "stats_available": isinstance(stats, dict),
                },
                head_only=head_only,
            )
            return

        if path == "/api/stats":
            try:
                with self.server.engine_lock:
                    stats = self.server.engine.stats()
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "stats": stats},
                    head_only=head_only,
                )
            except Exception:
                LOGGER.exception("Impossible de lire les statistiques")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "Statistiques indisponibles."},
                    head_only=head_only,
                )
            return

        if path == "/api/memories":
            query = parse_qs(parsed.query, keep_blank_values=False)
            try:
                limit = int(query.get("limit", ["20"])[0])
            except ValueError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "Le paramètre limit doit être un entier."},
                    head_only=head_only,
                )
                return
            limit = min(max(limit, 1), MAX_MEMORIES)
            try:
                with self.server.engine_lock:
                    memories = self._list_memories(limit)
                items = _items(
                    memories,
                    ("results", "memories", "ranked_episodes", "episodes", "events"),
                )
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "memories": items, "count": len(items)},
                    head_only=head_only,
                )
            except Exception:
                LOGGER.exception("Impossible de lister les souvenirs")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "Souvenirs indisponibles."},
                    head_only=head_only,
                )
            return

        if path.startswith("/api/"):
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": "Route API inconnue."},
                head_only=head_only,
            )
            return
        self._serve_static(path, head_only=head_only)

    def _list_memories(self, limit: int) -> Any:
        engine = self.server.engine
        for method_name in ("list_memories", "recent", "memories"):
            method = getattr(engine, method_name, None)
            if callable(method):
                try:
                    return method(limit=limit)
                except TypeError:
                    return method(limit)
        return engine.recall("", top_k=limit)

    def _serve_static(self, request_path: str, *, head_only: bool) -> None:
        relative = "index.html" if request_path in {"", "/"} else unquote(request_path.lstrip("/"))
        if "\x00" in relative:
            self._error(HTTPStatus.BAD_REQUEST, "Chemin invalide.")
            return
        try:
            candidate = (self.server.web_root / relative).resolve()
            candidate.relative_to(self.server.web_root)
        except (OSError, ValueError):
            self._error(HTTPStatus.NOT_FOUND, "Fichier introuvable.")
            return
        if not candidate.is_file():
            self._error(HTTPStatus.NOT_FOUND, "Fichier introuvable.")
            return
        try:
            size = candidate.stat().st_size
            if size > MAX_STATIC_BYTES:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Fichier statique trop volumineux.")
                return
            body = candidate.read_bytes()
        except OSError:
            LOGGER.exception("Impossible de lire un fichier statique")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Interface indisponible.")
            return
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime in {"application/javascript", "application/json"}:
            mime += "; charset=utf-8"
        cache = "public, max-age=3600" if candidate.name != "index.html" else "no-store"
        self._send_bytes(HTTPStatus.OK, body, mime, head_only=head_only, cache=cache)

    def do_POST(self) -> None:  # noqa: N802 - API de BaseHTTPRequestHandler
        path = urlsplit(self.path).path
        if path not in {"/api/chat", "/api/import"}:
            if path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "Route API inconnue.")
            else:
                self._error(HTTPStatus.METHOD_NOT_ALLOWED, "Méthode non permise.")
            return
        if not self._same_origin_or_non_browser():
            self._error(HTTPStatus.FORBIDDEN, "Origine non autorisée.")
            return
        payload = self._read_json_object(
            max_bytes=MAX_CHAT_BODY_BYTES if path == "/api/chat" else MAX_BODY_BYTES
        )
        if payload is None:
            return
        if path == "/api/import":
            self._handle_json_import(payload)
            return
        message = payload.get("message")
        if not isinstance(message, str):
            self._error(HTTPStatus.BAD_REQUEST, "Le champ message doit être une chaîne.")
            return
        message = unicodedata.normalize("NFC", message).strip()
        if not message:
            self._error(HTTPStatus.BAD_REQUEST, "Le message ne peut pas être vide.")
            return
        if len(message) > MAX_MESSAGE_CHARS:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Le message est trop long.")
            return

        try:
            response = self._chat(message)
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, _clean_text(error, 300) or "Demande invalide.")
            return
        except Exception:
            LOGGER.exception("Erreur du moteur pendant une conversation")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Le moteur de mémoire a rencontré une erreur.")
            return
        self._send_json(HTTPStatus.OK, {"ok": True, **response})

    def _handle_json_import(self, payload: dict[str, Any]) -> None:
        mode = payload.get("mode")
        if mode not in {"preview", "commit"}:
            self._error(HTTPStatus.BAD_REQUEST, "mode doit etre 'preview' ou 'commit'.")
            return
        has_data = "data" in payload
        has_content = "content" in payload
        if has_data == has_content:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "Fournissez exactement un champ data ou content.",
            )
            return
        filename = payload.get("filename")
        try:
            data = (
                decode_json_import_content(payload["content"])
                if has_content
                else payload["data"]
            )
            with self.server.engine_lock:
                if mode == "preview":
                    result = self.server.engine.preview_json_import(
                        data,
                        filename=filename,
                    )
                else:
                    import_id = payload.get("import_id")
                    if not isinstance(import_id, str) or not import_id.strip():
                        raise JSONImportError(
                            "import_id retourne par l'aperçu est requis pour confirmer"
                        )
                    result = self.server.engine.import_json(
                        data,
                        import_id=import_id,
                        filename=filename,
                    )
        except JSONImportError as error:
            message = _clean_text(error, 400) or "Import JSON invalide."
            status = (
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                if "depasse la limite" in message.casefold()
                or "profondeur maximale" in message.casefold()
                else HTTPStatus.BAD_REQUEST
            )
            self._error(status, message)
            return
        except (TypeError, ValueError) as error:
            self._error(HTTPStatus.BAD_REQUEST, _clean_text(error, 400) or "Import JSON invalide.")
            return
        except Exception:
            LOGGER.exception("Erreur pendant l'import JSON")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "L'import JSON a rencontre une erreur.")
            return
        self._send_json(HTTPStatus.OK, {"ok": True, "mode": mode, **result})

    def _chat(self, message: str) -> dict[str, Any]:
        observe_match = _OBSERVE_RE.match(message)
        if observe_match:
            fact = observe_match.group(1).strip()
            if not fact:
                raise ValueError("Il manque le souvenir à enregistrer.")
            with self.server.engine_lock:
                result = self.server.engine.observe(
                    fact,
                    episode_id=self.server.conversation_episode_id,
                    source="user_confirmed",
                )
            duplicate = bool(result.get("duplicate")) if isinstance(result, dict) else False
            event_id = result.get("event_id") if isinstance(result, dict) else None
            episode_id = result.get("episode_id") if isinstance(result, dict) else None
            if duplicate:
                reply = "Ce souvenir était déjà enregistré."
            else:
                reply = "C’est retenu dans ma mémoire associative."
            if event_id:
                reply += f" Son identifiant est {event_id}."
            return {
                "intent": "observe",
                "reply": reply,
                "data": result,
                "details": {
                    "intent": "observe",
                    "created": not duplicate,
                    "event_id": event_id,
                    "episode_id": episode_id,
                    "duplicate": duplicate,
                },
            }

        recall_match = _RECALL_RE.match(message)
        explain_requested = False
        if recall_match is None:
            recall_match = _KNOW_RE.match(message)
        if recall_match is None:
            recall_match = _REMIND_RE.match(message)
        if recall_match is None:
            recall_match = _EXPLAIN_RE.match(message)
            explain_requested = recall_match is not None
        if recall_match:
            query = recall_match.group(1).strip()
            with self.server.engine_lock:
                result = self.server.engine.recall(query, top_k=5)
            return {
                "intent": "recall",
                "reply": _explanation_reply(result) if explain_requested else _memory_reply(result),
                "data": result,
                "details": _compact_recall_details(result),
            }

        predict_match = _PREDICT_RE.match(message)
        if predict_match:
            history = predict_match.group(1).strip()
            if not history:
                raise ValueError("Il manque l’historique à compléter.")
            with self.server.engine_lock:
                result = self.server.engine.predict(history, top_k=5)
            return {
                "intent": "predict",
                "reply": _prediction_reply(result),
                "data": result,
                "details": _compact_prediction_details(result),
            }

        forget_match = _FORGET_RE.match(message)
        if forget_match:
            event_id = forget_match.group(1)
            with self.server.engine_lock:
                result = self.server.engine.forget(event_id)
            forgotten = bool(result.get("forgotten")) if isinstance(result, dict) else bool(result)
            reply = (
                f"L’événement {event_id} a été oublié et les associations ont été recalculées."
                if forgotten
                else f"Je n’ai trouvé aucun événement portant l’identifiant {event_id}."
            )
            return {
                "intent": "forget",
                "reply": reply,
                "data": result,
                "details": {
                    "intent": "forget",
                    "forgotten": forgotten,
                    "event_id": event_id,
                    "episode_id": result.get("episode_id") if isinstance(result, dict) else None,
                    "occurrences_removed": (
                        result.get("occurrences_removed") if isinstance(result, dict) else None
                    ),
                    "proofs_rebuilt_or_removed": (
                        result.get("proofs_rebuilt_or_removed")
                        if isinstance(result, dict)
                        else None
                    ),
                },
            }

        # Une demande libre est traitee comme un indice de rappel. Aucune sortie
        # du moteur ne sera reinjectee automatiquement comme observation.
        with self.server.engine_lock:
            result = self.server.engine.recall(message, top_k=5)
        return {
            "intent": "associative_recall",
            "reply": _memory_reply(result, generic=True),
            "data": result,
            "details": _compact_recall_details(result, intent="associative_recall"),
        }

    def do_OPTIONS(self) -> None:  # noqa: N802 - API de BaseHTTPRequestHandler
        # Pas de CORS : l'interface et l'API doivent rester sur la meme origine.
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "Les requêtes inter-origines ne sont pas permises.")

    def do_PUT(self) -> None:  # noqa: N802 - API de BaseHTTPRequestHandler
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "Méthode non permise.")

    def do_DELETE(self) -> None:  # noqa: N802 - API de BaseHTTPRequestHandler
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "Méthode non permise.")

    def do_PATCH(self) -> None:  # noqa: N802 - API de BaseHTTPRequestHandler
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "Méthode non permise.")


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("le port doit être un entier") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("le port doit être compris entre 1 et 65535")
    return port


def _loopback_host(value: str) -> str:
    host = value.strip().casefold()
    if host not in _LOOPBACK_HOSTS:
        allowed = ", ".join(sorted(_LOOPBACK_HOSTS))
        raise argparse.ArgumentTypeError(
            "ce prototype sans authentification accepte seulement " + allowed
        )
    return host


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interface locale du moteur de mémoire associative")
    parser.add_argument(
        "--host",
        type=_loopback_host,
        default="127.0.0.1",
        help="adresse locale: 127.0.0.1, localhost ou ::1 (défaut: 127.0.0.1)",
    )
    parser.add_argument("--port", type=_port, default=8765, help="port HTTP (défaut: 8765)")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"base SQLite (défaut: {DEFAULT_DB_PATH})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    db_path = args.db.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = MemoryEngine(db_path)
    server: MemoryHTTPServer | None = None
    try:
        server = MemoryHTTPServer((args.host, args.port), engine)
        host, port = server.server_address[:2]
        display_host = f"[{host}]" if ":" in str(host) else host
        LOGGER.info("Memoire disponible sur http://%s:%s", display_host, port)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        LOGGER.info("Arret demande")
    finally:
        if server is not None:
            server.server_close()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
