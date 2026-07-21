"""Petit serveur HTTP local pour dialoguer avec :class:`MemoryEngine`.

Le module n'utilise que la bibliotheque standard. Il peut etre lance avec::

    python -m memory_agent.server --port 8765 --db data/memory.sqlite3

L'interface n'est volontairement pas un agent generatif : elle classe des
souvenirs et des continuations deja observes par le moteur.
"""

from __future__ import annotations

import argparse
import hashlib
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

from memory_agent.json_import import (
    JSONImportError,
    commit_json_import,
    decode_json_import_content,
    prepare_json_import,
)
from memory_agent.math_engine import MathEngine, MathEngineError, MathLimits
from memory_agent.memory import MemoryEngine
from memory_agent.pipeline import (
    IdempotencyConflictError,
    MemoryPipeline,
    QueueStateError,
)

LOGGER = logging.getLogger("memory_agent.server")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_ROOT = PROJECT_ROOT / "web"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "memory.sqlite3"

MAX_BODY_BYTES = 3 * 1024 * 1024
MAX_CHAT_BODY_BYTES = 16 * 1024
MAX_MESSAGE_CHARS = 4_000
MAX_STATIC_BYTES = 2 * 1024 * 1024
MAX_MEMORIES = 100
MAX_PIPELINE_TEST_ITEMS = 250
MAX_CONVERSATION_EPISODE_EVENTS = 32
MAX_MATH_EXPRESSION_CHARS = MathLimits().max_expression_chars
MAX_MATH_CATALOG_ITEMS = 500


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
_CALCULATE_RE = re.compile(
    r"^\s*(?:calcule|calculer|combien\s+font)\s+(.+?)\s*[?!.]?\s*$",
    re.IGNORECASE | re.DOTALL,
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


def _math_error_payload(error: BaseException) -> dict[str, Any]:
    code = _clean_text(getattr(error, "code", "math_error"), 80) or "math_error"
    message = _clean_text(getattr(error, "message", None) or error, 500)
    return {
        "ok": False,
        "code": code,
        "error": message or "Expression mathematique invalide.",
    }


def _math_catalog_version(catalog: dict[str, Any]) -> str:
    value = catalog.get("version", catalog.get("catalog_version", "v1"))
    version = unicodedata.normalize("NFKC", str(value)).strip()
    if not version or len(version) > 100:
        raise ValueError("Version du catalogue mathematique invalide")
    return version


def _math_catalog_import_items(
    math_engine: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """Build deterministic memory facts from descriptions/rules, never results."""

    catalog = math_engine.catalog()
    if not isinstance(catalog, dict):
        raise ValueError("Le catalogue mathematique doit etre un objet")
    version = _math_catalog_version(catalog)
    entries = math_engine.catalog_entries()
    if not isinstance(entries, list):
        raise ValueError("Les entrees du catalogue doivent former une liste")
    if len(entries) > MAX_MATH_CATALOG_ITEMS:
        raise ValueError(
            f"Le catalogue depasse {MAX_MATH_CATALOG_ITEMS} algorithmes"
        )

    selected_fields = (
        "description",
        "category",
        "signature",
        "rule",
        "rules",
        "syntax",
        "formula",
        "domain",
        "aliases",
        "exact",
        "exactness",
        "learning_level",
        "maturity",
        "returns",
    )
    prepared: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Une entree du catalogue mathematique est invalide")
        raw_name = entry.get("name", entry.get("id"))
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Un algorithme du catalogue n'a pas de nom")
        display_name = unicodedata.normalize("NFKC", raw_name).strip()
        stable_name = display_name.casefold()
        if len(stable_name) > 200:
            raise ValueError("Un nom d'algorithme est trop long")
        if stable_name in seen_names:
            raise ValueError("Le catalogue contient deux fois le meme algorithme")
        seen_names.add(stable_name)

        details: list[str] = []
        for field in selected_fields:
            if field not in entry or entry[field] in (None, "", [], {}):
                continue
            value = entry[field]
            if isinstance(value, str):
                encoded = " ".join(value.split())
            else:
                encoded = json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=_json_default,
                )
            details.append(f"{field}: {encoded}")
        if not details:
            raise ValueError(
                f"L'algorithme {display_name!r} n'a ni description ni regle"
            )
        text = f"Algorithme mathematique {display_name}. " + ". ".join(details)
        if len(text) > 50_000:
            raise ValueError("Une description d'algorithme est trop longue")
        stable_key = f"math-catalog:{version}:{stable_name}"
        if len(stable_key) > 500:
            # Preserve the requested readable prefix while bounding DB keys.
            digest = hashlib.sha256(stable_name.encode("utf-8")).hexdigest()
            stable_key = f"math-catalog:{version}:{digest}"
        prepared.append(
            {
                "name": display_name,
                "stable_name": stable_name,
                "text": text,
                "idempotency_key": stable_key,
            }
        )
    return version, prepared


def _math_result_text(result: dict[str, Any]) -> str:
    value = result.get(
        "display",
        result.get("result", result.get("value", result.get("answer"))),
    )
    if value is None:
        return "Calcul terminé."
    return f"Résultat : {value}"


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


def _public_pipeline_job(job: dict[str, Any]) -> dict[str, Any]:
    """Expose l'etat d'un travail sans republier le souvenir qu'il contient."""

    result = job.get("result")
    public_result: dict[str, Any] | None = None
    if isinstance(result, dict):
        public_result = {
            key: result.get(key)
            for key in ("event_id", "episode_id", "created", "duplicate")
            if key in result
        }
    return {
        "job_id": job.get("job_id"),
        "sequence": job.get("sequence"),
        "state": job.get("state"),
        "attempts": job.get("attempts"),
        "max_attempts": job.get("max_attempts"),
        "enqueued_at": job.get("enqueued_at"),
        "updated_at": job.get("updated_at"),
        "completed_at": job.get("completed_at"),
        "last_error": _clean_text(job.get("last_error"), 300) or None,
        "duplicate_submission": bool(job.get("duplicate")),
        "retried": bool(job.get("retried")),
        "result": public_result,
    }


def _public_test_run(run: dict[str, Any]) -> dict[str, Any]:
    """Expose persistent lifecycle state without ticket payloads or keys."""

    return {
        "run_id": run.get("run_id"),
        "state": run.get("state"),
        "expected_count": run.get("expected_count"),
        "terminal_count": run.get("terminal_count"),
        "successful_count": run.get("successful_count"),
        "failed_count": run.get("failed_count"),
        "forgotten_count": run.get("forgotten_count"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "cleaned_at": run.get("cleaned_at"),
        "last_error": _clean_text(run.get("last_error"), 300) or None,
    }


def _public_pipeline_status(pipeline: MemoryPipeline) -> dict[str, Any]:
    queue = pipeline.queue.stats()
    worker = pipeline.worker.stats()
    test_runs = pipeline.queue.list_test_runs(limit=50)
    durable_cleanup_error = next(
        (
            run.get("last_error")
            for run in test_runs
            if run.get("state") == "cleanup_failed" and run.get("last_error")
        ),
        None,
    )
    received = int(queue.get("enqueue_requests", queue.get("total", 0)))
    deduplicated = int(queue.get("deduplicated_requests", 0))
    unique = int(queue.get("total", 0))
    memory_path = Path(pipeline.reader_engine.db_path).expanduser()
    if pipeline.reader_engine.db_path == ":memory:":
        size_bytes = None
    else:
        size_bytes = sum(
            candidate.stat().st_size if candidate.exists() else 0
            for candidate in (
                memory_path,
                Path(str(memory_path) + "-wal"),
                Path(str(memory_path) + "-shm"),
            )
        )
    return {
        "enabled": True,
        "mode": "injection_separee",
        "reader_writer_separated": (
            pipeline.reader_engine is not pipeline.writer_engine
        ),
        "pending": int(queue.get("pending", 0)),
        "processing": int(queue.get("processing", 0)),
        "completed": int(queue.get("completed", 0)),
        "failed": int(queue.get("failed", 0)),
        "unfinished": int(queue.get("unfinished", 0)),
        "total_unique_jobs": unique,
        "received_submissions": received,
        "deduplicated_submissions": deduplicated,
        "deduplication_percent": (
            round(100.0 * deduplicated / received, 2) if received else 0.0
        ),
        "lag_seconds": float(queue.get("lag_seconds", 0.0)),
        "worker_running": bool(worker.get("running")),
        "worker_error": _clean_text(worker.get("fatal_error"), 300) or None,
        "test_cleanup_error": _clean_text(
            worker.get("maintenance_error") or durable_cleanup_error, 300
        ) or None,
        "last_error": _clean_text(queue.get("last_error"), 300) or None,
        "test_runs": [
            _public_test_run(run) for run in test_runs
        ],
        "memory_size_bytes": size_bytes,
        "queue_size_bytes": queue.get("database_size_bytes"),
        "storage_size_bytes": (
            int(size_bytes or 0) + int(queue.get("database_size_bytes") or 0)
        ),
    }


class _PipelineImportAdapter:
    """Adapte l'import JSON existant vers la file durable."""

    def __init__(self, pipeline: MemoryPipeline):
        self.pipeline = pipeline
        self.jobs: list[dict[str, Any]] = []

    def observe(
        self,
        text: str,
        episode_id: str | None = None,
        context: dict[str, Any] | None = None,
        source: str | dict[str, Any] = "user_confirmed",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        job = self.pipeline.enqueue(
            text,
            episode_id=episode_id,
            context=context,
            source=source,
            idempotency_key=idempotency_key or f"json-fallback:{uuid4()}",
            payload_fingerprint=idempotency_key,
            retry_terminal=True,
        )
        self.jobs.append(job)
        return {
            "event_id": job["job_id"],
            "episode_id": episode_id,
            "created": not bool(job.get("duplicate")),
            "duplicate": bool(job.get("duplicate")),
        }


class MemoryHTTPServer(ThreadingHTTPServer):
    """Serveur de lecture, avec un pipeline d'ecriture facultatif."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        engine: MemoryEngine,
        web_root: Path = DEFAULT_WEB_ROOT,
        pipeline: MemoryPipeline | None = None,
        math_engine: Any | None = None,
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
        self.pipeline = pipeline
        self.math_engine = math_engine if math_engine is not None else MathEngine()
        self.engine_lock = threading.RLock()
        self.web_root = web_root.resolve()
        self.started_at = time.monotonic()
        # Une execution locale represente une conversation. Tous ses messages
        # confirmes appartiennent au meme episode et apprennent donc aussi les
        # transitions entre deux appels distincts a ``observe``.
        self.conversation_episode_id = str(uuid4())
        self.conversation_episode_events = 0
        self.conversation_lock = threading.Lock()

    def _reserve_conversation_episode_locked(self) -> str:
        if self.conversation_episode_events >= MAX_CONVERSATION_EPISODE_EVENTS:
            self.conversation_episode_id = str(uuid4())
            self.conversation_episode_events = 0
        self.conversation_episode_events += 1
        return self.conversation_episode_id

    def reserve_conversation_episode(self) -> str:
        """Keep temporal episodes useful while bounding rebuild cost."""

        with self.conversation_lock:
            return self._reserve_conversation_episode_locked()

    def enqueue_conversation_observation(
        self, fact: str, *, idempotency_key: str
    ) -> tuple[dict[str, Any], str]:
        """Atomically preserve a request's original temporal episode on retry."""

        if self.pipeline is None:
            raise RuntimeError("Le pipeline n'est pas active")
        with self.conversation_lock:
            existing = self.pipeline.queue.get_by_idempotency_key(idempotency_key)
            existing_episode = (
                existing.get("episode_id") if isinstance(existing, dict) else None
            )
            episode_id = (
                existing_episode
                if isinstance(existing_episode, str) and existing_episode
                else self._reserve_conversation_episode_locked()
            )
            job = self.pipeline.enqueue(
                fact,
                episode_id=episode_id,
                source="user_confirmed",
                idempotency_key=idempotency_key,
                retry_terminal=True,
            )
            return job, episode_id


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
                pipeline_status = (
                    _public_pipeline_status(self.server.pipeline)
                    if self.server.pipeline is not None
                    else {"enabled": False, "mode": "synchrone"}
                )
            except Exception:
                LOGGER.exception("La verification du moteur a echoue")
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "status": "unavailable"},
                    head_only=head_only,
                )
                return
            pipeline_degraded = bool(
                pipeline_status.get("enabled")
                and (
                    pipeline_status.get("worker_error")
                    or (
                        pipeline_status.get("unfinished", 0)
                        and not pipeline_status.get("worker_running")
                    )
                )
            )
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE if pipeline_degraded else HTTPStatus.OK,
                {
                    "ok": not pipeline_degraded,
                    "status": "degraded" if pipeline_degraded else "ready",
                    "read_available": True,
                    "uptime_seconds": round(time.monotonic() - self.server.started_at, 3),
                    "engine": "memory",
                    "stats_available": isinstance(stats, dict),
                    "pipeline": pipeline_status,
                },
                head_only=head_only,
            )
            return

        if path == "/api/pipeline":
            if self.server.pipeline is None:
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "pipeline": {"enabled": False, "mode": "synchrone"},
                    },
                    head_only=head_only,
                )
                return
            try:
                status = _public_pipeline_status(self.server.pipeline)
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "pipeline": status},
                    head_only=head_only,
                )
            except Exception:
                LOGGER.exception("Impossible de lire l'etat du pipeline")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "Etat du pipeline indisponible."},
                    head_only=head_only,
                )
            return

        if path == "/api/math/catalog":
            if self.server.math_engine is None:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "error": "Moteur mathematique indisponible."},
                    head_only=head_only,
                )
                return
            try:
                catalog = self.server.math_engine.catalog()
                entries = self.server.math_engine.catalog_entries()
                if not isinstance(catalog, dict) or not isinstance(entries, list):
                    raise ValueError("Catalogue mathematique invalide")
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "catalog": catalog, "count": len(entries)},
                    head_only=head_only,
                )
            except MathEngineError as error:
                self._send_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    _math_error_payload(error),
                    head_only=head_only,
                )
            except Exception:
                LOGGER.exception("Impossible de lire le catalogue mathematique")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "Catalogue mathematique indisponible."},
                    head_only=head_only,
                )
            return

        if path.startswith("/api/pipeline/test/runs/"):
            if self.server.pipeline is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "Pipeline non active."},
                    head_only=head_only,
                )
                return
            run_id = unquote(path.removeprefix("/api/pipeline/test/runs/"))
            if re.fullmatch(r"[a-f0-9]{12}", run_id) is None:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "run_id de test invalide."},
                    head_only=head_only,
                )
                return
            try:
                run = self.server.pipeline.queue.get_test_run(run_id)
            except Exception:
                LOGGER.exception("Impossible de lire le run de test")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "Etat du test indisponible."},
                    head_only=head_only,
                )
                return
            if run is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "Run de test inconnu."},
                    head_only=head_only,
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "test_run": _public_test_run(run)},
                head_only=head_only,
            )
            return

        if path.startswith("/api/pipeline/jobs/"):
            if self.server.pipeline is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "Pipeline non active."},
                    head_only=head_only,
                )
                return
            job_id = unquote(path.removeprefix("/api/pipeline/jobs/"))
            try:
                UUID(job_id)
            except (ValueError, AttributeError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "Identifiant de travail invalide."},
                    head_only=head_only,
                )
                return
            try:
                job = self.server.pipeline.queue.get(job_id)
            except Exception:
                LOGGER.exception("Impossible de lire un travail du pipeline")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "Travail indisponible."},
                    head_only=head_only,
                )
                return
            if job is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "Travail inconnu."},
                    head_only=head_only,
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "job": _public_pipeline_job(job)},
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
        if path not in {
            "/api/chat",
            "/api/calculate",
            "/api/import",
            "/api/math/catalog/import",
            "/api/pipeline/test",
            "/api/pipeline/test/cleanup",
            "/api/pipeline/jobs/status",
        }:
            if path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "Route API inconnue.")
            else:
                self._error(HTTPStatus.METHOD_NOT_ALLOWED, "Méthode non permise.")
            return
        if not self._same_origin_or_non_browser():
            self._error(HTTPStatus.FORBIDDEN, "Origine non autorisée.")
            return
        payload = self._read_json_object(
            max_bytes=(
                MAX_CHAT_BODY_BYTES
                if path in {"/api/chat", "/api/calculate"}
                else MAX_BODY_BYTES
            )
        )
        if payload is None:
            return
        if path == "/api/calculate":
            self._handle_calculate(payload)
            return
        if path == "/api/math/catalog/import":
            self._handle_math_catalog_import(payload)
            return
        if path == "/api/pipeline/jobs/status":
            self._handle_pipeline_job_status(payload)
            return
        if path == "/api/pipeline/test/cleanup":
            self._handle_pipeline_test_cleanup(payload)
            return
        if path == "/api/pipeline/test":
            self._handle_pipeline_test(payload)
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
        request_id = payload.get("request_id")
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id.strip():
                self._error(HTTPStatus.BAD_REQUEST, "request_id doit etre une chaine non vide.")
                return
            request_id = request_id.strip()
            if len(request_id) > 128:
                self._error(HTTPStatus.BAD_REQUEST, "request_id est trop long.")
                return

        try:
            response = self._chat(message, request_id=request_id)
        except IdempotencyConflictError as error:
            self._error(HTTPStatus.CONFLICT, _clean_text(error, 300))
            return
        except MathEngineError as error:
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY, _math_error_payload(error)
            )
            return
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, _clean_text(error, 300) or "Demande invalide.")
            return
        except Exception:
            LOGGER.exception("Erreur du moteur pendant une conversation")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Le moteur de mémoire a rencontré une erreur.")
            return
        status = response.pop("_http_status", HTTPStatus.OK)
        self._send_json(status, {"ok": True, **response})

    def _handle_calculate(self, payload: dict[str, Any]) -> None:
        expression = payload.get("expression")
        if not isinstance(expression, str):
            self._error(HTTPStatus.BAD_REQUEST, "expression doit etre une chaine.")
            return
        expression = unicodedata.normalize("NFC", expression).strip()
        if not expression:
            self._error(HTTPStatus.BAD_REQUEST, "expression ne peut pas etre vide.")
            return
        if len(expression) > MAX_MATH_EXPRESSION_CHARS:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"expression depasse {MAX_MATH_EXPRESSION_CHARS} caracteres.",
            )
            return
        if self.server.math_engine is None:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Moteur mathematique indisponible.",
            )
            return
        try:
            result = self.server.math_engine.evaluate(expression)
            if not isinstance(result, dict):
                raise TypeError("Le moteur mathematique doit retourner un objet")
            _json_bytes(result)
        except MathEngineError as error:
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY, _math_error_payload(error)
            )
            return
        except Exception:
            LOGGER.exception("Erreur interne du moteur mathematique")
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Le calcul mathematique a rencontre une erreur interne.",
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "intent": "calculate",
                "expression": expression,
                "result": result,
            },
        )

    def _handle_math_catalog_import(self, payload: dict[str, Any]) -> None:
        if payload:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "L'import du catalogue n'accepte aucun parametre.",
            )
            return
        pipeline = self.server.pipeline
        if pipeline is None:
            self._error(
                HTTPStatus.CONFLICT,
                "Le pipeline d'apprentissage doit etre active pour cet import.",
            )
            return
        if self.server.math_engine is None:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Moteur mathematique indisponible.",
            )
            return

        jobs: list[dict[str, Any]] = []
        try:
            version, entries = _math_catalog_import_items(self.server.math_engine)
            for entry in entries:
                job = pipeline.enqueue(
                    entry["text"],
                    episode_id=(
                        f"math-catalog:{version}:{entry['stable_name']}"
                    ),
                    context={
                        "category": "math_algorithm",
                        "algorithm": entry["name"],
                        "catalog_version": version,
                    },
                    source={
                        "type": "executed",
                        "origin": "math_engine_catalog",
                        "catalog_version": version,
                        "algorithm": entry["name"],
                        "external": True,
                        "deterministic": True,
                    },
                    idempotency_key=entry["idempotency_key"],
                    retry_terminal=True,
                )
                jobs.append(job)
        except IdempotencyConflictError as error:
            self._send_json(
                HTTPStatus.CONFLICT,
                {
                    "ok": False,
                    "error": _clean_text(error, 400),
                    "queued_count": sum(
                        int(bool(job.get("retried")) or not bool(job.get("duplicate")))
                        for job in jobs
                    ),
                    "job_ids": [job["job_id"] for job in jobs],
                },
            )
            return
        except MathEngineError as error:
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY, _math_error_payload(error)
            )
            return
        except (TypeError, ValueError) as error:
            self._error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                _clean_text(error, 500) or "Catalogue mathematique invalide.",
            )
            return
        except Exception:
            LOGGER.exception("Erreur pendant l'import du catalogue mathematique")
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "L'import du catalogue mathematique a rencontre une erreur.",
            )
            return

        queued_count = sum(
            int(bool(job.get("retried")) or not bool(job.get("duplicate")))
            for job in jobs
        )
        duplicate_count = sum(
            int(bool(job.get("duplicate")) and not bool(job.get("retried")))
            for job in jobs
        )
        self._send_json(
            HTTPStatus.ACCEPTED,
            {
                "ok": True,
                "queued": True,
                "catalog_version": version,
                "total_count": len(jobs),
                "queued_count": queued_count,
                "duplicate_count": duplicate_count,
                "deduplicated_count": duplicate_count,
                "job_ids": [job["job_id"] for job in jobs],
                "consistency": "visible_apres_consolidation",
            },
        )

    def _handle_pipeline_job_status(self, payload: dict[str, Any]) -> None:
        pipeline = self.server.pipeline
        if pipeline is None:
            self._error(HTTPStatus.CONFLICT, "Le pipeline d'apprentissage n'est pas active.")
            return
        job_ids = payload.get("job_ids")
        if not isinstance(job_ids, list) or not 1 <= len(job_ids) <= 250:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "job_ids doit contenir entre 1 et 250 identifiants.",
            )
            return
        clean_ids: list[str] = []
        for value in job_ids:
            if not isinstance(value, str):
                self._error(HTTPStatus.BAD_REQUEST, "Chaque job_id doit etre une chaine.")
                return
            candidate = value.strip()
            try:
                UUID(candidate)
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "Un job_id est invalide.")
                return
            if candidate not in clean_ids:
                clean_ids.append(candidate)
        try:
            jobs = pipeline.queue.get_many(clean_ids)
        except Exception:
            LOGGER.exception("Impossible de lire les travaux du pipeline")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Travaux indisponibles.")
            return
        self._send_json(
            HTTPStatus.OK,
            {"ok": True, "jobs": [_public_pipeline_job(job) for job in jobs]},
        )

    def _handle_pipeline_test(self, payload: dict[str, Any]) -> None:
        pipeline = self.server.pipeline
        if pipeline is None:
            self._error(HTTPStatus.CONFLICT, "Le pipeline d'apprentissage n'est pas active.")
            return
        count = payload.get("count", 25)
        if isinstance(count, bool) or not isinstance(count, int):
            self._error(HTTPStatus.BAD_REQUEST, "count doit etre un entier.")
            return
        if not 1 <= count <= MAX_PIPELINE_TEST_ITEMS:
            self._error(
                HTTPStatus.BAD_REQUEST,
                f"count doit etre compris entre 1 et {MAX_PIPELINE_TEST_ITEMS}.",
            )
            return

        run_id = uuid4().hex[:12]
        try:
            # Le registre et tous les tickets sont commits ensemble. Il est
            # impossible de perdre run_id/job_ids apres une insertion partielle.
            test_run = pipeline.enqueue_test_run(run_id, count=count)
            jobs = test_run["jobs"]
        except Exception:
            LOGGER.exception("Impossible de mettre le test en file")
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Le test n'a pas pu etre place dans la file.",
            )
            return
        self._send_json(
            HTTPStatus.ACCEPTED,
            {
                "ok": True,
                "queued": True,
                "run_id": run_id,
                "queued_count": len(jobs),
                "job_ids": [job["job_id"] for job in jobs],
            },
        )

    def _handle_pipeline_test_cleanup(self, payload: dict[str, Any]) -> None:
        pipeline = self.server.pipeline
        if pipeline is None:
            self._error(HTTPStatus.CONFLICT, "Le pipeline d'apprentissage n'est pas active.")
            return
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or re.fullmatch(r"[a-f0-9]{12}", run_id) is None:
            self._error(HTTPStatus.BAD_REQUEST, "run_id de test invalide.")
            return
        job_ids = payload.get("job_ids")
        if job_ids is not None and (
            not isinstance(job_ids, list)
            or not 1 <= len(job_ids) <= MAX_PIPELINE_TEST_ITEMS
            or any(not isinstance(value, str) for value in job_ids)
        ):
            self._error(HTTPStatus.BAD_REQUEST, "job_ids de test invalides.")
            return
        try:
            registered = pipeline.queue.get_test_run(run_id)
            if registered is None:
                self._error(HTTPStatus.NOT_FOUND, "Run de test inconnu.")
                return
            if job_ids is not None and set(job_ids) != set(registered["job_ids"]):
                self._error(HTTPStatus.FORBIDDEN, "Ces travaux n'appartiennent pas au test.")
                return
            cleaned = pipeline.cleanup_test_run(run_id)
        except QueueStateError as error:
            self._error(HTTPStatus.CONFLICT, _clean_text(error, 300))
            return
        except Exception:
            LOGGER.exception("Impossible de nettoyer les souvenirs de test")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Nettoyage du test incomplet.")
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "run_id": run_id,
                "forgotten": cleaned["forgotten_count"],
                "successful_count": cleaned["successful_count"],
                "failed_count": cleaned["failed_count"],
                "state": cleaned["state"],
            },
        )

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
        response_status: HTTPStatus = HTTPStatus.OK
        try:
            data = (
                decode_json_import_content(payload["content"])
                if has_content
                else payload["data"]
            )
            if mode == "preview":
                with self.server.engine_lock:
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
                if self.server.pipeline is None:
                    with self.server.engine_lock:
                        result = self.server.engine.import_json(
                            data,
                            import_id=import_id,
                            filename=filename,
                        )
                else:
                    # La validation et la categorisation restent synchrones;
                    # seules les ecritures de memoire partent en arriere-plan.
                    plan = prepare_json_import(data, filename=filename)
                    adapter = _PipelineImportAdapter(self.server.pipeline)
                    result = commit_json_import(
                        adapter,
                        plan,
                        import_id=import_id,
                    )
                    result.update(
                        {
                            "queued": True,
                            "queued_count": sum(
                                int(
                                    bool(job.get("retried"))
                                    or not bool(job.get("duplicate"))
                                )
                                for job in adapter.jobs
                            ),
                            "job_ids": [job["job_id"] for job in adapter.jobs],
                            "consistency": "visible_apres_consolidation",
                        }
                    )
                    response_status = HTTPStatus.ACCEPTED
        except IdempotencyConflictError as error:
            self._error(HTTPStatus.CONFLICT, _clean_text(error, 400))
            return
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
        self._send_json(response_status, {"ok": True, "mode": mode, **result})

    def _chat(
        self, message: str, *, request_id: str | None = None
    ) -> dict[str, Any]:
        calculate_match = _CALCULATE_RE.match(message)
        if calculate_match:
            expression = calculate_match.group(1).strip()
            if not expression:
                raise ValueError("Il manque l'expression a calculer.")
            if len(expression) > MAX_MATH_EXPRESSION_CHARS:
                raise ValueError(
                    f"L'expression depasse {MAX_MATH_EXPRESSION_CHARS} caracteres."
                )
            if self.server.math_engine is None:
                raise RuntimeError("Moteur mathematique indisponible")
            result = self.server.math_engine.evaluate(expression)
            if not isinstance(result, dict):
                raise TypeError("Le moteur mathematique doit retourner un objet")
            _json_bytes(result)
            return {
                "intent": "calculate",
                "reply": _math_result_text(result),
                "data": result,
                "details": {
                    "intent": "calculate",
                    "expression": expression,
                    "algorithm": result.get("algorithm"),
                    "algorithm_version": result.get("algorithm_version"),
                },
            }

        observe_match = _OBSERVE_RE.match(message)
        if observe_match:
            fact = observe_match.group(1).strip()
            if not fact:
                raise ValueError("Il manque le souvenir à enregistrer.")
            if self.server.pipeline is not None:
                source_key = f"chat:{request_id or uuid4()}"
                job, episode_id = self.server.enqueue_conversation_observation(
                    fact, idempotency_key=source_key
                )
                public_job = _public_pipeline_job(job)
                return {
                    "_http_status": HTTPStatus.ACCEPTED,
                    "intent": "observe",
                    "reply": (
                        "Souvenir reçu et placé dans la file d’apprentissage. "
                        "Il deviendra interrogeable dès sa consolidation."
                    ),
                    "data": {
                        **public_job,
                        "queued": True,
                        "episode_id": episode_id,
                    },
                    "details": {
                        "intent": "observe",
                        "queued": True,
                        "job_id": job["job_id"],
                        "state": job["state"],
                        "episode_id": episode_id,
                    },
                }
            episode_id = self.server.reserve_conversation_episode()
            with self.server.engine_lock:
                result = self.server.engine.observe(
                    fact,
                    episode_id=episode_id,
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
            if self.server.pipeline is not None:
                # Pipeline deletions use the synchronous=FULL writer.  The
                # NORMAL reader must never acknowledge a less durable delete.
                result = self.server.pipeline.writer_engine.forget(event_id)
            else:
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


def _bounded_integer(value: str, *, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} doit etre un entier") from error
    if not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(
            f"{name} doit etre compris entre {minimum} et {maximum}"
        )
    return parsed


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
    parser.add_argument(
        "--async-injection",
        action="store_true",
        help="separe l'injection, la consolidation et la lecture",
    )
    parser.add_argument(
        "--queue-db",
        type=Path,
        default=None,
        help="journal SQLite des injections (defaut: injection.sqlite3 pres de --db)",
    )
    parser.add_argument(
        "--pipeline-batch-size",
        type=lambda value: _bounded_integer(
            value, name="pipeline-batch-size", minimum=1, maximum=1_000
        ),
        default=16,
        help="nombre maximal de souvenirs consolides par lot (defaut: 16)",
    )
    parser.add_argument(
        "--pipeline-poll-ms",
        type=lambda value: _bounded_integer(
            value, name="pipeline-poll-ms", minimum=10, maximum=60_000
        ),
        default=100,
        help="attente du consolidateur en millisecondes (defaut: 100)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    db_path = args.db.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine: MemoryEngine | None = None
    pipeline: MemoryPipeline | None = None
    server: MemoryHTTPServer | None = None
    try:
        if args.async_injection:
            queue_path = (
                args.queue_db.expanduser().resolve()
                if args.queue_db is not None
                else db_path.with_name("injection.sqlite3")
            )
            if queue_path == db_path:
                raise ValueError("La file d'injection doit etre distincte de la memoire.")
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            pipeline = MemoryPipeline(
                db_path,
                queue_path,
                batch_size=args.pipeline_batch_size,
                poll_interval=args.pipeline_poll_ms / 1_000.0,
            )
            engine = pipeline.reader_engine
        else:
            engine = MemoryEngine(db_path)
        server = MemoryHTTPServer(
            (args.host, args.port),
            engine,
            pipeline=pipeline,
        )
        host, port = server.server_address[:2]
        display_host = f"[{host}]" if ":" in str(host) else host
        mode = "pipeline separe" if pipeline is not None else "ecriture synchrone"
        LOGGER.info("Memoire disponible sur http://%s:%s (%s)", display_host, port, mode)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        LOGGER.info("Arret demande")
    finally:
        if server is not None:
            server.server_close()
        if pipeline is not None:
            pipeline.close()
        elif engine is not None:
            engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
