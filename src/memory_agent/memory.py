"""Coeur SQLite d'une memoire episodique et associative explicable.

Le journal d'evenements et ses occurrences sont la source de verite. Les
motifs et continuations sont des vues consolidees, integralement
reconstruisibles a partir de ces occurrences. Cela rend ``forget`` fiable :
une suppression retire aussi toutes les preuves derivees de l'evenement.

Le prototype n'essaie pas de "comprendre" librement une phrase. Il apprend
des motifs de tokens normalises d'ordre 1 a 3 et retrouve les episodes dans
lesquels les mots d'une question ont ete observes.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import unicodedata
from typing import Any, Iterable, Iterator, Mapping, Sequence
from uuid import uuid4


_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)
_TRUSTED_SOURCES = frozenset({"observed", "executed", "user_confirmed"})
_KNOWN_SOURCES = _TRUSTED_SOURCES | frozenset({"inferred", "generated"})
_MAX_PATTERN_ORDER = 3
_MAX_TOP_K = 100
_MAX_QUERY_TOKENS = 64
_MAX_QUERY_CHARACTERS = 50_000
_MAX_RECALL_EVENTS = 12
_MAX_RECALL_MATCH_EVENTS = 6
_RECALL_EVENT_NEIGHBOURHOOD = 1
_MAX_RECALL_PATH_ITEMS = 100
_MAX_RECALL_EVIDENCE = 25
_MAX_RETURNED_EVENT_TEXT = 4_000
_SCHEMA_VERSION = 1


def _utcnow() -> str:
    """Return a sortable UTC timestamp with microsecond precision."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalise_token(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).strip().casefold())
    folded: list[str] = []
    latin_base = False
    for character in decomposed:
        if unicodedata.combining(character):
            if latin_base:
                continue
        else:
            latin_base = "LATIN" in unicodedata.name(character, "")
        folded.append(character)
    text = unicodedata.normalize("NFKC", "".join(folded))
    return text.replace("\u2019", "'")


def _tokenise(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    return [
        token
        for token in (_normalise_token(item) for item in _TOKEN_RE.findall(text))
        if token
    ]


def _json_ready(value: Any) -> Any:
    """Convert common values to deterministic, JSON-compatible structures."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_ready(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _canonical_context(context: Mapping[str, Any] | None) -> tuple[dict[str, Any], str, str]:
    if context is None:
        clean: dict[str, Any] = {}
    elif not isinstance(context, Mapping):
        raise TypeError("context doit etre un dictionnaire ou None")
    else:
        clean = dict(_json_ready(context))
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    signature = "global" if not clean else hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return clean, encoded, signature


def _bounded_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k doit etre un entier")
    if not 1 <= top_k <= _MAX_TOP_K:
        raise ValueError(f"top_k doit etre compris entre 1 et {_MAX_TOP_K}")
    return top_k


class MemoryEngine:
    """Persistent and explainable local memory.

    Parameters
    ----------
    db_path:
        SQLite file. Use ``":memory:"`` for an ephemeral engine.

    Notes
    -----
    ``observe`` accepts an optional ``idempotency_key`` in addition to the
    minimal public contract. For integrations that cannot pass that keyword,
    the same key may be placed in ``context["idempotency_key"]`` or supplied
    in a source mapping.
    """

    def __init__(self, db_path: str | Path):
        if db_path is None:
            raise TypeError("db_path est obligatoire")

        raw_path = str(db_path)
        if not raw_path.strip():
            raise ValueError("db_path ne peut pas etre vide")
        self.db_path = raw_path
        if raw_path != ":memory:":
            Path(raw_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            raw_path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            if raw_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.execute("PRAGMA synchronous = NORMAL")
            self._create_schema()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    def _create_schema(self) -> None:
        """Create a new v1 schema or validate an existing v1 database.

        A database carrying another schema version is never modified. This is
        intentional: migrations must be explicit so an older executable cannot
        silently reinterpret or downgrade newer memory data.
        """

        existing_tables = {
            row["name"]
            for row in self._connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        }
        new_database = not existing_tables
        if not new_database:
            if "metadata" not in existing_tables:
                raise RuntimeError(
                    "Base SQLite existante sans schema_version; migration explicite requise"
                )
            version_row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if version_row is None:
                raise RuntimeError(
                    "Base SQLite existante sans schema_version; migration explicite requise"
                )
            try:
                existing_version = int(version_row["value"])
            except (TypeError, ValueError) as error:
                raise RuntimeError("schema_version SQLite invalide") from error
            if existing_version != _SCHEMA_VERSION:
                raise RuntimeError(
                    "Version SQLite incompatible: "
                    f"moteur={_SCHEMA_VERSION}, base={existing_version}"
                )

        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                context_json TEXT NOT NULL DEFAULT '{}',
                context_signature TEXT NOT NULL DEFAULT 'global',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                ingest_order INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_json TEXT NOT NULL,
                idempotency_key TEXT UNIQUE,
                context_json TEXT NOT NULL,
                context_signature TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_key TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
                concept_id INTEGER NOT NULL REFERENCES concepts(id),
                ordinal INTEGER NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(event_id, ordinal)
            );

            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_hash TEXT NOT NULL,
                sequence_json TEXT NOT NULL,
                length INTEGER NOT NULL CHECK(length BETWEEN 1 AND 3),
                context_signature TEXT NOT NULL,
                context_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(sequence_json, context_signature)
            );

            CREATE TABLE IF NOT EXISTS pattern_items (
                pattern_id INTEGER NOT NULL REFERENCES patterns(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                concept_id INTEGER NOT NULL REFERENCES concepts(id),
                PRIMARY KEY(pattern_id, ordinal)
            );

            CREATE TABLE IF NOT EXISTS continuations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL REFERENCES patterns(id) ON DELETE CASCADE,
                to_concept_id INTEGER NOT NULL REFERENCES concepts(id),
                support_count INTEGER NOT NULL DEFAULT 0 CHECK(support_count >= 0),
                episode_support_count INTEGER NOT NULL DEFAULT 0 CHECK(episode_support_count >= 0),
                decayed_support REAL NOT NULL DEFAULT 0,
                first_seen TEXT,
                last_seen TEXT,
                UNIQUE(pattern_id, to_concept_id)
            );

            CREATE TABLE IF NOT EXISTS evidence_spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                continuation_id INTEGER NOT NULL REFERENCES continuations(id) ON DELETE CASCADE,
                episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
                event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                start_occurrence_id INTEGER NOT NULL REFERENCES occurrences(id) ON DELETE CASCADE,
                end_occurrence_id INTEGER NOT NULL REFERENCES occurrences(id) ON DELETE CASCADE,
                next_occurrence_id INTEGER NOT NULL REFERENCES occurrences(id) ON DELETE CASCADE,
                source_type TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                algorithm_version TEXT NOT NULL DEFAULT 'patterns-v1',
                UNIQUE(continuation_id, episode_id, start_occurrence_id, next_occurrence_id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_episode
                ON events(episode_id, ingest_order);
            CREATE INDEX IF NOT EXISTS idx_events_context
                ON events(context_signature);
            CREATE INDEX IF NOT EXISTS idx_occurrences_concept_episode
                ON occurrences(concept_id, episode_id);
            CREATE INDEX IF NOT EXISTS idx_occurrences_episode
                ON occurrences(episode_id, id);
            CREATE INDEX IF NOT EXISTS idx_patterns_hash_context
                ON patterns(sequence_hash, context_signature);
            CREATE INDEX IF NOT EXISTS idx_continuations_pattern
                ON continuations(pattern_id, support_count DESC);
            CREATE INDEX IF NOT EXISTS idx_evidence_continuation
                ON evidence_spans(continuation_id, observed_at DESC);
            CREATE INDEX IF NOT EXISTS idx_evidence_episode
                ON evidence_spans(episode_id);
            """
        )
        if new_database:
            self._connection.execute(
                "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Le moteur de memoire est ferme")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Run an atomic write, including rollback after a failed commit."""

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            if self._connection.in_transaction:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise
        else:
            try:
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    try:
                        self._connection.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                raise

    @contextmanager
    def _read_snapshot(self) -> Iterator[None]:
        """Keep all SELECT statements of a public read on one SQLite snapshot."""

        self._connection.execute("BEGIN")
        try:
            yield
        except BaseException:
            if self._connection.in_transaction:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise
        else:
            try:
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    try:
                        self._connection.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                raise

    @staticmethod
    def _parse_source(source: str | Mapping[str, Any]) -> tuple[str, dict[str, Any], str | None]:
        if isinstance(source, str):
            source_type = source.strip()
            source_data: dict[str, Any] = {"type": source_type}
            source_idempotency = None
        elif isinstance(source, Mapping):
            source_data = dict(_json_ready(source))
            source_type = str(source_data.get("type", "user_confirmed")).strip()
            source_data["type"] = source_type
            source_idempotency = source_data.pop("idempotency_key", None)
        else:
            raise TypeError("source doit etre une chaine ou un dictionnaire")

        if source_type not in _KNOWN_SOURCES:
            allowed = ", ".join(sorted(_KNOWN_SOURCES))
            raise ValueError(f"source inconnue: {source_type!r}; valeurs permises: {allowed}")
        if source_idempotency is not None:
            source_idempotency = str(source_idempotency).strip() or None
        return source_type, source_data, source_idempotency

    @staticmethod
    def _normalise_history(history: Any) -> tuple[list[str], Mapping[str, Any] | None]:
        context: Mapping[str, Any] | None = None
        value = history
        if isinstance(history, Mapping):
            value = history.get("history", history.get("text", history.get("items", [])))
            raw_context = history.get("context")
            if raw_context is not None and not isinstance(raw_context, Mapping):
                raise TypeError("history['context'] doit etre un dictionnaire")
            context = raw_context

        if isinstance(value, str):
            tokens = _tokenise(value)
        elif isinstance(value, Iterable):
            tokens = []
            for item in value:
                if isinstance(item, str):
                    parsed = _tokenise(item)
                    tokens.extend(parsed if parsed else [_normalise_token(item)])
                else:
                    token = _normalise_token(item)
                    if token:
                        tokens.append(token)
        else:
            raise TypeError("history doit etre une chaine, une sequence ou un dictionnaire")
        return tokens, context

    def _event_result(self, event_id: str, *, duplicate: bool) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT e.*, COUNT(o.id) AS token_count
            FROM events e
            LEFT JOIN occurrences o ON o.event_id = e.id
            WHERE e.id = ?
            GROUP BY e.id
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Evenement introuvable apres ingestion")
        token_rows = self._connection.execute(
            "SELECT token FROM occurrences WHERE event_id = ? ORDER BY ordinal",
            (event_id,),
        ).fetchall()
        return {
            "event_id": row["id"],
            "episode_id": row["episode_id"],
            "created": not duplicate,
            "duplicate": duplicate,
            "text": row["text"],
            "tokens": [item["token"] for item in token_rows],
            "source": row["source_type"],
            "context": json.loads(row["context_json"]),
            "idempotency_key": row["idempotency_key"],
            "created_at": row["created_at"],
        }

    def observe(
        self,
        text: str,
        episode_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        source: str | Mapping[str, Any] = "user_confirmed",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Record one textual event and learn its token continuations.

        Reusing a non-empty idempotency key returns the original event without
        increasing any support counter. If an ``episode_id`` is reused, tokens
        from successive trusted observations form one continuous episode.
        ``generated`` and ``inferred`` events remain recallable but never
        reinforce factual continuations automatically.
        """

        self._ensure_open()
        if not isinstance(text, str):
            raise TypeError("text doit etre une chaine")
        original_text = text.strip()
        if not original_text:
            raise ValueError("text ne peut pas etre vide")
        if len(original_text) > 1_000_000:
            raise ValueError("text depasse la limite du prototype (1 000 000 caracteres)")
        tokens = _tokenise(original_text)
        if not tokens:
            raise ValueError("text ne contient aucun concept exploitable")

        source_type, source_data, source_key = self._parse_source(source)
        raw_context = dict(context or {})
        context_key = raw_context.pop("_idempotency_key", None)
        if context_key is None:
            context_key = raw_context.pop("idempotency_key", None)
        supplied_key = idempotency_key if idempotency_key is not None else source_key
        if supplied_key is None:
            supplied_key = context_key
        if supplied_key is not None:
            supplied_key = str(supplied_key).strip()
            if not supplied_key:
                supplied_key = None
            elif len(supplied_key) > 500:
                raise ValueError("idempotency_key est trop longue")

        clean_context, context_json, context_signature = _canonical_context(raw_context)
        explicit_episode = None if episode_id is None else str(episode_id).strip()
        if episode_id is not None and not explicit_episode:
            raise ValueError("episode_id ne peut pas etre vide")
        if explicit_episode and len(explicit_episode) > 500:
            raise ValueError("episode_id est trop long")

        now = _utcnow()
        event_id = str(uuid4())
        selected_episode = explicit_episode or str(uuid4())
        source_json = json.dumps(source_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        with self._lock:
            self._ensure_open()
            with self._transaction():
                if supplied_key:
                    existing = self._connection.execute(
                        "SELECT id FROM events WHERE idempotency_key = ?",
                        (supplied_key,),
                    ).fetchone()
                    if existing is not None:
                        return self._event_result(existing["id"], duplicate=True)

                episode = self._connection.execute(
                    "SELECT id, context_json, context_signature FROM episodes WHERE id = ?",
                    (selected_episode,),
                ).fetchone()
                if episode is None:
                    self._connection.execute(
                        """
                        INSERT INTO episodes(
                            id, context_json, context_signature, created_at, updated_at
                        ) VALUES(?, ?, ?, ?, ?)
                        """,
                        (selected_episode, context_json, context_signature, now, now),
                    )
                else:
                    # An omitted context inherits the episode's context. An
                    # explicit one remains attached to this event.
                    if context is None:
                        context_json = episode["context_json"]
                        context_signature = episode["context_signature"]
                        clean_context = json.loads(context_json)
                    self._connection.execute(
                        "UPDATE episodes SET updated_at = ? WHERE id = ?",
                        (now, selected_episode),
                    )

                self._connection.execute(
                    """
                    INSERT INTO events(
                        id, episode_id, text, normalized_text, source_type,
                        source_json, idempotency_key, context_json,
                        context_signature, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        selected_episode,
                        original_text,
                        " ".join(tokens),
                        source_type,
                        source_json,
                        supplied_key,
                        context_json,
                        context_signature,
                        now,
                    ),
                )

                concept_ids: list[int] = []
                for token in tokens:
                    self._connection.execute(
                        """
                        INSERT INTO concepts(canonical_key, label, created_at)
                        VALUES(?, ?, ?)
                        ON CONFLICT(canonical_key) DO NOTHING
                        """,
                        (token, token, now),
                    )
                    concept_row = self._connection.execute(
                        "SELECT id FROM concepts WHERE canonical_key = ?",
                        (token,),
                    ).fetchone()
                    assert concept_row is not None
                    concept_id = int(concept_row["id"])
                    concept_ids.append(concept_id)

                for ordinal, (token, concept_id) in enumerate(zip(tokens, concept_ids)):
                    self._connection.execute(
                        """
                        INSERT INTO occurrences(
                            event_id, episode_id, concept_id, ordinal, token, created_at
                        ) VALUES(?, ?, ?, ?, ?, ?)
                        """,
                        (event_id, selected_episode, concept_id, ordinal, token, now),
                    )

                self._rebuild_episode_evidence(selected_episode)
                self._refresh_aggregates()
                result = self._event_result(event_id, duplicate=False)

        # Keep the explicit local variable meaningful in debuggers and make
        # it clear that the canonical context is the one returned from SQL.
        del clean_context
        return result

    def _get_or_create_pattern(
        self,
        concept_ids: Sequence[int],
        context_json: str,
        context_signature: str,
        created_at: str,
    ) -> int:
        sequence_json = json.dumps(list(concept_ids), separators=(",", ":"))
        sequence_hash = hashlib.sha256(sequence_json.encode("ascii")).hexdigest()
        self._connection.execute(
            """
            INSERT INTO patterns(
                sequence_hash, sequence_json, length, context_signature,
                context_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(sequence_json, context_signature) DO NOTHING
            """,
            (
                sequence_hash,
                sequence_json,
                len(concept_ids),
                context_signature,
                context_json,
                created_at,
            ),
        )
        row = self._connection.execute(
            """
            SELECT id FROM patterns
            WHERE sequence_json = ? AND context_signature = ?
            """,
            (sequence_json, context_signature),
        ).fetchone()
        assert row is not None
        pattern_id = int(row["id"])
        self._connection.executemany(
            """
            INSERT INTO pattern_items(pattern_id, ordinal, concept_id)
            VALUES(?, ?, ?)
            ON CONFLICT(pattern_id, ordinal) DO NOTHING
            """,
            [
                (pattern_id, ordinal, concept_id)
                for ordinal, concept_id in enumerate(concept_ids)
            ],
        )
        return pattern_id

    def _get_or_create_continuation(self, pattern_id: int, concept_id: int) -> int:
        self._connection.execute(
            """
            INSERT INTO continuations(pattern_id, to_concept_id)
            VALUES(?, ?)
            ON CONFLICT(pattern_id, to_concept_id) DO NOTHING
            """,
            (pattern_id, concept_id),
        )
        row = self._connection.execute(
            """
            SELECT id FROM continuations
            WHERE pattern_id = ? AND to_concept_id = ?
            """,
            (pattern_id, concept_id),
        ).fetchone()
        assert row is not None
        return int(row["id"])

    def _rebuild_episode_evidence(self, episode_id: str) -> None:
        """Recreate all evidence for one episode from its occurrences."""

        self._connection.execute(
            "DELETE FROM evidence_spans WHERE episode_id = ?",
            (episode_id,),
        )
        rows = self._connection.execute(
            """
            SELECT
                o.id, o.concept_id, o.event_id, o.token, o.created_at,
                e.source_type, e.context_json, e.context_signature,
                e.ingest_order, o.ordinal
            FROM occurrences o
            JOIN events e ON e.id = o.event_id
            WHERE o.episode_id = ?
            ORDER BY e.ingest_order, o.ordinal
            """,
            (episode_id,),
        ).fetchall()

        # Untrusted model output is a hard boundary, not a hidden bridge
        # between two real observations.
        segments: list[list[sqlite3.Row]] = []
        current: list[sqlite3.Row] = []
        current_event: str | None = None
        current_trusted = True
        for row in rows:
            if row["event_id"] != current_event:
                current_event = row["event_id"]
                current_trusted = row["source_type"] in _TRUSTED_SOURCES
                if not current_trusted and current:
                    segments.append(current)
                    current = []
            if current_trusted:
                current.append(row)
        if current:
            segments.append(current)

        for segment in segments:
            for next_index in range(1, len(segment)):
                next_row = segment[next_index]
                max_order = min(_MAX_PATTERN_ORDER, next_index)
                for order in range(1, max_order + 1):
                    history_rows = segment[next_index - order : next_index]
                    concept_ids = [int(item["concept_id"]) for item in history_rows]
                    pattern_id = self._get_or_create_pattern(
                        concept_ids,
                        next_row["context_json"],
                        next_row["context_signature"],
                        next_row["created_at"],
                    )
                    continuation_id = self._get_or_create_continuation(
                        pattern_id,
                        int(next_row["concept_id"]),
                    )
                    self._connection.execute(
                        """
                        INSERT INTO evidence_spans(
                            continuation_id, episode_id, event_id,
                            start_occurrence_id, end_occurrence_id,
                            next_occurrence_id, source_type, observed_at
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(
                            continuation_id, episode_id,
                            start_occurrence_id, next_occurrence_id
                        ) DO NOTHING
                        """,
                        (
                            continuation_id,
                            episode_id,
                            next_row["event_id"],
                            history_rows[0]["id"],
                            history_rows[-1]["id"],
                            next_row["id"],
                            next_row["source_type"],
                            next_row["created_at"],
                        ),
                    )

    def _refresh_aggregates(self) -> None:
        self._connection.execute(
            """
            DELETE FROM continuations
            WHERE NOT EXISTS (
                SELECT 1 FROM evidence_spans es
                WHERE es.continuation_id = continuations.id
            )
            """
        )
        self._connection.execute(
            """
            UPDATE continuations
            SET support_count = (
                    SELECT COUNT(*) FROM evidence_spans es
                    WHERE es.continuation_id = continuations.id
                ),
                episode_support_count = (
                    SELECT COUNT(DISTINCT episode_id) FROM evidence_spans es
                    WHERE es.continuation_id = continuations.id
                ),
                decayed_support = CAST((
                    SELECT COUNT(*) FROM evidence_spans es
                    WHERE es.continuation_id = continuations.id
                ) AS REAL),
                first_seen = (
                    SELECT MIN(observed_at) FROM evidence_spans es
                    WHERE es.continuation_id = continuations.id
                ),
                last_seen = (
                    SELECT MAX(observed_at) FROM evidence_spans es
                    WHERE es.continuation_id = continuations.id
                )
            """
        )
        self._connection.execute(
            """
            DELETE FROM patterns
            WHERE NOT EXISTS (
                SELECT 1 FROM continuations c WHERE c.pattern_id = patterns.id
            )
            """
        )
        self._connection.execute(
            """
            DELETE FROM concepts
            WHERE NOT EXISTS (
                SELECT 1 FROM occurrences o WHERE o.concept_id = concepts.id
            )
              AND NOT EXISTS (
                SELECT 1 FROM pattern_items pi WHERE pi.concept_id = concepts.id
            )
              AND NOT EXISTS (
                SELECT 1 FROM continuations c WHERE c.to_concept_id = concepts.id
            )
            """
        )

    @staticmethod
    def _context_match(pattern_context: Mapping[str, Any], desired: Mapping[str, Any] | None) -> float:
        if desired is None:
            return 0.0
        if not desired:
            return 1.0 if not pattern_context else 0.0
        if pattern_context == desired:
            return 1.0
        if not pattern_context:
            return 0.25
        desired_items = {
            (str(key), json.dumps(_json_ready(value), sort_keys=True, ensure_ascii=False))
            for key, value in desired.items()
        }
        pattern_items = {
            (str(key), json.dumps(_json_ready(value), sort_keys=True, ensure_ascii=False))
            for key, value in pattern_context.items()
        }
        union = desired_items | pattern_items
        return len(desired_items & pattern_items) / len(union) if union else 0.0

    def _episode_payload(
        self,
        episode_id: str,
        matched_concept_ids: set[int],
        query_tokens: Sequence[str],
        desired_context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Build one bounded recall result from a consistent read snapshot.

        For a lexical query, only matching events and a one-event temporal
        neighbourhood are eligible. Both the seed set and final payload are
        capped, so one very long episode cannot dominate memory or response
        size. An empty query returns only the most recent events.
        """

        episode_row = self._connection.execute(
            "SELECT context_json, updated_at FROM episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        assert episode_row is not None

        event_count = int(
            self._connection.execute(
                "SELECT COUNT(*) AS amount FROM events WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()["amount"]
        )
        seed_rows: list[sqlite3.Row]
        if query_tokens and matched_concept_ids:
            concept_placeholders = ",".join("?" for _ in matched_concept_ids)
            seed_rows = list(
                self._connection.execute(
                    f"""
                    SELECT e.id, e.ingest_order,
                           COUNT(DISTINCT o.concept_id) AS match_count
                    FROM events e
                    JOIN occurrences o ON o.event_id = e.id
                    WHERE e.episode_id = ?
                      AND o.concept_id IN ({concept_placeholders})
                    GROUP BY e.id, e.ingest_order
                    ORDER BY match_count DESC, e.ingest_order DESC
                    LIMIT ?
                    """,
                    (
                        episode_id,
                        *matched_concept_ids,
                        _MAX_RECALL_MATCH_EVENTS,
                    ),
                ).fetchall()
            )
        else:
            seed_rows = list(
                self._connection.execute(
                    """
                    SELECT id, ingest_order, 0 AS match_count
                    FROM events
                    WHERE episode_id = ?
                    ORDER BY ingest_order DESC
                    LIMIT ?
                    """,
                    (episode_id, _MAX_RECALL_EVENTS),
                ).fetchall()
            )

        # Candidate values are (priority, event_id). Matching seeds always
        # survive the cap before their bounded temporal neighbours.
        candidates: dict[int, tuple[int, str]] = {
            int(row["ingest_order"]): (0, row["id"]) for row in seed_rows
        }
        if query_tokens:
            for seed in seed_rows:
                seed_order = int(seed["ingest_order"])
                previous_rows = self._connection.execute(
                    """
                    SELECT id, ingest_order FROM events
                    WHERE episode_id = ? AND ingest_order < ?
                    ORDER BY ingest_order DESC
                    LIMIT ?
                    """,
                    (episode_id, seed_order, _RECALL_EVENT_NEIGHBOURHOOD),
                ).fetchall()
                next_rows = self._connection.execute(
                    """
                    SELECT id, ingest_order FROM events
                    WHERE episode_id = ? AND ingest_order > ?
                    ORDER BY ingest_order
                    LIMIT ?
                    """,
                    (episode_id, seed_order, _RECALL_EVENT_NEIGHBOURHOOD),
                ).fetchall()
                for neighbour in (*previous_rows, *next_rows):
                    neighbour_order = int(neighbour["ingest_order"])
                    candidates.setdefault(neighbour_order, (1, neighbour["id"]))

        selected_candidates = sorted(
            candidates.items(),
            key=lambda item: (item[1][0], -item[0]),
        )[:_MAX_RECALL_EVENTS]
        selected_event_ids = [item[1][1] for item in selected_candidates]
        seed_event_ids = {row["id"] for row in seed_rows}
        if selected_event_ids:
            event_placeholders = ",".join("?" for _ in selected_event_ids)
            event_rows = self._connection.execute(
                f"""
                SELECT id, ingest_order, text, source_type, context_json, created_at
                FROM events
                WHERE id IN ({event_placeholders})
                ORDER BY ingest_order
                """,
                tuple(selected_event_ids),
            ).fetchall()
        else:
            event_rows = []

        matched_rows: list[sqlite3.Row] = []
        matched_tokens: list[str] = []
        matched_occurrence_count = 0
        if query_tokens and selected_event_ids and matched_concept_ids:
            event_placeholders = ",".join("?" for _ in selected_event_ids)
            concept_placeholders = ",".join("?" for _ in matched_concept_ids)
            match_parameters = (*selected_event_ids, *matched_concept_ids)
            matched_rows = list(
                self._connection.execute(
                    f"""
                    SELECT o.id, o.event_id, o.concept_id, o.token, o.ordinal,
                           o.created_at
                    FROM occurrences o
                    JOIN events e ON e.id = o.event_id
                    WHERE o.event_id IN ({event_placeholders})
                      AND o.concept_id IN ({concept_placeholders})
                    ORDER BY e.ingest_order, o.ordinal
                    LIMIT ?
                    """,
                    (*match_parameters, _MAX_RECALL_EVIDENCE),
                ).fetchall()
            )
            matched_occurrence_count = int(
                self._connection.execute(
                    f"""
                    SELECT COUNT(*) AS amount
                    FROM occurrences
                    WHERE event_id IN ({event_placeholders})
                      AND concept_id IN ({concept_placeholders})
                    """,
                    match_parameters,
                ).fetchone()["amount"]
            )
            token_rows = self._connection.execute(
                f"""
                SELECT DISTINCT c.canonical_key
                FROM occurrences o
                JOIN concepts c ON c.id = o.concept_id
                WHERE o.event_id IN ({event_placeholders})
                  AND o.concept_id IN ({concept_placeholders})
                """,
                match_parameters,
            ).fetchall()
            token_set = {row["canonical_key"] for row in token_rows}
            matched_tokens = [
                token for token in dict.fromkeys(query_tokens) if token in token_set
            ]

        path_rows: list[sqlite3.Row] = []
        path_truncated = False
        if selected_event_ids:
            event_placeholders = ",".join("?" for _ in selected_event_ids)
            path_rows = list(
                self._connection.execute(
                    f"""
                    SELECT o.token
                    FROM occurrences o
                    JOIN events e ON e.id = o.event_id
                    WHERE o.event_id IN ({event_placeholders})
                    ORDER BY e.ingest_order, o.ordinal
                    LIMIT ?
                    """,
                    (*selected_event_ids, _MAX_RECALL_PATH_ITEMS + 1),
                ).fetchall()
            )
            path_truncated = len(path_rows) > _MAX_RECALL_PATH_ITEMS
            path_rows = path_rows[:_MAX_RECALL_PATH_ITEMS]

        episode_context = json.loads(episode_row["context_json"])
        query_unique = list(dict.fromkeys(query_tokens))
        matched_unique = len(set(matched_tokens))
        coverage = matched_unique / max(1, len(set(query_unique))) if query_unique else 0.0
        occurrence_component = (
            math.log1p(matched_occurrence_count) if matched_occurrence_count else 0.0
        )
        context_component = self._context_match(episode_context, desired_context)
        age_days = max(
            0.0,
            (datetime.now(timezone.utc) - _parse_datetime(episode_row["updated_at"])).total_seconds()
            / 86400.0,
        )
        recency_component = 1.0 / (1.0 + age_days / 30.0)
        match_component = float(matched_unique * 2)
        coverage_component = coverage * 2.0
        score = (
            match_component
            + coverage_component
            + occurrence_component
            + context_component
            + recency_component
        )
        if not query_tokens:
            score = recency_component

        evidence = [
            {
                "occurrence_id": row["id"],
                "event_id": row["event_id"],
                "concept": row["token"],
                "ordinal": row["ordinal"],
                "observed_at": row["created_at"],
            }
            for row in matched_rows
        ]
        path = [row["token"] for row in path_rows]
        events: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for row in event_rows:
            full_text = row["text"]
            text_truncated = len(full_text) > _MAX_RETURNED_EVENT_TEXT
            returned_text = (
                full_text[: _MAX_RETURNED_EVENT_TEXT - 1] + "\u2026"
                if text_truncated
                else full_text
            )
            text_parts.append(returned_text)
            events.append(
                {
                    "event_id": row["id"],
                    "text": returned_text,
                    "text_truncated": text_truncated,
                    "source": row["source_type"],
                    "context": json.loads(row["context_json"]),
                    "created_at": row["created_at"],
                    "matched_query": row["id"] in seed_event_ids,
                }
            )
        returned_event_count = len(events)
        events_truncated = event_count > returned_event_count
        return {
            "episode_id": episode_id,
            "score": round(score, 6),
            "score_kind": "recall_score_v1",
            "text": "\n".join(text_parts),
            "query_concepts": query_unique,
            "matched_concepts": matched_tokens,
            "context": episode_context,
            "events": events,
            "event_count": event_count,
            "returned_event_count": returned_event_count,
            "events_truncated": events_truncated,
            "path": path,
            "path_truncated": path_truncated,
            "evidence": evidence,
            "explanation": {
                "summary": (
                    "Episode recent retourne sans indice lexical."
                    if not query_tokens
                    else f"{matched_unique} concept(s) de la requete retrouves dans cet episode."
                ),
                "components": {
                    "concept_matches": round(match_component, 6),
                    "query_coverage": round(coverage_component, 6),
                    "occurrence_support": round(occurrence_component, 6),
                    "context_match": round(context_component, 6),
                    "recency": round(recency_component, 6),
                },
                "algorithm_version": "recall-v1",
                "proof_count": len(evidence),
                "total_matching_occurrences": matched_occurrence_count,
                "evidence_truncated": matched_occurrence_count > len(evidence),
                "event_selection": (
                    "matching_events_with_bounded_neighbourhood"
                    if query_tokens
                    else "most_recent_events"
                ),
            },
        }

    def recall(self, query: str | Mapping[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve episodes that converge on the concepts in ``query``.

        An empty query returns the most recent episodes. A mapping may be used
        as ``{"query": "...", "context": {...}}`` to add a transparent
        context score without making context an access boundary. Query length,
        returned events, neighbourhood, path and evidence are all bounded.
        """

        self._ensure_open()
        limit = _bounded_top_k(top_k)
        desired_context: Mapping[str, Any] | None = None
        if isinstance(query, Mapping):
            query_text = query.get("query", query.get("text", ""))
            desired_context = query.get("context")
            if desired_context is not None and not isinstance(desired_context, Mapping):
                raise TypeError("query['context'] doit etre un dictionnaire")
        else:
            query_text = query
        if not isinstance(query_text, str):
            raise TypeError("query doit etre une chaine ou un dictionnaire")
        if len(query_text) > _MAX_QUERY_CHARACTERS:
            raise ValueError(
                f"query depasse la limite de {_MAX_QUERY_CHARACTERS} caracteres"
            )
        query_tokens = _tokenise(query_text)
        if len(query_tokens) > _MAX_QUERY_TOKENS:
            raise ValueError(
                f"query depasse la limite de {_MAX_QUERY_TOKENS} tokens"
            )

        with self._lock:
            self._ensure_open()
            with self._read_snapshot():
                if not query_tokens:
                    episode_rows = self._connection.execute(
                        "SELECT id FROM episodes ORDER BY updated_at DESC, id LIMIT ?",
                        (limit,),
                    ).fetchall()
                    payloads = [
                        self._episode_payload(row["id"], set(), [], desired_context)
                        for row in episode_rows
                    ]
                    return payloads

                unique_tokens = list(dict.fromkeys(query_tokens))
                placeholders = ",".join("?" for _ in unique_tokens)
                concept_rows = self._connection.execute(
                    f"SELECT id, canonical_key FROM concepts WHERE canonical_key IN ({placeholders})",
                    tuple(unique_tokens),
                ).fetchall()
                if not concept_rows:
                    return []
                concept_ids = {int(row["id"]) for row in concept_rows}
                id_placeholders = ",".join("?" for _ in concept_ids)
                episode_rows = self._connection.execute(
                    f"""
                    SELECT o.episode_id,
                           COUNT(DISTINCT o.concept_id) AS matched,
                           COUNT(*) AS occurrences,
                           MAX(o.created_at) AS latest
                    FROM occurrences o
                    WHERE o.concept_id IN ({id_placeholders})
                    GROUP BY o.episode_id
                    ORDER BY matched DESC, occurrences DESC, latest DESC
                    LIMIT ?
                    """,
                    (*concept_ids, max(limit * 4, limit)),
                ).fetchall()
                payloads = [
                    self._episode_payload(
                        row["episode_id"], concept_ids, query_tokens, desired_context
                    )
                    for row in episode_rows
                ]
                payloads.sort(
                    key=lambda item: (
                        item["score"],
                        len(item["evidence"]),
                        item["episode_id"],
                    ),
                    reverse=True,
                )
                return payloads[:limit]

    def _continuation_evidence(self, continuation_id: int, limit: int = 5) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT
                es.id, es.episode_id, es.event_id, es.start_occurrence_id,
                es.end_occurrence_id, es.next_occurrence_id,
                es.source_type, es.observed_at, es.algorithm_version,
                e.text
            FROM evidence_spans es
            JOIN events e ON e.id = es.event_id
            WHERE es.continuation_id = ?
            ORDER BY es.observed_at DESC, es.id DESC
            LIMIT ?
            """,
            (continuation_id, limit),
        ).fetchall()
        payloads: list[dict[str, Any]] = []
        for row in rows:
            full_text = row["text"]
            text_truncated = len(full_text) > _MAX_RETURNED_EVENT_TEXT
            payloads.append({
                "evidence_id": row["id"],
                "episode_id": row["episode_id"],
                "event_id": row["event_id"],
                "text": (
                    full_text[: _MAX_RETURNED_EVENT_TEXT - 1] + "\u2026"
                    if text_truncated
                    else full_text
                ),
                "text_truncated": text_truncated,
                "source": row["source_type"],
                "occurrence_ids": {
                    "start": row["start_occurrence_id"],
                    "end": row["end_occurrence_id"],
                    "next": row["next_occurrence_id"],
                },
                "observed_at": row["observed_at"],
                "algorithm_version": row["algorithm_version"],
            })
        return payloads

    def predict(self, history: Any, top_k: int = 5) -> list[dict[str, Any]]:
        """Rank next concepts on one coherent SQLite read snapshot."""

        self._ensure_open()
        limit = _bounded_top_k(top_k)
        history_tokens, desired_context = self._normalise_history(history)
        if not history_tokens:
            return []

        with self._lock:
            self._ensure_open()
            with self._read_snapshot():
                return self._predict_from_tokens(history_tokens, desired_context, limit)

    def _predict_from_tokens(
        self,
        history_tokens: list[str],
        desired_context: Mapping[str, Any] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Predict while the caller holds the lock and a read transaction."""

        with self._lock:
            self._ensure_open()
            selected_rows: list[sqlite3.Row] = []
            selected_suffix: list[str] = []
            selected_length = 0

            for order in range(min(_MAX_PATTERN_ORDER, len(history_tokens)), 0, -1):
                suffix = history_tokens[-order:]
                placeholders = ",".join("?" for _ in suffix)
                concept_rows = self._connection.execute(
                    f"SELECT id, canonical_key FROM concepts WHERE canonical_key IN ({placeholders})",
                    tuple(suffix),
                ).fetchall()
                by_key = {row["canonical_key"]: int(row["id"]) for row in concept_rows}
                if any(token not in by_key for token in suffix):
                    continue
                sequence_ids = [by_key[token] for token in suffix]
                sequence_json = json.dumps(sequence_ids, separators=(",", ":"))
                rows = self._connection.execute(
                    """
                    SELECT
                        c.id AS continuation_id, c.to_concept_id,
                        c.support_count, c.episode_support_count,
                        c.first_seen, c.last_seen,
                        p.id AS pattern_id, p.context_json,
                        p.context_signature, p.length,
                        target.canonical_key, target.label
                    FROM patterns p
                    JOIN continuations c ON c.pattern_id = p.id
                    JOIN concepts target ON target.id = c.to_concept_id
                    WHERE p.sequence_json = ?
                      AND c.support_count > 0
                    ORDER BY c.support_count DESC, c.last_seen DESC
                    """,
                    (sequence_json,),
                ).fetchall()
                if rows:
                    selected_rows = list(rows)
                    selected_suffix = suffix
                    selected_length = order
                    break

            if not selected_rows:
                return []

            grouped: dict[int, dict[str, Any]] = {}
            for row in selected_rows:
                concept_id = int(row["to_concept_id"])
                bucket = grouped.setdefault(
                    concept_id,
                    {
                        "concept": row["canonical_key"],
                        "label": row["label"],
                        "support_count": 0,
                        "episode_ids": set(),
                        "last_seen": row["last_seen"],
                        "context_match": 0.0,
                        "continuation_ids": [],
                        "contexts": [],
                        "evidence": [],
                    },
                )
                bucket["support_count"] += int(row["support_count"])
                pattern_context = json.loads(row["context_json"])
                bucket["context_match"] = max(
                    bucket["context_match"],
                    self._context_match(pattern_context, desired_context),
                )
                bucket["contexts"].append(pattern_context)
                bucket["continuation_ids"].append(int(row["continuation_id"]))
                evidence = self._continuation_evidence(int(row["continuation_id"]), limit=5)
                bucket["evidence"].extend(evidence)
                bucket["episode_ids"].update(item["episode_id"] for item in evidence)
                if (row["last_seen"] or "") > (bucket["last_seen"] or ""):
                    bucket["last_seen"] = row["last_seen"]

            results: list[dict[str, Any]] = []
            for bucket in grouped.values():
                # Fetch every supporting episode count exactly; the evidence
                # payload itself remains deliberately bounded.
                continuation_ids = bucket["continuation_ids"]
                placeholders = ",".join("?" for _ in continuation_ids)
                support_row = self._connection.execute(
                    f"""
                    SELECT COUNT(DISTINCT episode_id) AS episode_count
                    FROM evidence_spans
                    WHERE continuation_id IN ({placeholders})
                    """,
                    tuple(continuation_ids),
                ).fetchone()
                episode_support = int(support_row["episode_count"])
                support_component = math.log1p(bucket["support_count"]) * 2.0
                episode_component = math.log1p(episode_support)
                suffix_component = selected_length * 0.75
                context_component = float(bucket["context_match"])
                age_days = max(
                    0.0,
                    (
                        datetime.now(timezone.utc)
                        - _parse_datetime(bucket["last_seen"])
                    ).total_seconds()
                    / 86400.0,
                )
                recency_component = 1.0 / (1.0 + age_days / 30.0)
                score = (
                    support_component
                    + episode_component
                    + suffix_component
                    + context_component
                    + recency_component
                )
                evidence_by_id = {
                    item["evidence_id"]: item for item in bucket["evidence"]
                }
                evidence = sorted(
                    evidence_by_id.values(),
                    key=lambda item: (item["observed_at"], item["evidence_id"]),
                    reverse=True,
                )[:10]
                contexts: list[dict[str, Any]] = []
                seen_contexts: set[str] = set()
                for item in bucket["contexts"]:
                    encoded = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if encoded not in seen_contexts:
                        contexts.append(item)
                        seen_contexts.add(encoded)
                result = {
                    "concept": bucket["concept"],
                    "label": bucket["label"],
                    "score": round(score, 6),
                    "score_kind": "ranking_score_v1",
                    "suffix_used": selected_suffix,
                    "path": [*selected_suffix, bucket["concept"]],
                    "support_count": int(bucket["support_count"]),
                    "episode_support_count": episode_support,
                    "contexts": contexts,
                    "evidence": evidence,
                    "explanation": {
                        "summary": (
                            f"Suite observee {bucket['support_count']} fois apres "
                            f"le motif {' -> '.join(selected_suffix)}."
                        ),
                        "components": {
                            "support": round(support_component, 6),
                            "episode_diversity": round(episode_component, 6),
                            "suffix_length": round(suffix_component, 6),
                            "context_match": round(context_component, 6),
                            "recency": round(recency_component, 6),
                        },
                        "algorithm_version": "predict-v1",
                        "continuation_ids": continuation_ids,
                        "proof_count": int(bucket["support_count"]),
                    },
                }
                results.append(result)

            results.sort(
                key=lambda item: (
                    item["score"],
                    item["support_count"],
                    item["concept"],
                ),
                reverse=True,
            )
            results = results[:limit]
            total_score = sum(max(0.0, item["score"]) for item in results)
            for item in results:
                item["ranking_share"] = (
                    round(max(0.0, item["score"]) / total_score, 6)
                    if total_score
                    else 0.0
                )
                item["ranking_share_kind"] = "relative_score_not_probability"
            return results

    def _set_episode_bounds(self, episode_id: str) -> None:
        """Synchronise episode timestamps with its remaining event segment."""

        bounds = self._connection.execute(
            """
            SELECT MIN(created_at) AS first_seen, MAX(created_at) AS last_seen
            FROM events WHERE episode_id = ?
            """,
            (episode_id,),
        ).fetchone()
        if bounds is not None and bounds["first_seen"] is not None:
            self._connection.execute(
                """
                UPDATE episodes
                SET created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (bounds["first_seen"], bounds["last_seen"], episode_id),
            )

    def forget(self, event_id: str) -> dict[str, Any]:
        """Hard-delete one event without inventing a new temporal adjacency.

        If the removed event sits between older and newer events, the episode
        is split at that gap before patterns are rebuilt. The newer segment
        keeps the original episode identifier so an active caller can safely
        continue appending to it; the older segment receives a fresh ID.
        """

        self._ensure_open()
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("event_id doit etre une chaine non vide")
        target = event_id.strip()
        with self._lock:
            self._ensure_open()
            with self._transaction():
                row = self._connection.execute(
                    """
                    SELECT e.episode_id, e.ingest_order,
                           ep.context_json, ep.context_signature,
                           COUNT(o.id) AS occurrence_count
                    FROM events e
                    JOIN episodes ep ON ep.id = e.episode_id
                    LEFT JOIN occurrences o ON o.event_id = e.id
                    WHERE e.id = ?
                    GROUP BY e.id
                    """,
                    (target,),
                ).fetchone()
                if row is None:
                    return {
                        "forgotten": False,
                        "event_id": target,
                        "reason": "event_not_found",
                    }
                episode_id = row["episode_id"]
                target_order = int(row["ingest_order"])
                occurrence_count = int(row["occurrence_count"])
                before_count = int(
                    self._connection.execute(
                        """
                        SELECT COUNT(*) AS amount FROM events
                        WHERE episode_id = ? AND ingest_order < ?
                        """,
                        (episode_id, target_order),
                    ).fetchone()["amount"]
                )
                after_count = int(
                    self._connection.execute(
                        """
                        SELECT COUNT(*) AS amount FROM events
                        WHERE episode_id = ? AND ingest_order > ?
                        """,
                        (episode_id, target_order),
                    ).fetchone()["amount"]
                )
                proof_count = int(
                    self._connection.execute(
                        """
                        SELECT COUNT(*) AS amount FROM evidence_spans
                        WHERE episode_id = ?
                        """,
                        (episode_id,),
                    ).fetchone()["amount"]
                )

                # Remove every old proof before changing episode ownership.
                # The journal remains the source from which both independent
                # segments are reconstructed below.
                self._connection.execute(
                    "DELETE FROM evidence_spans WHERE episode_id = ?",
                    (episode_id,),
                )
                self._connection.execute("DELETE FROM events WHERE id = ?", (target,))

                split_episode_id: str | None = None
                remaining_episode_ids: list[str] = []
                if before_count and after_count:
                    split_episode_id = str(uuid4())
                    before_bounds = self._connection.execute(
                        """
                        SELECT MIN(created_at) AS first_seen,
                               MAX(created_at) AS last_seen
                        FROM events
                        WHERE episode_id = ? AND ingest_order < ?
                        """,
                        (episode_id, target_order),
                    ).fetchone()
                    self._connection.execute(
                        """
                        INSERT INTO episodes(
                            id, context_json, context_signature,
                            created_at, updated_at
                        ) VALUES(?, ?, ?, ?, ?)
                        """,
                        (
                            split_episode_id,
                            row["context_json"],
                            row["context_signature"],
                            before_bounds["first_seen"],
                            before_bounds["last_seen"],
                        ),
                    )
                    self._connection.execute(
                        """
                        UPDATE events SET episode_id = ?
                        WHERE episode_id = ? AND ingest_order < ?
                        """,
                        (split_episode_id, episode_id, target_order),
                    )
                    self._connection.execute(
                        """
                        UPDATE occurrences SET episode_id = ?
                        WHERE event_id IN (
                            SELECT id FROM events WHERE episode_id = ?
                        )
                        """,
                        (split_episode_id, split_episode_id),
                    )
                    self._set_episode_bounds(split_episode_id)
                    self._set_episode_bounds(episode_id)
                    self._rebuild_episode_evidence(split_episode_id)
                    self._rebuild_episode_evidence(episode_id)
                    remaining_episode_ids = [split_episode_id, episode_id]
                elif before_count or after_count:
                    self._set_episode_bounds(episode_id)
                    self._rebuild_episode_evidence(episode_id)
                    remaining_episode_ids = [episode_id]
                else:
                    self._connection.execute(
                        "DELETE FROM episodes WHERE id = ?",
                        (episode_id,),
                    )
                self._refresh_aggregates()
                return {
                    "forgotten": True,
                    "event_id": target,
                    "episode_id": episode_id,
                    "occurrences_removed": occurrence_count,
                    "proofs_rebuilt_or_removed": proof_count,
                    "episode_removed": not bool(before_count or after_count),
                    "episode_split": split_episode_id is not None,
                    "split_episode_id": split_episode_id,
                    "remaining_episode_ids": remaining_episode_ids,
                }

    def stats(self) -> dict[str, Any]:
        """Return small, non-sensitive counters describing the local memory."""

        self._ensure_open()
        table_names = {
            "episodes": "episodes",
            "events": "events",
            "concepts": "concepts",
            "occurrences": "occurrences",
            "patterns": "patterns",
            "continuations": "continuations",
            "evidence_spans": "evidence_spans",
        }
        with self._lock:
            self._ensure_open()
            with self._read_snapshot():
                counts = {
                    public_name: int(
                        self._connection.execute(
                            f"SELECT COUNT(*) AS amount FROM {table_name}"
                        ).fetchone()["amount"]
                    )
                    for public_name, table_name in table_names.items()
                }
                source_rows = self._connection.execute(
                    "SELECT source_type, COUNT(*) AS amount FROM events GROUP BY source_type"
                ).fetchall()
                source_counts = {
                    row["source_type"]: int(row["amount"]) for row in source_rows
                }
                if self.db_path == ":memory:":
                    size_bytes = None
                else:
                    path = Path(self.db_path).expanduser()
                    size_bytes = path.stat().st_size if path.exists() else 0
                return {
                    **counts,
                    "evidence": counts["evidence_spans"],
                    "trusted_events": sum(
                        source_counts.get(name, 0) for name in _TRUSTED_SOURCES
                    ),
                    "sources": source_counts,
                    "schema_version": _SCHEMA_VERSION,
                    "max_pattern_order": _MAX_PATTERN_ORDER,
                    "database": self.db_path,
                    "database_size_bytes": size_bytes,
                }

    def preview_json_import(
        self,
        data: Any,
        *,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Validate and categorise JSON without changing the database."""

        self._ensure_open()
        from .json_import import prepare_json_import, preview_json_import

        plan = prepare_json_import(data, filename=filename)
        return preview_json_import(plan)

    def import_json(
        self,
        data: Any,
        *,
        import_id: str,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Import JSON leaves using content-derived idempotency keys.

        ``import_id`` must be the value returned by :meth:`preview_json_import`
        for the exact same data. This makes preview/confirmation mismatches
        visible instead of silently importing a changed document.
        """

        self._ensure_open()
        from .json_import import commit_json_import, prepare_json_import

        plan = prepare_json_import(data, filename=filename)
        return commit_json_import(self, plan, import_id=import_id)

    def close(self) -> None:
        """Close the SQLite connection. Safe to call more than once."""

        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "MemoryEngine":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Interpreter shutdown can remove module globals in arbitrary order;
        # never surface an exception from a best-effort cleanup.
        try:
            self.close()
        except Exception:
            pass
