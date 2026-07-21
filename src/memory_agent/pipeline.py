"""Pipeline durable d'injection et de consolidation de la memoire.

Ce module separe volontairement trois responsabilites :

* :class:`DurableInjectionQueue` journalise rapidement les observations;
* :class:`BackgroundConsolidator` les transmet au moteur d'apprentissage;
* :class:`MemoryPipeline` fournit un assemblage avec un lecteur SQLite
  distinct de l'ecrivain.

La file est une base SQLite independante. Un travail n'est marque termine
qu'apres le retour de ``MemoryEngine.observe``. Si le processus s'interrompt
entre ces deux operations, la reprise est sans double apprentissage : la cle
d'idempotence de la source est conservee dans le journal puis rejouee au
moteur.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from .memory import MemoryEngine, MemoryIdempotencyConflictError


_QUEUE_SCHEMA_VERSION = 3
_STATES = ("pending", "processing", "completed", "failed")
_TEST_RUN_STATES = ("active", "cleaning", "cleanup_failed", "cleaned")
_MAX_BATCH_SIZE = 1_000
_MAX_TEST_RUN_ITEMS = 250
_MAX_ERROR_CHARACTERS = 4_000


class InjectionQueueError(RuntimeError):
    """Base class for durable injection queue errors."""


class IdempotencyConflictError(InjectionQueueError):
    """Raised when one idempotency key is reused for another payload."""


class QueueStateError(InjectionQueueError):
    """Raised when a stale worker tries to finish a job it does not own."""


def _utc_timestamp() -> float:
    return time.time()


def _iso_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(
        timespec="microseconds"
    )


def _sqlite_file_sizes(db_path: str) -> dict[str, int]:
    if db_path == ":memory:":
        return {"main": 0, "wal": 0, "shm": 0, "total": 0}
    path = Path(db_path).expanduser()
    main = path.stat().st_size if path.exists() else 0
    wal_path = Path(str(path) + "-wal")
    shm_path = Path(str(path) + "-shm")
    wal = wal_path.stat().st_size if wal_path.exists() else 0
    shm = shm_path.stat().st_size if shm_path.exists() else 0
    return {"main": main, "wal": wal, "shm": shm, "total": main + wal + shm}


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalised = [_json_value(item) for item in value]
        return sorted(
            normalised,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    if isinstance(value, datetime):
        parsed = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    return str(value)


def _encode_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _positive_integer(value: int, *, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} doit etre un entier")
    if value < 1:
        raise ValueError(f"{name} doit etre superieur ou egal a 1")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} ne peut pas depasser {maximum}")
    return value


class DurableInjectionQueue:
    """File SQLite durable avec livraison idempotente au moins une fois.

    ``enqueue`` exige une cle d'idempotence fournie par la source. Rejouer la
    meme cle et le meme contenu retourne le travail original; reutiliser la
    cle avec un contenu different est refuse explicitement.
    """

    def __init__(self, db_path: str | Path):
        if db_path is None:
            raise TypeError("db_path est obligatoire")
        raw_path = str(db_path)
        if not raw_path.strip():
            raise ValueError("db_path ne peut pas etre vide")

        self.db_path = raw_path
        if raw_path != ":memory:":
            Path(raw_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
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
            self._connection.execute("PRAGMA busy_timeout = 30000")
            if raw_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
                # Un HTTP 202 signifie que le journal d'injection est durable,
                # y compris face a une panne OS/electrique apres le commit.
                self._connection.execute("PRAGMA synchronous = FULL")
            self._create_schema()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS injection_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS injection_jobs (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                idempotency_key TEXT NOT NULL UNIQUE,
                payload_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                episode_id TEXT,
                context_json TEXT NOT NULL,
                source_json TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending'
                    CHECK(state IN ('pending', 'processing', 'completed', 'failed')),
                attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
                max_attempts INTEGER NOT NULL CHECK(max_attempts >= 1),
                available_at REAL NOT NULL,
                enqueued_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                claimed_at REAL,
                completed_at REAL,
                claimed_by TEXT,
                last_error TEXT,
                result_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_injection_jobs_ready
                ON injection_jobs(state, available_at, sequence);
            CREATE INDEX IF NOT EXISTS idx_injection_jobs_claimed
                ON injection_jobs(state, claimed_by);
            CREATE INDEX IF NOT EXISTS idx_injection_jobs_last_error
                ON injection_jobs(updated_at DESC)
                WHERE last_error IS NOT NULL;

            CREATE TABLE IF NOT EXISTS injection_test_runs (
                run_id TEXT PRIMARY KEY,
                expected_count INTEGER NOT NULL CHECK(expected_count >= 1),
                state TEXT NOT NULL DEFAULT 'active'
                    CHECK(state IN ('active', 'cleaning', 'cleanup_failed', 'cleaned')),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                cleaned_at REAL,
                successful_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0,
                forgotten_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                cleanup_owner TEXT,
                cleanup_claimed_at REAL
            );

            CREATE TABLE IF NOT EXISTS injection_test_run_jobs (
                run_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
                job_id TEXT NOT NULL UNIQUE,
                PRIMARY KEY(run_id, ordinal),
                FOREIGN KEY(run_id) REFERENCES injection_test_runs(run_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_injection_test_runs_state
                ON injection_test_runs(state, updated_at);
            """
        )
        row = self._connection.execute(
            "SELECT value FROM injection_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO injection_metadata(key, value) VALUES('schema_version', ?)",
                (str(_QUEUE_SCHEMA_VERSION),),
            )
        elif int(row["value"]) > _QUEUE_SCHEMA_VERSION:
            raise RuntimeError(
                "Version de file SQLite incompatible: "
                f"module={_QUEUE_SCHEMA_VERSION}, base={row['value']}"
            )
        elif int(row["value"]) < _QUEUE_SCHEMA_VERSION:
            # Versions 2/3 ajoutent uniquement le registre durable des essais.
            # Les tickets de la version 1 restent donc directement compatibles.
            columns = {
                item["name"]
                for item in self._connection.execute(
                    "PRAGMA table_info(injection_test_runs)"
                ).fetchall()
            }
            if "cleanup_owner" not in columns:
                self._connection.execute(
                    "ALTER TABLE injection_test_runs ADD COLUMN cleanup_owner TEXT"
                )
            if "cleanup_claimed_at" not in columns:
                self._connection.execute(
                    "ALTER TABLE injection_test_runs ADD COLUMN cleanup_claimed_at REAL"
                )
            self._connection.execute(
                "UPDATE injection_metadata SET value = ? WHERE key = 'schema_version'",
                (str(_QUEUE_SCHEMA_VERSION),),
            )
        for counter in ("enqueue_requests", "deduplicated_requests"):
            self._connection.execute(
                """
                INSERT OR IGNORE INTO injection_metadata(key, value)
                VALUES(?, '0')
                """,
                (counter,),
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("La file d'injection est fermee")

    def _begin_write(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    def _commit(self) -> None:
        self._connection.execute("COMMIT")

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.execute("ROLLBACK")

    def bind_memory(self, database_id: str) -> None:
        """Bind this journal to one stable memory database identity."""

        if not isinstance(database_id, str) or not database_id.strip():
            raise ValueError("database_id doit etre une chaine non vide")
        clean_id = database_id.strip()
        with self._lock:
            self._ensure_open()
            self._begin_write()
            try:
                row = self._connection.execute(
                    "SELECT value FROM injection_metadata WHERE key = 'memory_database_id'"
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        """
                        INSERT INTO injection_metadata(key, value)
                        VALUES('memory_database_id', ?)
                        """,
                        (clean_id,),
                    )
                elif row["value"] != clean_id:
                    count = int(
                        self._connection.execute(
                            "SELECT COUNT(*) AS amount FROM injection_jobs"
                        ).fetchone()["amount"]
                    )
                    if count:
                        raise RuntimeError(
                            "Cette file appartient a une autre base memoire"
                        )
                    self._connection.execute(
                        """
                        UPDATE injection_metadata SET value = ?
                        WHERE key = 'memory_database_id'
                        """,
                        (clean_id,),
                    )
                self._commit()
            except BaseException:
                self._rollback()
                raise

    @staticmethod
    def _row_to_dict(
        row: sqlite3.Row,
        *,
        duplicate: bool = False,
        retried: bool = False,
    ) -> dict[str, Any]:
        return {
            "job_id": row["id"],
            "sequence": int(row["sequence"]),
            "idempotency_key": row["idempotency_key"],
            "text": row["text"],
            "episode_id": row["episode_id"],
            "context": json.loads(row["context_json"]),
            "source": json.loads(row["source_json"]),
            "state": row["state"],
            "attempts": int(row["attempts"]),
            "max_attempts": int(row["max_attempts"]),
            "available_at": _iso_timestamp(row["available_at"]),
            "enqueued_at": _iso_timestamp(row["enqueued_at"]),
            "updated_at": _iso_timestamp(row["updated_at"]),
            "claimed_at": _iso_timestamp(row["claimed_at"]),
            "completed_at": _iso_timestamp(row["completed_at"]),
            "claimed_by": row["claimed_by"],
            "last_error": row["last_error"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "created": not duplicate,
            "duplicate": duplicate,
            "retried": retried,
        }

    def enqueue(
        self,
        text: str,
        *,
        idempotency_key: str,
        episode_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        source: str | Mapping[str, Any] = "observed",
        max_attempts: int = 3,
        payload_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Persist one observation and return immediately.

        The operation is safe to repeat after a network timeout as long as
        ``idempotency_key`` and the payload are unchanged.
        """

        self._ensure_open()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text doit etre une chaine non vide")
        clean_text = text.strip()
        if len(clean_text) > 1_000_000:
            raise ValueError("text depasse 1 000 000 caracteres")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key doit etre une chaine non vide")
        clean_key = idempotency_key.strip()
        if len(clean_key) > 500:
            raise ValueError("idempotency_key depasse 500 caracteres")
        if episode_id is not None:
            if not isinstance(episode_id, str) or not episode_id.strip():
                raise ValueError("episode_id doit etre une chaine non vide ou None")
            episode_id = episode_id.strip()
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("context doit etre un dictionnaire ou None")
        if not isinstance(source, (str, Mapping)):
            raise TypeError("source doit etre une chaine ou un dictionnaire")
        if payload_fingerprint is not None:
            if not isinstance(payload_fingerprint, str) or not payload_fingerprint.strip():
                raise ValueError("payload_fingerprint doit etre une chaine non vide")
            if len(payload_fingerprint) > 2_000:
                raise ValueError("payload_fingerprint depasse 2 000 caracteres")
        max_attempts = _positive_integer(max_attempts, name="max_attempts")

        context_json = _encode_json(dict(context or {}))
        source_json = _encode_json(source)
        payload_json = _encode_json(
            {
                "text": clean_text,
                "episode_id": episode_id,
                "context": json.loads(context_json),
                "source": json.loads(source_json),
                "max_attempts": max_attempts,
            }
        )
        fingerprint = payload_fingerprint.strip() if payload_fingerprint else payload_json
        payload_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        now = _utc_timestamp()

        with self._lock:
            self._ensure_open()
            self._begin_write()
            try:
                self._connection.execute(
                    """
                    UPDATE injection_metadata
                    SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
                    WHERE key = 'enqueue_requests'
                    """
                )
                existing = self._connection.execute(
                    "SELECT * FROM injection_jobs WHERE idempotency_key = ?",
                    (clean_key,),
                ).fetchone()
                if existing is not None:
                    if existing["payload_hash"] != payload_hash:
                        # A rejected request was still received and is therefore
                        # counted, but it is not a successful deduplication.
                        self._commit()
                        raise IdempotencyConflictError(
                            "Cette idempotency_key designe deja un autre contenu"
                        )
                    self._connection.execute(
                        """
                        UPDATE injection_metadata
                        SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
                        WHERE key = 'deduplicated_requests'
                        """
                    )
                    self._commit()
                    return self._row_to_dict(existing, duplicate=True)

                job_id = str(uuid4())
                self._connection.execute(
                    """
                    INSERT INTO injection_jobs(
                        id, idempotency_key, payload_hash, text, episode_id,
                        context_json, source_json, state, attempts, max_attempts,
                        available_at, enqueued_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        clean_key,
                        payload_hash,
                        clean_text,
                        episode_id,
                        context_json,
                        source_json,
                        max_attempts,
                        now,
                        now,
                        now,
                    ),
                )
                row = self._connection.execute(
                    "SELECT * FROM injection_jobs WHERE id = ?", (job_id,)
                ).fetchone()
                self._commit()
                return self._row_to_dict(row)
            except BaseException:
                self._rollback()
                raise

    def enqueue_test_run(self, run_id: str, *, count: int) -> dict[str, Any]:
        """Create a complete synthetic test run in one durable transaction.

        The run registry and every one of its tickets commit together.  A
        failure at any point therefore leaves neither an orphan run nor a
        partial set of synthetic memories.
        """

        if not isinstance(run_id, str):
            raise TypeError("run_id doit etre une chaine")
        clean_run_id = run_id.strip().lower()
        if len(clean_run_id) != 12 or any(
            character not in "0123456789abcdef" for character in clean_run_id
        ):
            raise ValueError("run_id de test invalide")
        count = _positive_integer(
            count, name="count", maximum=_MAX_TEST_RUN_ITEMS
        )
        now = _utc_timestamp()
        jobs: list[dict[str, Any]] = []

        with self._lock:
            self._ensure_open()
            self._begin_write()
            try:
                existing = self._connection.execute(
                    "SELECT expected_count FROM injection_test_runs WHERE run_id = ?",
                    (clean_run_id,),
                ).fetchone()
                if existing is not None:
                    if int(existing["expected_count"]) != count:
                        raise IdempotencyConflictError(
                            "Ce run_id de test designe deja un autre essai"
                        )
                    rows = self._connection.execute(
                        """
                        SELECT j.* FROM injection_test_run_jobs rj
                        JOIN injection_jobs j ON j.id = rj.job_id
                        WHERE rj.run_id = ? ORDER BY rj.ordinal
                        """,
                        (clean_run_id,),
                    ).fetchall()
                    self._commit()
                    return {
                        "run_id": clean_run_id,
                        "expected_count": count,
                        "jobs": [self._row_to_dict(row, duplicate=True) for row in rows],
                        "duplicate": True,
                    }

                self._connection.execute(
                    """
                    INSERT INTO injection_test_runs(
                        run_id, expected_count, state, created_at, updated_at
                    ) VALUES(?, ?, 'active', ?, ?)
                    """,
                    (clean_run_id, count, now, now),
                )
                for index in range(1, count + 1):
                    text = (
                        f"Essai pipeline {clean_run_id}, souvenir numero "
                        f"{index} sur {count}."
                    )
                    episode_id = f"pipeline-test-{clean_run_id}-{index}"
                    context = {
                        "category": "pipeline_test",
                        "run_id": clean_run_id,
                    }
                    source = {
                        "type": "generated",
                        "origin": "pipeline_test",
                        "run_id": clean_run_id,
                        "item_index": index,
                        "expected_count": count,
                    }
                    idempotency_key = f"pipeline-test:{clean_run_id}:{index}"
                    context_json = _encode_json(context)
                    source_json = _encode_json(source)
                    payload_json = _encode_json(
                        {
                            "text": text,
                            "episode_id": episode_id,
                            "context": context,
                            "source": source,
                            "max_attempts": 3,
                        }
                    )
                    payload_hash = hashlib.sha256(
                        payload_json.encode("utf-8")
                    ).hexdigest()
                    job_id = str(uuid4())
                    self._connection.execute(
                        """
                        INSERT INTO injection_jobs(
                            id, idempotency_key, payload_hash, text, episode_id,
                            context_json, source_json, state, attempts, max_attempts,
                            available_at, enqueued_at, updated_at
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, 'pending', 0, 3, ?, ?, ?)
                        """,
                        (
                            job_id,
                            idempotency_key,
                            payload_hash,
                            text,
                            episode_id,
                            context_json,
                            source_json,
                            now,
                            now,
                            now,
                        ),
                    )
                    self._connection.execute(
                        """
                        INSERT INTO injection_test_run_jobs(run_id, ordinal, job_id)
                        VALUES(?, ?, ?)
                        """,
                        (clean_run_id, index, job_id),
                    )
                    row = self._connection.execute(
                        "SELECT * FROM injection_jobs WHERE id = ?", (job_id,)
                    ).fetchone()
                    jobs.append(self._row_to_dict(row))
                self._connection.execute(
                    """
                    UPDATE injection_metadata
                    SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT)
                    WHERE key = 'enqueue_requests'
                    """,
                    (count,),
                )
                self._commit()
                return {
                    "run_id": clean_run_id,
                    "expected_count": count,
                    "jobs": jobs,
                    "duplicate": False,
                }
            except BaseException:
                self._rollback()
                raise

    def get_by_idempotency_key(
        self, idempotency_key: str
    ) -> dict[str, Any] | None:
        """Look up the original durable ticket for a source request."""

        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key doit etre une chaine non vide")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT * FROM injection_jobs WHERE idempotency_key = ?",
                (idempotency_key.strip(),),
            ).fetchone()
            return self._row_to_dict(row) if row is not None else None

    def get(self, job_id: str) -> dict[str, Any] | None:
        self._ensure_open()
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id doit etre une chaine non vide")
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT * FROM injection_jobs WHERE id = ?", (job_id.strip(),)
            ).fetchone()
            return self._row_to_dict(row) if row is not None else None

    def get_many(self, job_ids: list[str]) -> list[dict[str, Any]]:
        """Read several tickets in one bounded query."""

        if not isinstance(job_ids, list) or len(job_ids) > 250:
            raise ValueError("job_ids doit etre une liste de 250 elements maximum")
        clean_ids = list(dict.fromkeys(str(value).strip() for value in job_ids))
        if any(not value for value in clean_ids):
            raise ValueError("Chaque job_id doit etre non vide")
        if not clean_ids:
            return []
        with self._lock:
            self._ensure_open()
            placeholders = ",".join("?" for _ in clean_ids)
            rows = self._connection.execute(
                f"SELECT * FROM injection_jobs WHERE id IN ({placeholders})",
                clean_ids,
            ).fetchall()
            by_id = {row["id"]: row for row in rows}
            return [
                self._row_to_dict(by_id[job_id])
                for job_id in clean_ids
                if job_id in by_id
            ]

    def _test_run_summary(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            """
            SELECT run_id, expected_count, state, created_at, updated_at,
                   cleaned_at, successful_count, failed_count,
                   forgotten_count, last_error, cleanup_owner,
                   cleanup_claimed_at
            FROM injection_test_runs WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        job_rows = self._connection.execute(
            """
            SELECT rj.ordinal, rj.job_id, j.idempotency_key, j.state,
                   j.result_json, j.last_error
            FROM injection_test_run_jobs rj
            LEFT JOIN injection_jobs j ON j.id = rj.job_id
            WHERE rj.run_id = ? ORDER BY rj.ordinal
            """,
            (run_id,),
        ).fetchall()
        live_counts = {state: 0 for state in _STATES}
        for job_row in job_rows:
            state = job_row["state"]
            if state in live_counts:
                live_counts[state] += 1
        cleaned = row["state"] == "cleaned"
        successful_count = (
            int(row["successful_count"])
            if cleaned
            else live_counts["completed"]
        )
        failed_count = (
            int(row["failed_count"]) if cleaned else live_counts["failed"]
        )
        terminal_count = successful_count + failed_count
        return {
            "run_id": row["run_id"],
            "expected_count": int(row["expected_count"]),
            "state": row["state"],
            "created_at": _iso_timestamp(row["created_at"]),
            "updated_at": _iso_timestamp(row["updated_at"]),
            "cleaned_at": _iso_timestamp(row["cleaned_at"]),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "forgotten_count": int(row["forgotten_count"]),
            "terminal_count": terminal_count,
            "last_error": row["last_error"],
            "cleanup_owner": row["cleanup_owner"],
            "cleanup_claimed_at": _iso_timestamp(row["cleanup_claimed_at"]),
            "job_ids": [job_row["job_id"] for job_row in job_rows],
            "jobs": [
                {
                    "job_id": job_row["job_id"],
                    "ordinal": int(job_row["ordinal"]),
                    "idempotency_key": job_row["idempotency_key"],
                    "state": job_row["state"] or "purged",
                    "last_error": job_row["last_error"],
                    "result": (
                        json.loads(job_row["result_json"])
                        if job_row["result_json"]
                        else None
                    ),
                }
                for job_row in job_rows
            ],
        }

    def get_test_run(self, run_id: str) -> dict[str, Any] | None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id doit etre une chaine non vide")
        with self._lock:
            self._ensure_open()
            return self._test_run_summary(run_id.strip().lower())

    def list_test_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent durable test-run states for UI reconnection."""

        limit = _positive_integer(limit, name="limit", maximum=250)
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT run_id FROM injection_test_runs
                ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                summary
                for row in rows
                if (summary := self._test_run_summary(row["run_id"])) is not None
            ]

    def claim_test_run_for_cleanup(
        self,
        run_id: str | None = None,
        *,
        owner: str,
        stale_after_seconds: float = 30.0,
        retry_after_seconds: float = 1.0,
    ) -> dict[str, Any] | None:
        """Claim one fully terminal test run for idempotent cleanup."""

        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds ne peut pas etre negatif")
        if retry_after_seconds < 0:
            raise ValueError("retry_after_seconds ne peut pas etre negatif")
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("owner doit etre une chaine non vide")
        clean_owner = owner.strip()
        clean_run_id = run_id.strip().lower() if isinstance(run_id, str) else None
        now = _utc_timestamp()
        stale_cutoff = now - stale_after_seconds
        retry_cutoff = now - retry_after_seconds
        with self._lock:
            self._ensure_open()
            self._begin_write()
            try:
                parameters: list[Any] = [retry_cutoff, stale_cutoff]
                run_filter = ""
                if clean_run_id:
                    run_filter = "AND r.run_id = ?"
                    parameters.append(clean_run_id)
                row = self._connection.execute(
                    f"""
                    SELECT r.run_id
                    FROM injection_test_runs r
                    JOIN injection_test_run_jobs rj ON rj.run_id = r.run_id
                    JOIN injection_jobs j ON j.id = rj.job_id
                    WHERE (
                        r.state = 'active'
                        OR (r.state = 'cleanup_failed' AND r.updated_at <= ?)
                        OR (r.state = 'cleaning' AND (
                            r.cleanup_claimed_at IS NULL
                            OR r.cleanup_claimed_at <= ?
                        ))
                    )
                    {run_filter}
                    GROUP BY r.run_id, r.expected_count, r.created_at
                    HAVING COUNT(*) = r.expected_count
                       AND SUM(CASE WHEN j.state IN ('completed', 'failed')
                                    THEN 1 ELSE 0 END) = r.expected_count
                    ORDER BY r.created_at
                    LIMIT 1
                    """,
                    parameters,
                ).fetchone()
                if row is None:
                    self._commit()
                    return None
                self._connection.execute(
                    """
                    UPDATE injection_test_runs
                    SET state = 'cleaning', updated_at = ?, last_error = NULL,
                        cleanup_owner = ?, cleanup_claimed_at = ?
                    WHERE run_id = ?
                    """,
                    (now, clean_owner, now, row["run_id"]),
                )
                self._commit()
                return self._test_run_summary(row["run_id"])
            except BaseException:
                self._rollback()
                raise

    def finish_test_run_cleanup(
        self,
        run_id: str,
        *,
        forgotten_count: int,
        owner: str,
    ) -> dict[str, Any]:
        """Durably purge all terminal tickets and mark their run cleaned."""

        clean_run_id = str(run_id).strip().lower()
        now = _utc_timestamp()
        with self._lock:
            self._ensure_open()
            self._begin_write()
            try:
                run = self._connection.execute(
                    """
                    SELECT expected_count, state, cleanup_owner
                    FROM injection_test_runs
                    WHERE run_id = ?
                    """,
                    (clean_run_id,),
                ).fetchone()
                if run is None:
                    raise QueueStateError("Run de test inconnu")
                if run["state"] == "cleaned":
                    self._commit()
                    summary = self._test_run_summary(clean_run_id)
                    assert summary is not None
                    return summary
                if run["state"] != "cleaning" or run["cleanup_owner"] != owner:
                    raise QueueStateError("Le lease de nettoyage n'appartient plus a ce processus")
                jobs = self._connection.execute(
                    """
                    SELECT j.state FROM injection_test_run_jobs rj
                    JOIN injection_jobs j ON j.id = rj.job_id
                    WHERE rj.run_id = ?
                    """,
                    (clean_run_id,),
                ).fetchall()
                if len(jobs) != int(run["expected_count"]) or any(
                    job["state"] not in {"completed", "failed"} for job in jobs
                ):
                    raise QueueStateError("Le run de test n'est pas entierement terminal")
                successful = sum(job["state"] == "completed" for job in jobs)
                failed = sum(job["state"] == "failed" for job in jobs)
                cursor = self._connection.execute(
                    """
                    DELETE FROM injection_jobs
                    WHERE id IN (
                        SELECT job_id FROM injection_test_run_jobs WHERE run_id = ?
                    ) AND state IN ('completed', 'failed')
                    """,
                    (clean_run_id,),
                )
                if int(cursor.rowcount) != int(run["expected_count"]):
                    raise QueueStateError("La purge du run de test est incomplete")
                self._connection.execute(
                    """
                    UPDATE injection_test_runs
                    SET state = 'cleaned', updated_at = ?, cleaned_at = ?,
                        successful_count = ?, failed_count = ?,
                        forgotten_count = ?, last_error = NULL,
                        cleanup_owner = NULL, cleanup_claimed_at = NULL
                    WHERE run_id = ? AND cleanup_owner = ?
                    """,
                    (
                        now,
                        now,
                        successful,
                        failed,
                        max(0, int(forgotten_count)),
                        clean_run_id,
                        owner,
                    ),
                )
                self._commit()
                summary = self._test_run_summary(clean_run_id)
                assert summary is not None
                return summary
            except BaseException:
                self._rollback()
                raise

    def mark_test_run_cleanup_failed(
        self, run_id: str, error: BaseException | str, *, owner: str
    ) -> None:
        message = (
            f"{type(error).__name__}: {error}"
            if isinstance(error, BaseException)
            else str(error)
        )[:_MAX_ERROR_CHARACTERS]
        with self._lock:
            self._ensure_open()
            self._connection.execute(
                """
                UPDATE injection_test_runs
                SET state = 'cleanup_failed', updated_at = ?, last_error = ?,
                    cleanup_owner = NULL, cleanup_claimed_at = NULL
                WHERE run_id = ? AND state = 'cleaning' AND cleanup_owner = ?
                """,
                (
                    _utc_timestamp(),
                    message,
                    str(run_id).strip().lower(),
                    owner,
                ),
            )

    def recover_processing(
        self, *, stale_after_seconds: float = 30.0
    ) -> dict[str, int]:
        """Requeue only expired leases without consuming a retry attempt."""

        self._ensure_open()
        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds ne peut pas etre negatif")
        now = _utc_timestamp()
        cutoff = now - stale_after_seconds
        with self._lock:
            self._ensure_open()
            self._begin_write()
            try:
                recovered = self._connection.execute(
                    """
                    UPDATE injection_jobs
                        SET state = 'pending', available_at = ?, updated_at = ?,
                        attempts = CASE WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                        claimed_by = NULL, claimed_at = NULL,
                        last_error = COALESCE(last_error, 'worker_interrupted')
                    WHERE state = 'processing'
                      AND (claimed_at IS NULL OR claimed_at <= ?)
                    """,
                    (now, now, cutoff),
                ).rowcount
                self._commit()
                return {"recovered": int(recovered), "failed": 0}
            except BaseException:
                self._rollback()
                raise

    def heartbeat(self, worker_id: str) -> int:
        """Renew leases held by one live worker."""

        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id doit etre une chaine non vide")
        now = _utc_timestamp()
        with self._lock:
            self._ensure_open()
            return int(
                self._connection.execute(
                    """
                    UPDATE injection_jobs SET claimed_at = ?
                    WHERE state = 'processing' AND claimed_by = ?
                    """,
                    (now, worker_id.strip()),
                ).rowcount
            )

    def claim(self, *, batch_size: int, worker_id: str) -> list[dict[str, Any]]:
        """Atomically reserve up to ``batch_size`` ready jobs."""

        self._ensure_open()
        batch_size = _positive_integer(
            batch_size, name="batch_size", maximum=_MAX_BATCH_SIZE
        )
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id doit etre une chaine non vide")
        clean_worker = worker_id.strip()
        now = _utc_timestamp()
        with self._lock:
            self._ensure_open()
            self._begin_write()
            try:
                # Un seul lot peut etre actif. Cela preserve l'ordre global et
                # empeche deux processus de traiter deux morceaux du meme episode.
                processing = self._connection.execute(
                    "SELECT 1 FROM injection_jobs WHERE state = 'processing' LIMIT 1"
                ).fetchone()
                if processing is not None:
                    self._commit()
                    return []

                candidates = self._connection.execute(
                    """
                    SELECT id, available_at FROM injection_jobs
                    WHERE state = 'pending'
                      AND attempts < max_attempts
                    ORDER BY sequence
                    LIMIT ?
                    """,
                    (batch_size,),
                ).fetchall()
                ids: list[str] = []
                for row in candidates:
                    if float(row["available_at"]) > now:
                        break
                    ids.append(row["id"])
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    self._connection.execute(
                        f"""
                        UPDATE injection_jobs
                        SET state = 'processing', attempts = attempts + 1,
                            claimed_at = ?, updated_at = ?, claimed_by = ?,
                            last_error = NULL
                        WHERE id IN ({placeholders}) AND state = 'pending'
                        """,
                        (now, now, clean_worker, *ids),
                    )
                    claimed = self._connection.execute(
                        f"""
                        SELECT * FROM injection_jobs
                        WHERE id IN ({placeholders}) AND claimed_by = ?
                        ORDER BY sequence
                        """,
                        (*ids, clean_worker),
                    ).fetchall()
                else:
                    claimed = []
                self._commit()
                return [self._row_to_dict(row) for row in claimed]
            except BaseException:
                self._rollback()
                raise

    def complete(
        self,
        job_id: str,
        *,
        worker_id: str,
        result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()
        now = _utc_timestamp()
        raw_result = dict(result or {})
        compact_result = {
            key: raw_result.get(key)
            for key in ("event_id", "episode_id", "created", "duplicate")
            if key in raw_result
        }
        result_json = _encode_json(compact_result)
        with self._lock:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                UPDATE injection_jobs
                SET state = 'completed', updated_at = ?, completed_at = ?,
                    claimed_by = NULL, claimed_at = NULL, last_error = NULL,
                    result_json = ?
                WHERE id = ? AND state = 'processing' AND claimed_by = ?
                """,
                (now, now, result_json, job_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise QueueStateError(
                    "Le travail n'est plus en traitement pour ce worker"
                )
            return self.get(job_id)  # type: ignore[return-value]

    def release(
        self,
        job_ids: list[str],
        *,
        worker_id: str,
        retry_delay_seconds: float = 0.0,
    ) -> int:
        """Release claimed-but-unattempted jobs without spending an attempt."""

        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds ne peut pas etre negatif")
        clean_ids = [str(job_id).strip() for job_id in job_ids if str(job_id).strip()]
        if not clean_ids:
            return 0
        now = _utc_timestamp()
        with self._lock:
            self._ensure_open()
            placeholders = ",".join("?" for _ in clean_ids)
            cursor = self._connection.execute(
                f"""
                UPDATE injection_jobs
                SET state = 'pending', available_at = ?, updated_at = ?,
                    attempts = CASE WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                    claimed_by = NULL, claimed_at = NULL
                WHERE id IN ({placeholders})
                  AND state = 'processing' AND claimed_by = ?
                """,
                (now + retry_delay_seconds, now, *clean_ids, worker_id),
            )
            return int(cursor.rowcount)

    def redeliver(self, job_id: str) -> dict[str, Any]:
        """Explicitly retry a terminal ticket after a new user submission."""

        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id doit etre une chaine non vide")
        now = _utc_timestamp()
        with self._lock:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                UPDATE injection_jobs
                SET state = 'pending', attempts = 0, available_at = ?,
                    updated_at = ?, claimed_at = NULL, completed_at = NULL,
                    claimed_by = NULL, last_error = NULL, result_json = NULL
                WHERE id = ? AND state IN ('completed', 'failed')
                """,
                (now, now, job_id.strip()),
            )
            if cursor.rowcount != 1:
                raise QueueStateError("Le travail n'est pas dans un etat terminal")
            row = self._connection.execute(
                "SELECT * FROM injection_jobs WHERE id = ?", (job_id.strip(),)
            ).fetchone()
            return self._row_to_dict(row, duplicate=True, retried=True)

    def purge_completed(self, job_ids: list[str]) -> int:
        """Remove completed synthetic tickets after their explicit cleanup."""

        clean_ids = list(dict.fromkeys(str(value).strip() for value in job_ids))
        if not clean_ids:
            return 0
        with self._lock:
            self._ensure_open()
            placeholders = ",".join("?" for _ in clean_ids)
            eligible = int(
                self._connection.execute(
                    f"""
                    SELECT COUNT(*) AS amount FROM injection_jobs
                    WHERE id IN ({placeholders}) AND state = 'completed'
                    """,
                    clean_ids,
                ).fetchone()["amount"]
            )
            if eligible != len(clean_ids):
                raise QueueStateError("Tous les travaux doivent etre completed")
            cursor = self._connection.execute(
                f"DELETE FROM injection_jobs WHERE id IN ({placeholders})",
                clean_ids,
            )
            return int(cursor.rowcount)

    def checkpoint(self, *, truncate: bool = False) -> None:
        if self.db_path == ":memory:":
            return
        mode = "TRUNCATE" if truncate else "PASSIVE"
        with self._lock:
            self._ensure_open()
            self._connection.execute(f"PRAGMA wal_checkpoint({mode})").fetchall()

    def fail(
        self,
        job_id: str,
        *,
        worker_id: str,
        error: BaseException | str,
        retry_delay_seconds: float = 0.0,
        permanent: bool = False,
    ) -> dict[str, Any]:
        """Retry a failed delivery, or terminally fail it at its bound."""

        self._ensure_open()
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds ne peut pas etre negatif")
        message = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        message = message[:_MAX_ERROR_CHARACTERS]
        now = _utc_timestamp()
        with self._lock:
            self._ensure_open()
            self._begin_write()
            try:
                row = self._connection.execute(
                    """
                    SELECT attempts, max_attempts FROM injection_jobs
                    WHERE id = ? AND state = 'processing' AND claimed_by = ?
                    """,
                    (job_id, worker_id),
                ).fetchone()
                if row is None:
                    raise QueueStateError(
                        "Le travail n'est plus en traitement pour ce worker"
                    )
                terminal = permanent or int(row["attempts"]) >= int(row["max_attempts"])
                state = "failed" if terminal else "pending"
                self._connection.execute(
                    """
                    UPDATE injection_jobs
                    SET state = ?, available_at = ?, updated_at = ?,
                        completed_at = ?, claimed_by = NULL, claimed_at = NULL,
                        last_error = ?
                    WHERE id = ?
                    """,
                    (
                        state,
                        now + retry_delay_seconds,
                        now,
                        now if terminal else None,
                        message,
                        job_id,
                    ),
                )
                updated = self._connection.execute(
                    "SELECT * FROM injection_jobs WHERE id = ?", (job_id,)
                ).fetchone()
                self._commit()
                return self._row_to_dict(updated)
            except BaseException:
                self._rollback()
                raise

    def stats(self) -> dict[str, Any]:
        """Return state counters and age of the oldest unfinished item."""

        self._ensure_open()
        now = _utc_timestamp()
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT state, COUNT(*) AS amount FROM injection_jobs GROUP BY state"
            ).fetchall()
            counts = {state: 0 for state in _STATES}
            counts.update({row["state"]: int(row["amount"]) for row in rows})
            oldest = self._connection.execute(
                """
                SELECT MIN(enqueued_at) AS oldest
                FROM injection_jobs WHERE state IN ('pending', 'processing')
                """
            ).fetchone()["oldest"]
            ready = int(
                self._connection.execute(
                    """
                    SELECT COUNT(*) AS amount FROM injection_jobs
                    WHERE state = 'pending' AND available_at <= ?
                    """,
                    (now,),
                ).fetchone()["amount"]
            )
            attempts = int(
                self._connection.execute(
                    "SELECT COALESCE(SUM(attempts), 0) AS amount FROM injection_jobs"
                ).fetchone()["amount"]
            )
            counter_rows = self._connection.execute(
                """
                SELECT key, value FROM injection_metadata
                WHERE key IN ('enqueue_requests', 'deduplicated_requests')
                """
            ).fetchall()
            counters = {row["key"]: int(row["value"]) for row in counter_rows}
            latest_error_row = self._connection.execute(
                """
                SELECT last_error FROM injection_jobs
                WHERE last_error IS NOT NULL AND last_error != ''
                ORDER BY updated_at DESC, sequence DESC LIMIT 1
                """
            ).fetchone()
            lag = max(0.0, now - float(oldest)) if oldest is not None else 0.0
            sizes = _sqlite_file_sizes(self.db_path)
            return {
                **counts,
                "total": sum(counts.values()),
                "enqueue_requests": counters.get("enqueue_requests", 0),
                "deduplicated_requests": counters.get(
                    "deduplicated_requests", 0
                ),
                "unfinished": counts["pending"] + counts["processing"],
                "ready_pending": ready,
                "attempts": attempts,
                "last_error": (
                    latest_error_row["last_error"]
                    if latest_error_row is not None
                    else None
                ),
                "oldest_unfinished_at": _iso_timestamp(oldest),
                "lag_seconds": round(lag, 6),
                "database": self.db_path,
                "database_size_bytes": sizes["total"],
                "database_main_size_bytes": sizes["main"],
                "database_wal_size_bytes": sizes["wal"],
                "database_shm_size_bytes": sizes["shm"],
                "schema_version": _QUEUE_SCHEMA_VERSION,
            }

    def close(self) -> None:
        """Close the queue connection. Safe to call repeatedly."""

        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "DurableInjectionQueue":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class BackgroundConsolidator:
    """Bounded background worker that drains a durable injection queue."""

    def __init__(
        self,
        queue: DurableInjectionQueue,
        writer_engine: MemoryEngine,
        *,
        batch_size: int = 32,
        poll_interval: float = 0.1,
        retry_base_delay: float = 0.25,
        retry_max_delay: float = 30.0,
        lease_seconds: float = 30.0,
        worker_id: str | None = None,
        maintenance_callback: Callable[[], Any] | None = None,
    ):
        if not isinstance(queue, DurableInjectionQueue):
            raise TypeError("queue doit etre une DurableInjectionQueue")
        if not isinstance(writer_engine, MemoryEngine):
            raise TypeError("writer_engine doit etre un MemoryEngine")
        self.queue = queue
        self.writer_engine = writer_engine
        self.batch_size = _positive_integer(
            batch_size, name="batch_size", maximum=_MAX_BATCH_SIZE
        )
        if poll_interval < 0:
            raise ValueError("poll_interval ne peut pas etre negatif")
        if retry_base_delay < 0 or retry_max_delay < 0:
            raise ValueError("les delais de retry ne peuvent pas etre negatifs")
        if retry_max_delay < retry_base_delay:
            raise ValueError("retry_max_delay doit etre >= retry_base_delay")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds doit etre strictement positif")
        self.poll_interval = float(poll_interval)
        self.retry_base_delay = float(retry_base_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.lease_seconds = float(lease_seconds)
        self.heartbeat_interval = min(5.0, self.lease_seconds / 3.0)
        self.worker_id = worker_id or f"worker-{uuid4()}"
        self.maintenance_callback = maintenance_callback
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._lifecycle_lock = threading.RLock()
        self._fatal_error: str | None = None
        self._maintenance_error: str | None = None
        self._recovery = {"recovered": 0, "failed": 0}

    def start(self, *, recover: bool = True) -> "BackgroundConsolidator":
        """Start the worker once; optionally reclaim interrupted work."""

        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_event.clear()
            self._wake_event.clear()
            self._fatal_error = None
            if recover:
                self._recovery = self.queue.recover_processing(
                    stale_after_seconds=self.lease_seconds
                )
            self._thread = threading.Thread(
                target=self._run_loop,
                name=f"memory-consolidator-{self.worker_id}",
                daemon=True,
            )
            self._thread.start()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"memory-heartbeat-{self.worker_id}",
                daemon=True,
            )
            self._heartbeat_thread.start()
            return self

    def notify(self) -> None:
        """Wake the worker after an enqueue instead of waiting for its poll."""

        self._wake_event.set()

    def run_once(self) -> int:
        """Process one bounded batch synchronously and return its size."""

        jobs = self.queue.claim(batch_size=self.batch_size, worker_id=self.worker_id)
        attempted = 0
        for index, job in enumerate(jobs):
            attempted += 1
            try:
                result = self.writer_engine.observe(
                    job["text"],
                    episode_id=job["episode_id"],
                    context=job["context"],
                    source=job["source"],
                    # Conserver la cle de la source evite aussi de dupliquer un
                    # souvenir deja ecrit avant l'activation du pipeline.
                    idempotency_key=job["idempotency_key"],
                )
                self.queue.complete(
                    job["job_id"], worker_id=self.worker_id, result=result
                )
            except Exception as error:
                attempt_index = max(0, int(job["attempts"]) - 1)
                delay = min(
                    self.retry_max_delay,
                    self.retry_base_delay * (2**attempt_index),
                )
                failed = self.queue.fail(
                    job["job_id"],
                    worker_id=self.worker_id,
                    error=error,
                    retry_delay_seconds=delay,
                    permanent=isinstance(error, MemoryIdempotencyConflictError),
                )
                # Les travaux reclames mais pas encore executes sont remis en
                # file. Si l'erreur sera retentee, ils attendent derriere elle
                # afin de ne jamais inverser l'ordre temporel.
                remaining = [item["job_id"] for item in jobs[index + 1 :]]
                if remaining:
                    self.queue.release(
                        remaining,
                        worker_id=self.worker_id,
                        retry_delay_seconds=(
                            delay if failed["state"] == "pending" else 0.0
                        ),
                    )
                break
        return attempted

    def _run_loop(self) -> None:
        try:
            next_recovery = time.monotonic() + self.heartbeat_interval
            next_maintenance = 0.0
            while not self._stop_event.is_set():
                processed = self.run_once()
                now = time.monotonic()
                if self.maintenance_callback is not None and (
                    processed or now >= next_maintenance
                ):
                    try:
                        self.maintenance_callback()
                        self._maintenance_error = None
                    except Exception as error:
                        # Une panne de nettoyage ne doit jamais arreter les
                        # lectures ni la consolidation. Le registre durable
                        # permettra une nouvelle tentative au prochain passage.
                        self._maintenance_error = f"{type(error).__name__}: {error}"
                    next_maintenance = now + max(0.5, self.poll_interval)
                if processed:
                    continue
                if time.monotonic() >= next_recovery:
                    recovered = self.queue.recover_processing(
                        stale_after_seconds=self.lease_seconds
                    )
                    next_recovery = time.monotonic() + self.heartbeat_interval
                    if recovered["recovered"]:
                        continue
                self._wake_event.wait(self.poll_interval)
                self._wake_event.clear()
        except BaseException as error:
            self._fatal_error = f"{type(error).__name__}: {error}"
            self._stop_event.set()

    def _heartbeat_loop(self) -> None:
        try:
            while not self._stop_event.wait(self.heartbeat_interval):
                self.queue.heartbeat(self.worker_id)
        except BaseException as error:
            self._fatal_error = f"{type(error).__name__}: {error}"
            self._stop_event.set()
            self._wake_event.set()

    def stats(self) -> dict[str, Any]:
        thread = self._thread
        heartbeat = self._heartbeat_thread
        return {
            "worker_id": self.worker_id,
            "running": bool(thread is not None and thread.is_alive()),
            "stop_requested": self._stop_event.is_set(),
            "batch_size": self.batch_size,
            "lease_seconds": self.lease_seconds,
            "heartbeat_running": bool(
                heartbeat is not None and heartbeat.is_alive()
            ),
            "fatal_error": self._fatal_error,
            "maintenance_error": self._maintenance_error,
            "startup_recovery": dict(self._recovery),
        }

    def close(self, *, timeout: float = 10.0) -> None:
        """Request a clean stop and wait for the current bounded batch."""

        if timeout < 0:
            raise ValueError("timeout ne peut pas etre negatif")
        with self._lifecycle_lock:
            self._stop_event.set()
            self._wake_event.set()
            thread = self._thread
            heartbeat = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("Le consolidateur ne s'est pas arrete a temps")
        if heartbeat is not None and heartbeat is not threading.current_thread():
            heartbeat.join(min(timeout, self.heartbeat_interval + 1.0))

    stop = close

    def __enter__(self) -> "BackgroundConsolidator":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class MemoryPipeline:
    """Ready-to-use assembly with separate read and write engines."""

    def __init__(
        self,
        memory_db_path: str | Path,
        queue_db_path: str | Path,
        *,
        batch_size: int = 32,
        poll_interval: float = 0.1,
        retry_base_delay: float = 0.25,
        retry_max_delay: float = 30.0,
        lease_seconds: float = 30.0,
        autostart: bool = True,
    ):
        if str(memory_db_path) == ":memory:":
            raise ValueError(
                "MemoryPipeline exige un fichier SQLite pour partager lecteur et ecrivain"
            )
        self.queue = DurableInjectionQueue(queue_db_path)
        try:
            # Le commit memoire doit etre aussi durable que l'acquittement de
            # la file; sinon un ticket completed pourrait pointer vers un
            # commit perdu lors d'une panne electrique.
            self.writer_engine = MemoryEngine(memory_db_path, synchronous="FULL")
            self.queue.bind_memory(self.writer_engine.database_id())
        except BaseException:
            if hasattr(self, "writer_engine"):
                self.writer_engine.close()
            self.queue.close()
            raise
        try:
            self.reader_engine = MemoryEngine(memory_db_path)
        except BaseException:
            self.writer_engine.close()
            self.queue.close()
            raise
        self._maintenance_lock = threading.RLock()
        self._maintenance_owner = f"cleanup-{uuid4()}"
        try:
            self.worker = BackgroundConsolidator(
                self.queue,
                self.writer_engine,
                batch_size=batch_size,
                poll_interval=poll_interval,
                retry_base_delay=retry_base_delay,
                retry_max_delay=retry_max_delay,
                lease_seconds=lease_seconds,
                maintenance_callback=self.cleanup_terminal_test_runs,
            )
        except BaseException:
            self.reader_engine.close()
            self.writer_engine.close()
            self.queue.close()
            raise
        self._closed = False
        if autostart:
            try:
                self.worker.start()
            except BaseException:
                self.reader_engine.close()
                self.writer_engine.close()
                self.queue.close()
                self._closed = True
                raise

    def start(self) -> "MemoryPipeline":
        if self._closed:
            raise RuntimeError("Le pipeline est ferme")
        self.worker.start()
        return self

    def enqueue(self, text: str, **kwargs: Any) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Le pipeline est ferme")
        retry_terminal = bool(kwargs.pop("retry_terminal", False))
        result = self.queue.enqueue(text, **kwargs)
        if result.get("duplicate") and retry_terminal:
            should_redeliver = result.get("state") == "failed"
            if result.get("state") == "completed":
                memory_result = result.get("result")
                event_id = (
                    memory_result.get("event_id")
                    if isinstance(memory_result, dict)
                    else None
                )
                should_redeliver = not (
                    isinstance(event_id, str)
                    and self.reader_engine.event_exists(event_id)
                )
            if should_redeliver:
                result = self.queue.redeliver(result["job_id"])
        self.worker.notify()
        return result

    def enqueue_test_run(self, run_id: str, *, count: int) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Le pipeline est ferme")
        result = self.queue.enqueue_test_run(run_id, count=count)
        self.worker.notify()
        return result

    def _cleanup_claimed_test_run(self, run: dict[str, Any]) -> dict[str, Any]:
        """Forget synthetic events durably before deleting their tickets."""

        source_keys: list[str] = []
        for job in run.get("jobs", []):
            source_key = job.get("idempotency_key")
            if not isinstance(source_key, str) or not source_key:
                raise QueueStateError("Un ticket de test a perdu sa cle source")
            source_keys.append(source_key)

        try:
            # writer_engine is configured with synchronous=FULL.  Each forget
            # is therefore committed durably before the FULL queue records the
            # purge.  Repeating this sequence after a crash is safe because
            # forgetting an already absent event is idempotent.
            # Search by source key for *every* ticket, including failed ones:
            # observe may have committed just before queue.complete crashed.
            forgotten_count = 0
            for source_key in source_keys:
                result = self.writer_engine.forget_by_idempotency_key(source_key)
                forgotten_count += int(bool(result.get("forgotten")))
            leaked_keys = [
                source_key
                for source_key in source_keys
                if self.writer_engine.event_id_for_idempotency_key(source_key)
                is not None
            ]
            if leaked_keys:
                raise RuntimeError("Des evenements synthetiques sont encore presents")
            self.writer_engine.checkpoint(truncate=True)
            cleaned = self.queue.finish_test_run_cleanup(
                run["run_id"],
                forgotten_count=forgotten_count,
                owner=self._maintenance_owner,
            )
            self.queue.checkpoint(truncate=True)
            return cleaned
        except BaseException as error:
            self.queue.mark_test_run_cleanup_failed(
                run["run_id"], error, owner=self._maintenance_owner
            )
            raise

    def cleanup_terminal_test_runs(self, *, limit: int = 10) -> list[dict[str, Any]]:
        """Automatically resume and clean fully terminal synthetic runs."""

        if self._closed:
            return []
        limit = _positive_integer(limit, name="limit", maximum=100)
        cleaned: list[dict[str, Any]] = []
        with self._maintenance_lock:
            for _ in range(limit):
                run = self.queue.claim_test_run_for_cleanup(
                    # A durable owner/lease prevents another process from
                    # cleaning the same run; an expired lease is recoverable.
                    owner=self._maintenance_owner,
                    stale_after_seconds=max(1.0, self.worker.lease_seconds),
                    retry_after_seconds=max(1.0, self.worker.poll_interval),
                )
                if run is None:
                    break
                try:
                    cleaned.append(self._cleanup_claimed_test_run(run))
                except Exception:
                    # The durable cleanup_failed state is enough for the next
                    # maintenance pass. Avoid a tight retry loop on one fault.
                    raise
        return cleaned

    def cleanup_test_run(self, run_id: str) -> dict[str, Any]:
        """Clean one run now, or return its already-cleaned durable status."""

        if self._closed:
            raise RuntimeError("Le pipeline est ferme")
        with self._maintenance_lock:
            current = self.queue.get_test_run(run_id)
            if current is None:
                raise QueueStateError("Run de test inconnu")
            if current["state"] == "cleaned":
                return current
            claimed = self.queue.claim_test_run_for_cleanup(
                run_id,
                owner=self._maintenance_owner,
                stale_after_seconds=max(1.0, self.worker.lease_seconds),
                retry_after_seconds=0.0,
            )
            if claimed is None:
                raise QueueStateError("Le test n'est pas encore entierement termine")
            return self._cleanup_claimed_test_run(claimed)

    def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.reader_engine.recall(query, top_k=top_k)

    def predict(self, history: Any, top_k: int = 5) -> list[dict[str, Any]]:
        return self.reader_engine.predict(history, top_k=top_k)

    def wait_until_idle(self, *, timeout: float = 10.0) -> bool:
        """Wait until no pending/processing work remains (failed is terminal)."""

        if timeout < 0:
            raise ValueError("timeout ne peut pas etre negatif")
        deadline = time.monotonic() + timeout
        while True:
            if self.queue.stats()["unfinished"] == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            self.worker.notify()
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def stats(self) -> dict[str, Any]:
        return {
            "queue": self.queue.stats(),
            "worker": self.worker.stats(),
            "memory": self.reader_engine.stats(),
            "reader_writer_separated": self.reader_engine is not self.writer_engine,
        }

    def close(self, *, timeout: float = 10.0) -> None:
        """Stop writes before closing both memory connections and the queue."""

        if self._closed:
            return
        self.worker.close(timeout=timeout)
        self.reader_engine.close()
        self.writer_engine.close()
        self.queue.close()
        self._closed = True

    def __enter__(self) -> "MemoryPipeline":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "BackgroundConsolidator",
    "DurableInjectionQueue",
    "IdempotencyConflictError",
    "InjectionQueueError",
    "MemoryPipeline",
    "QueueStateError",
]
