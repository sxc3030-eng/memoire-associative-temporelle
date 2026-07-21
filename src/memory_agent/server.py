"""Petit serveur HTTP local pour dialoguer avec :class:`MemoryEngine`.

Le module n'utilise que la bibliotheque standard. Il peut etre lance avec::

    python -m memory_agent.server --port 8765 --db data/memory.sqlite3

L'interface n'est volontairement pas un agent generatif : elle classe des
souvenirs et des continuations deja observes par le moteur.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import logging
import mimetypes
import os
from queue import Empty, Full, Queue
import re
import socket
import subprocess
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
from memory_agent.history_stress_lab import (
    HistoryStressConfig,
    history_stress_catalog,
    run_history_stress,
)
from memory_agent.math_engine import MathEngine, MathEngineError, MathLimits
from memory_agent.matlm_bridge import MATLMBridgeError, recall_native_capsule
from memory_agent.matlm_protocol import is_interactive_ready_frame
from memory_agent.memory import MemoryEngine, MemoryIdempotencyConflictError
from memory_agent.memory_hub import MemoryHub, SpacePolicy
from memory_agent.native_llm_contract import (
    ContractValidationError,
    MAX_CAPSULE_BYTES,
    validate_answer,
    validate_capsule,
)
from memory_agent.pipeline import (
    IdempotencyConflictError,
    MemoryPipeline,
    QueueStateError,
)
from memory_agent.science_curriculum import (
    ScienceCurriculumError,
    import_science_reference,
    load_science_dataset,
)

LOGGER = logging.getLogger("memory_agent.server")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_ROOT = PROJECT_ROOT / "web"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "memory.sqlite3"
DEFAULT_SCIENCE_DATASET = PROJECT_ROOT / "examples" / "science-biographies-v1.json"
DEFAULT_MATLM_ROOT = Path(r"D:\MAT-LM")
DEFAULT_MATLM_PYTHON = DEFAULT_MATLM_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_MATLM_MODEL = DEFAULT_MATLM_ROOT / "models" / "granite-3.3-2b-instruct"
DEFAULT_MATLM_ADAPTER = DEFAULT_MATLM_ROOT / "adapter"
DEFAULT_MATLM_SCRIPT = PROJECT_ROOT / "scripts" / "ask_matlm.py"

MAX_BODY_BYTES = 3 * 1024 * 1024
MAX_CHAT_BODY_BYTES = 16 * 1024
MAX_MATLM_BODY_BYTES = 16 * 1024
MAX_HISTORY_STRESS_BODY_BYTES = 4 * 1024
MAX_MESSAGE_CHARS = 4_000
MAX_MATLM_QUESTION_CHARS = 4_000
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


class MATLMWorkerError(RuntimeError):
    """Le lecteur MAT-LM local n'a pas pu produire une reponse sure."""


class MATLMUnavailableError(MATLMWorkerError):
    """Le worker n'est pas configure ou n'est pas demarre."""


class MATLMBusyError(MATLMWorkerError):
    """Une seule generation MAT-LM peut etre active a la fois."""


class MATLMTimeoutError(MATLMWorkerError):
    """La generation a depasse la borne configuree."""


@dataclass(frozen=True, slots=True)
class MATLMWorkerConfig:
    """Configuration strictement locale du processus persistant MAT-LM."""

    enabled: bool = False
    python_path: Path = DEFAULT_MATLM_PYTHON
    base_model_path: Path = DEFAULT_MATLM_MODEL
    adapter_path: Path = DEFAULT_MATLM_ADAPTER
    script_path: Path = DEFAULT_MATLM_SCRIPT
    load_mode: str = "auto"
    device_index: int = 0
    max_input_tokens: int = 4096
    max_new_tokens: int = 768
    request_timeout_seconds: float = 180.0
    stop_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.load_mode not in {"auto", "qlora-nf4", "bf16"}:
            raise ValueError("mode MAT-LM inconnu")
        if isinstance(self.device_index, bool) or not 0 <= self.device_index <= 15:
            raise ValueError("device_index MAT-LM doit etre compris entre 0 et 15")
        if not 256 <= self.max_input_tokens <= 131_072:
            raise ValueError("max_input_tokens MAT-LM hors limites")
        if not 16 <= self.max_new_tokens <= 8_192:
            raise ValueError("max_new_tokens MAT-LM hors limites")
        if not 5.0 <= float(self.request_timeout_seconds) <= 600.0:
            raise ValueError("timeout MAT-LM doit etre compris entre 5 et 600 secondes")
        if not 0.5 <= float(self.stop_timeout_seconds) <= 30.0:
            raise ValueError("timeout d'arret MAT-LM hors limites")


class MATLMWorker:
    """Garde un unique ``ask_matlm.py --interactive`` vivant entre les appels.

    Le worker ne transmet jamais ``--allow-model-download``. L'environnement
    Transformers est force hors ligne et la sortie est revalidee contre la
    capsule envoyee avant de franchir la route HTTP.
    """

    def __init__(self, config: MATLMWorkerConfig | None = None) -> None:
        self.config = config or MATLMWorkerConfig()
        self._state_lock = threading.RLock()
        self._request_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._responses: Queue[tuple[str, Any]] = Queue(maxsize=4)
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=12)
        self._generation = 0
        self._ready_generation: int | None = None
        self._state = "stopped" if self.config.enabled else "disabled"
        self._last_error: str | None = None
        self._started_at: float | None = None
        self._completed_requests = 0

    @staticmethod
    def _clean_error(value: Any, maximum: int = 300) -> str:
        return " ".join(str(value).split())[:maximum]

    def _configuration_issues(self) -> list[str]:
        if not self.config.enabled:
            return []
        checks = (
            (self.config.python_path, "interpreteur Python MAT-LM absent", True),
            (self.config.script_path, "scripts/ask_matlm.py absent", True),
            (self.config.base_model_path, "modele Granite local absent", False),
            (self.config.adapter_path, "adaptateur PEFT local absent", False),
        )
        issues: list[str] = []
        for raw_path, message, must_be_file in checks:
            path = Path(raw_path).expanduser()
            exists = path.is_file() if must_be_file else path.is_dir()
            if not exists:
                issues.append(message)
        return issues

    def _is_running_locked(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            running = self._is_running_locked()
            if self._process is not None and not running and self._state not in {
                "disabled",
                "stopped",
                "error",
            }:
                self._state = "error"
                self._last_error = "Le processus MAT-LM s'est arrete de facon inattendue."
            issues = self._configuration_issues()
            return {
                "enabled": self.config.enabled,
                "configured": self.config.enabled and not issues,
                "state": self._state,
                "running": running,
                "model_loaded": running and self._ready_generation == self._generation,
                "offline_only": True,
                "persistent_process": True,
                "one_model_at_a_time": True,
                "automatic_learning": False,
                "completed_requests": self._completed_requests,
                "uptime_seconds": (
                    round(time.monotonic() - self._started_at, 3)
                    if running and self._started_at is not None
                    else None
                ),
                "configuration_issues": issues,
                "last_error": self._last_error,
                "limits": {
                    "question_characters": MAX_MATLM_QUESTION_CHARS,
                    "capsule_bytes": MAX_CAPSULE_BYTES,
                    "request_timeout_seconds": self.config.request_timeout_seconds,
                },
                "runtime": {
                    "load_mode": self.config.load_mode,
                    "device": f"xpu:{self.config.device_index}",
                    "base_model": Path(self.config.base_model_path).name,
                    "adapter": Path(self.config.adapter_path).name,
                    "max_input_tokens": self.config.max_input_tokens,
                    "max_new_tokens": self.config.max_new_tokens,
                },
            }

    def _command(self) -> list[str]:
        return [
            str(Path(self.config.python_path).expanduser()),
            "-u",
            str(Path(self.config.script_path).expanduser()),
            "--adapter",
            str(Path(self.config.adapter_path).expanduser()),
            "--base-model",
            str(Path(self.config.base_model_path).expanduser()),
            "--load-mode",
            self.config.load_mode,
            "--device-index",
            str(self.config.device_index),
            "--max-input-tokens",
            str(self.config.max_input_tokens),
            "--max-new-tokens",
            str(self.config.max_new_tokens),
            "--interactive",
        ]

    def start(self) -> dict[str, Any]:
        with self._state_lock:
            if not self.config.enabled:
                raise MATLMUnavailableError(
                    "MAT-LM est desactive; redemarrez le serveur avec --enable-matlm."
                )
            if self._is_running_locked():
                return self.status()
            issues = self._configuration_issues()
            if issues:
                self._state = "error"
                self._last_error = "; ".join(issues)
                raise MATLMUnavailableError(self._last_error)

            environment = os.environ.copy()
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            environment["PYTHONUNBUFFERED"] = "1"
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                process = subprocess.Popen(
                    self._command(),
                    cwd=str(PROJECT_ROOT),
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    shell=False,
                    creationflags=creation_flags,
                )
            except OSError as error:
                self._state = "error"
                self._last_error = "Impossible de demarrer le processus MAT-LM local."
                raise MATLMUnavailableError(self._last_error) from error

            self._process = process
            self._responses = Queue(maxsize=4)
            self._stderr_tail.clear()
            self._generation += 1
            generation = self._generation
            self._ready_generation = None
            self._state = "starting"
            self._last_error = None
            self._started_at = time.monotonic()
            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                args=(process, generation, self._responses),
                name="matlm-stdout",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                args=(process, generation),
                name="matlm-stderr",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()
            return self.status()

    def _read_stdout(
        self,
        process: subprocess.Popen[str],
        generation: int,
        responses: Queue[tuple[str, Any]],
    ) -> None:
        stream = process.stdout
        try:
            if stream is not None:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        control = json.loads(line)
                    except (TypeError, json.JSONDecodeError):
                        control = None
                    if is_interactive_ready_frame(control):
                        with self._state_lock:
                            if self._process is process and self._generation == generation:
                                if self._state in {"starting", "busy"}:
                                    self._ready_generation = generation
                                if self._state == "starting":
                                    self._state = "ready"
                                    self._last_error = None
                        # Une trame de controle ne doit jamais devenir la
                        # reponse de la prochaine question.
                        continue
                    try:
                        responses.put(("line", line), timeout=1.0)
                    except Full:
                        self._mark_failed(
                            process,
                            generation,
                            "MAT-LM a produit trop de reponses inattendues.",
                        )
                        return
        finally:
            try:
                responses.put_nowait(("eof", process.poll()))
            except Full:
                pass
            with self._state_lock:
                if (
                    self._process is process
                    and self._generation == generation
                    and self._state not in {"disabled", "stopped", "error"}
                ):
                    self._state = "error"
                    self._last_error = "Le processus MAT-LM s'est arrete de facon inattendue."

    def _read_stderr(self, process: subprocess.Popen[str], generation: int) -> None:
        stream = process.stderr
        if stream is None:
            return
        for line in stream:
            clean = self._clean_error(line, 500)
            if not clean:
                continue
            with self._state_lock:
                if self._process is process and self._generation == generation:
                    self._stderr_tail.append(clean)

    def _mark_failed(
        self,
        process: subprocess.Popen[str],
        generation: int,
        message: str,
    ) -> None:
        with self._state_lock:
            if self._process is process and self._generation == generation:
                self._state = "error"
                self._last_error = message
        if process.poll() is None:
            process.terminate()

    def _terminate(self, *, error_message: str | None = None) -> None:
        with self._state_lock:
            process = self._process
            stdout_thread = self._stdout_thread
            stderr_thread = self._stderr_thread
            if process is None:
                self._state = "error" if error_message else (
                    "stopped" if self.config.enabled else "disabled"
                )
                self._last_error = error_message
                self._started_at = None
                self._ready_generation = None
                return
            self._state = "error" if error_message else "stopping"
            self._last_error = error_message

        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            # La fermeture de stdin laisse ``ask_matlm.py`` sortir de sa boucle
            # et appeler MATLMInferenceSession.close(), qui libere le modele et
            # le cache XPU. La terminaison forcee reste bornee en secours.
            process.wait(timeout=self.config.stop_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=self.config.stop_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.config.stop_timeout_seconds)

        current = threading.current_thread()
        for thread in (stdout_thread, stderr_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=1.0)
        for stream in (process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

        with self._state_lock:
            if self._process is process:
                self._process = None
                self._stdout_thread = None
                self._stderr_thread = None
                self._started_at = None
                self._ready_generation = None
                self._state = "error" if error_message else (
                    "stopped" if self.config.enabled else "disabled"
                )
                self._last_error = error_message

    def stop(self) -> dict[str, Any]:
        self._terminate()
        return self.status()

    def ask(self, capsule: dict[str, Any]) -> dict[str, Any]:
        trusted_capsule = validate_capsule(capsule)
        encoded = json.dumps(
            trusted_capsule,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded.encode("utf-8")) > MAX_CAPSULE_BYTES:
            raise MATLMWorkerError("La capsule MAT-LM depasse la limite autorisee.")
        if not self._request_lock.acquire(blocking=False):
            raise MATLMBusyError("MAT-LM traite deja une autre question.")
        try:
            with self._state_lock:
                process = self._process
                if process is None or process.poll() is not None:
                    raise MATLMUnavailableError(
                        "MAT-LM n'est pas demarre. Utilisez d'abord le bouton Demarrer."
                    )
                if process.stdin is None:
                    raise MATLMUnavailableError("Entree du processus MAT-LM indisponible.")
                responses = self._responses
                self._state = "busy"
            try:
                process.stdin.write(encoded + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                message = "Le processus MAT-LM ne repond plus."
                self._terminate(error_message=message)
                raise MATLMUnavailableError(message) from error

            try:
                kind, value = responses.get(timeout=self.config.request_timeout_seconds)
            except Empty as error:
                message = "MAT-LM a depasse le delai de reponse et a ete arrete."
                self._terminate(error_message=message)
                raise MATLMTimeoutError(message) from error
            if kind == "eof":
                message = "Le processus MAT-LM s'est arrete avant de repondre."
                self._terminate(error_message=message)
                raise MATLMUnavailableError(message)

            try:
                raw = json.loads(value)
            except (TypeError, json.JSONDecodeError) as error:
                message = "MAT-LM a produit une sortie qui n'est pas un objet JSON valide."
                self._terminate(error_message=message)
                raise MATLMWorkerError(message) from error
            if isinstance(raw, dict) and raw.get("ok") is False:
                child_error = self._clean_error(raw.get("error") or "capsule refusee")
                message = f"MAT-LM a refuse la requete: {child_error}"
                self._terminate(error_message=message)
                raise MATLMWorkerError(message)
            try:
                answer = validate_answer(value, trusted_capsule)
            except ContractValidationError as error:
                message = "MAT-LM a produit une reponse hors contrat et a ete arrete."
                self._terminate(error_message=message)
                raise MATLMWorkerError(message) from error
            with self._state_lock:
                if self._process is process and process.poll() is None:
                    self._ready_generation = self._generation
                    self._state = "ready"
                    self._last_error = None
                    self._completed_requests += 1
            return answer
        finally:
            self._request_lock.release()


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
                "space": memory.get("space", "personal"),
                "space_policy": memory.get("space_policy"),
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
        reference_engine: MemoryEngine | None = None,
        science_dataset_path: Path | None = None,
        matlm_worker: MATLMWorker | None = None,
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
        self.reference_engine = reference_engine
        self.reference_lock = threading.RLock()
        self.matlm_worker = matlm_worker or MATLMWorker()
        self.science_dataset_path = (
            science_dataset_path.expanduser().resolve()
            if science_dataset_path is not None
            else None
        )
        self.science_import_result: dict[str, Any] | None = None
        self.science_questions: list[dict[str, str]] = []
        self.memory_hub: MemoryHub | None = None
        try:
            if reference_engine is not None:
                self.memory_hub = MemoryHub(
                    {
                        "personal": engine,
                        "science-reference": reference_engine,
                    },
                    {
                        "personal": SpacePolicy.private("local-agent"),
                        "science-reference": SpacePolicy.reference(),
                    },
                )
                if self.science_dataset_path is not None:
                    self.reload_science_reference()
            elif self.science_dataset_path is not None:
                raise ValueError(
                    "science_dataset_path exige une memoire de reference separee"
                )
        except Exception:
            # A constructor failure must not leave either a listening socket or
            # the dedicated SQLite connection open.
            super().server_close()
            if reference_engine is not None:
                reference_engine.close()
            self.matlm_worker.stop()
            raise
        self.matlm_hub = self.memory_hub or MemoryHub(
            {"personal": engine},
            {"personal": SpacePolicy.private("local-agent")},
        )
        self.matlm_space_names = (
            ["personal", "science-reference"]
            if reference_engine is not None
            else ["personal"]
        )
        # Le laboratoire est synchrone pour le MVP. Ce verrou distinct garantit
        # qu'un seul corpus temporaire est actif sans bloquer les autres routes.
        self.history_stress_lock = threading.Lock()
        self.web_root = web_root.resolve()
        self.started_at = time.monotonic()
        # Une execution locale represente une conversation. Tous ses messages
        # confirmes appartiennent au meme episode et apprennent donc aussi les
        # transitions entre deux appels distincts a ``observe``.
        self.conversation_episode_id = str(uuid4())
        self.conversation_episode_events = 0
        self.conversation_lock = threading.Lock()

    def reload_science_reference(self) -> dict[str, Any]:
        """Validate and idempotently load claims into the read-only space."""

        if self.reference_engine is None or self.science_dataset_path is None:
            raise RuntimeError("Memoire scientifique de reference non configuree")
        with self.reference_lock:
            result = import_science_reference(
                self.science_dataset_path,
                self.reference_engine,
            )
            dataset = load_science_dataset(self.science_dataset_path)
            # Only public prompts cross this API boundary. The correction key
            # and supporting claim identifiers remain in the benchmark file.
            self.science_questions = [
                {"id": row["id"], "question": row["question"]}
                for row in dataset["evaluation_questions"]
            ]
            self.science_import_result = result
            return dict(result)

    def science_reference_status(self) -> dict[str, Any]:
        if self.reference_engine is None:
            return {
                "enabled": False,
                "ready": False,
                "policy": "reference",
                "storage": "separate_sqlite",
            }
        with self.reference_lock:
            stats = self.reference_engine.stats()
            imported = dict(self.science_import_result or {})
        return {
            "enabled": True,
            "ready": bool(imported),
            "policy": "reference",
            "storage": "separate_sqlite",
            "dataset_schema_version": imported.get("dataset_schema_version"),
            "claims": imported.get("claims_imported", 0),
            "dossiers": imported.get("dossiers", 0),
            "questions_available": len(self.science_questions),
            "stats": stats,
        }

    def recall_memories(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Recall from personal and scientific spaces without joining storage."""

        if self.memory_hub is None:
            with self.engine_lock:
                return self.engine.recall(query, top_k=top_k)
        with self.engine_lock, self.reference_lock:
            capsule = self.memory_hub.recall_capsule(
                "local-agent",
                query,
                space_names=["personal", "science-reference"],
                top_k=top_k,
                character_budget=1_000_000,
            )
        return list(capsule["items"][:top_k])

    def matlm_capsule(self, question: str, *, request_id: str) -> dict[str, Any]:
        """Rappelle les espaces autorises et construit une capsule en lecture seule."""

        options = {
            "request_id": request_id,
            "space_names": self.matlm_space_names,
            "top_k": 12,
            "hub_character_budget": 48_000,
            "character_budget": 32_768,
            "max_evidence_items": 12,
            "max_evidence_text_characters": 2_000,
            "max_answer_characters": 4_000,
            "max_calculations": 4,
        }
        if self.reference_engine is None:
            with self.engine_lock:
                return recall_native_capsule(
                    self.matlm_hub,
                    "local-agent",
                    question,
                    **options,
                )
        with self.engine_lock, self.reference_lock:
            return recall_native_capsule(
                self.matlm_hub,
                "local-agent",
                question,
                **options,
            )

    def server_close(self) -> None:
        """Close the listener, MAT-LM and the server-owned reference connection."""

        try:
            self.matlm_worker.stop()
        finally:
            try:
                super().server_close()
            finally:
                if self.reference_engine is not None:
                    self.reference_engine.close()

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
            # Le corps n'est volontairement pas lu. Fermer cette connexion
            # après la réponse empêche ses octets restants d'être interprétés
            # comme une nouvelle requête HTTP/1.1.
            self.close_connection = True
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

        if path == "/api/matlm/status":
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "matlm": self.server.matlm_worker.status()},
                head_only=head_only,
            )
            return

        if path == "/api/science/reference":
            try:
                reference = self.server.science_reference_status()
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "reference": reference},
                    head_only=head_only,
                )
            except Exception:
                LOGGER.exception("Impossible de lire la memoire scientifique")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "Memoire scientifique indisponible."},
                    head_only=head_only,
                )
            return

        if path == "/api/science/questions":
            questions = [dict(row) for row in self.server.science_questions]
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "enabled": self.server.reference_engine is not None,
                    "questions": questions,
                    "count": len(questions),
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

        if path == "/api/stress/history/catalog":
            try:
                library_catalog = history_stress_catalog()
                if not isinstance(library_catalog, dict):
                    raise TypeError("Le catalogue historique doit etre un objet")
                catalog = dict(library_catalog)
                library_bounds = library_catalog.get("config_bounds", {})
                if not isinstance(library_bounds, dict):
                    raise TypeError("Les limites historiques doivent etre un objet")
                http_bounds = dict(library_bounds)
                http_bounds["event_count"] = {"minimum": 5, "maximum": 100}
                catalog["config_bounds"] = http_bounds
                catalog["library_config_bounds"] = library_bounds
                catalog["interface"] = "http_local"
                _json_bytes(catalog)
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "catalog": catalog},
                    head_only=head_only,
                )
            except Exception:
                LOGGER.exception("Impossible de lire le catalogue historique")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "Catalogue historique indisponible."},
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
        mime_overrides = {
            ".ico": "image/x-icon",
            ".webp": "image/webp",
        }
        mime = (
            mime_overrides.get(candidate.suffix.lower())
            or mimetypes.guess_type(candidate.name)[0]
            or "application/octet-stream"
        )
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
            "/api/matlm/start",
            "/api/matlm/stop",
            "/api/matlm/ask",
            "/api/math/catalog/import",
            "/api/science/reference/import",
            "/api/stress/history/run",
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
                MAX_MATLM_BODY_BYTES
                if path.startswith("/api/matlm/")
                else MAX_CHAT_BODY_BYTES
                if path in {"/api/chat", "/api/calculate"}
                else MAX_HISTORY_STRESS_BODY_BYTES
                if path == "/api/stress/history/run"
                else MAX_BODY_BYTES
            )
        )
        if payload is None:
            return
        if path == "/api/matlm/start":
            self._handle_matlm_start(payload)
            return
        if path == "/api/matlm/stop":
            self._handle_matlm_stop(payload)
            return
        if path == "/api/matlm/ask":
            self._handle_matlm_ask(payload)
            return
        if path == "/api/stress/history/run":
            self._handle_history_stress_run(payload)
            return
        if path == "/api/calculate":
            self._handle_calculate(payload)
            return
        if path == "/api/math/catalog/import":
            self._handle_math_catalog_import(payload)
            return
        if path == "/api/science/reference/import":
            self._handle_science_reference_import(payload)
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

    def _handle_matlm_start(self, payload: dict[str, Any]) -> None:
        if payload:
            self._error(HTTPStatus.BAD_REQUEST, "Le demarrage MAT-LM n'accepte aucun parametre.")
            return
        try:
            status = self.server.matlm_worker.start()
        except MATLMUnavailableError as error:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, _clean_text(error, 300))
            return
        self._send_json(
            HTTPStatus.ACCEPTED if status.get("state") == "starting" else HTTPStatus.OK,
            {"ok": True, "matlm": status},
        )

    def _handle_matlm_stop(self, payload: dict[str, Any]) -> None:
        if payload:
            self._error(HTTPStatus.BAD_REQUEST, "L'arret MAT-LM n'accepte aucun parametre.")
            return
        try:
            status = self.server.matlm_worker.stop()
        except Exception:
            LOGGER.exception("Impossible d'arreter MAT-LM proprement")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Arret MAT-LM incomplet.")
            return
        self._send_json(HTTPStatus.OK, {"ok": True, "matlm": status})

    def _handle_matlm_ask(self, payload: dict[str, Any]) -> None:
        allowed_fields = {"question", "request_id"}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "Parametre MAT-LM inconnu: " + ", ".join(unknown_fields),
            )
            return
        question = payload.get("question")
        if not isinstance(question, str):
            self._error(HTTPStatus.BAD_REQUEST, "question doit etre une chaine.")
            return
        question = unicodedata.normalize("NFC", question).strip()
        if not question:
            self._error(HTTPStatus.BAD_REQUEST, "La question MAT-LM ne peut pas etre vide.")
            return
        if len(question) > MAX_MATLM_QUESTION_CHARS:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "La question MAT-LM est trop longue.")
            return
        request_id = payload.get("request_id")
        if request_id is None:
            request_id = f"web-matlm-{uuid4()}"
        elif (
            not isinstance(request_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", request_id.strip())
        ):
            self._error(HTTPStatus.BAD_REQUEST, "request_id MAT-LM invalide.")
            return
        else:
            request_id = request_id.strip()

        try:
            capsule = self.server.matlm_capsule(question, request_id=request_id)
            answer = validate_answer(self.server.matlm_worker.ask(capsule), capsule)
        except MATLMBusyError as error:
            self._error(HTTPStatus.CONFLICT, _clean_text(error, 300))
            return
        except MATLMTimeoutError as error:
            self._error(HTTPStatus.GATEWAY_TIMEOUT, _clean_text(error, 300))
            return
        except MATLMUnavailableError as error:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, _clean_text(error, 300))
            return
        except MATLMWorkerError as error:
            self._error(HTTPStatus.BAD_GATEWAY, _clean_text(error, 300))
            return
        except (MATLMBridgeError, ContractValidationError, ValueError) as error:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, _clean_text(error, 300))
            return
        except Exception:
            LOGGER.exception("Erreur locale pendant une question MAT-LM")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "MAT-LM a rencontre une erreur locale.")
            return

        abstention = answer.get("abstention", {})
        evidence_by_id = {
            row["evidence_id"]: row
            for row in capsule.get("evidence", [])
            if isinstance(row, dict) and isinstance(row.get("evidence_id"), str)
        }
        citations = [
            evidence_by_id[evidence_id]
            for evidence_id in answer.get("evidence_ids", [])
            if evidence_id in evidence_by_id
        ]
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "intent": "matlm",
                "reply": answer.get("answer", ""),
                "answer": answer,
                "citations": citations,
                "details": {
                    "request_id": request_id,
                    "evidence_available": len(capsule.get("evidence", [])),
                    "evidence_used": len(answer.get("evidence_ids", [])),
                    "abstained": bool(abstention.get("abstained")),
                    "automatic_learning": False,
                },
            },
        )

    def _handle_history_stress_run(self, payload: dict[str, Any]) -> None:
        allowed_fields = {"event_count", "seed"}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "Parametre historique inconnu: " + ", ".join(unknown_fields),
            )
            return

        event_count = payload.get("event_count")
        if isinstance(event_count, bool) or not isinstance(event_count, int):
            self._error(
                HTTPStatus.BAD_REQUEST,
                "event_count doit etre un entier entre 5 et 100.",
            )
            return
        if not 5 <= event_count <= 100:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "event_count doit etre compris entre 5 et 100.",
            )
            return

        seed = payload.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            self._error(HTTPStatus.BAD_REQUEST, "seed doit etre un entier.")
            return
        if not 0 <= seed <= 2**63 - 1:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "seed doit etre compris entre 0 et 9223372036854775807.",
            )
            return

        try:
            config = HistoryStressConfig(event_count=event_count, seed=seed)
        except (TypeError, ValueError) as error:
            self._error(
                HTTPStatus.BAD_REQUEST,
                _clean_text(error, 300) or "Configuration historique invalide.",
            )
            return

        if not self.server.history_stress_lock.acquire(blocking=False):
            self._error(
                HTTPStatus.CONFLICT,
                "Un test historique est deja en cours. Reessayez apres sa fin.",
            )
            return

        try:
            report = run_history_stress(config)
            if not isinstance(report, dict):
                raise TypeError("Le laboratoire historique doit retourner un objet")
            # Valide la reponse avant de commencer a ecrire les en-tetes HTTP.
            _json_bytes(report)
        except Exception:
            LOGGER.exception("Erreur interne du laboratoire historique")
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Le laboratoire historique a rencontre une erreur interne.",
            )
            return
        finally:
            self.server.history_stress_lock.release()

        self._send_json(
            HTTPStatus.OK,
            {"ok": True, "report": report},
        )

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

    def _handle_science_reference_import(self, payload: dict[str, Any]) -> None:
        if payload:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "L'import scientifique utilise uniquement le corpus local configure.",
            )
            return
        if (
            self.server.reference_engine is None
            or self.server.science_dataset_path is None
        ):
            self._error(
                HTTPStatus.CONFLICT,
                "Memoire scientifique de reference non configuree.",
            )
            return
        try:
            result = self.server.reload_science_reference()
            reference = self.server.science_reference_status()
        except MemoryIdempotencyConflictError as error:
            self._error(HTTPStatus.CONFLICT, _clean_text(error, 400))
            return
        except ScienceCurriculumError as error:
            self._error(HTTPStatus.BAD_REQUEST, _clean_text(error, 400))
            return
        except Exception:
            LOGGER.exception("Erreur pendant l'import scientifique")
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "L'import scientifique a rencontre une erreur.",
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {"ok": True, "reference": {**reference, "import": result}},
        )

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
            result = self.server.recall_memories(query, top_k=5)
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
        result = self.server.recall_memories(message, top_k=5)
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
        "--science-reference-db",
        type=Path,
        default=None,
        help=(
            "base scientifique separee "
            "(defaut: science-reference.sqlite3 pres de --db)"
        ),
    )
    parser.add_argument(
        "--science-dataset",
        type=Path,
        default=DEFAULT_SCIENCE_DATASET,
        help=f"corpus scientifique valide (defaut: {DEFAULT_SCIENCE_DATASET})",
    )
    parser.add_argument(
        "--no-science-reference",
        action="store_true",
        help="desactive explicitement la memoire scientifique separee",
    )
    parser.add_argument(
        "--enable-matlm",
        action="store_true",
        help="active le panneau MAT-LM local; le modele reste arrete jusqu'au bouton Demarrer",
    )
    parser.add_argument(
        "--matlm-python",
        type=Path,
        default=DEFAULT_MATLM_PYTHON,
        help=f"Python de l'environnement MAT-LM (defaut: {DEFAULT_MATLM_PYTHON})",
    )
    parser.add_argument(
        "--matlm-model",
        type=Path,
        default=DEFAULT_MATLM_MODEL,
        help=f"dossier Granite local (defaut: {DEFAULT_MATLM_MODEL})",
    )
    parser.add_argument(
        "--matlm-adapter",
        type=Path,
        default=DEFAULT_MATLM_ADAPTER,
        help=f"dossier adaptateur PEFT local (defaut: {DEFAULT_MATLM_ADAPTER})",
    )
    parser.add_argument(
        "--matlm-load-mode",
        choices=("auto", "qlora-nf4", "bf16"),
        default="auto",
        help="chargement XPU MAT-LM (defaut: auto)",
    )
    parser.add_argument(
        "--matlm-device-index",
        type=lambda value: _bounded_integer(
            value, name="matlm-device-index", minimum=0, maximum=15
        ),
        default=0,
        help="index Intel XPU (defaut: 0)",
    )
    parser.add_argument(
        "--matlm-max-input-tokens",
        type=lambda value: _bounded_integer(
            value, name="matlm-max-input-tokens", minimum=256, maximum=131_072
        ),
        default=4096,
        help="borne de tokens d'entree (defaut: 4096)",
    )
    parser.add_argument(
        "--matlm-max-new-tokens",
        type=lambda value: _bounded_integer(
            value, name="matlm-max-new-tokens", minimum=16, maximum=8_192
        ),
        default=768,
        help="borne de tokens generes (defaut: 768)",
    )
    parser.add_argument(
        "--matlm-timeout-seconds",
        type=lambda value: _bounded_integer(
            value, name="matlm-timeout-seconds", minimum=5, maximum=600
        ),
        default=180,
        help="delai maximal d'une question avant arret du worker (defaut: 180)",
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
    reference_engine: MemoryEngine | None = None
    matlm_worker: MATLMWorker | None = None
    server: MemoryHTTPServer | None = None
    try:
        matlm_worker = MATLMWorker(
            MATLMWorkerConfig(
                enabled=args.enable_matlm,
                python_path=args.matlm_python,
                base_model_path=args.matlm_model,
                adapter_path=args.matlm_adapter,
                load_mode=args.matlm_load_mode,
                device_index=args.matlm_device_index,
                max_input_tokens=args.matlm_max_input_tokens,
                max_new_tokens=args.matlm_max_new_tokens,
                request_timeout_seconds=float(args.matlm_timeout_seconds),
            )
        )
        queue_path: Path | None = None
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

        science_dataset_path: Path | None = None
        if not args.no_science_reference:
            candidate_dataset = args.science_dataset.expanduser().resolve()
            if candidate_dataset.is_file():
                reference_path = (
                    args.science_reference_db.expanduser().resolve()
                    if args.science_reference_db is not None
                    else db_path.with_name("science-reference.sqlite3")
                )
                forbidden_paths = {db_path}
                if queue_path is not None:
                    forbidden_paths.add(queue_path)
                if reference_path in forbidden_paths:
                    raise ValueError(
                        "La memoire scientifique doit etre distincte de la memoire "
                        "personnelle et de la file d'injection."
                    )
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                reference_engine = MemoryEngine(reference_path)
                science_dataset_path = candidate_dataset
            else:
                LOGGER.warning(
                    "Corpus scientifique absent; espace de reference desactive: %s",
                    candidate_dataset,
                )
        server = MemoryHTTPServer(
            (args.host, args.port),
            engine,
            pipeline=pipeline,
            reference_engine=reference_engine,
            science_dataset_path=science_dataset_path,
            matlm_worker=matlm_worker,
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
        elif reference_engine is not None:
            reference_engine.close()
        if server is None and matlm_worker is not None:
            matlm_worker.stop()
        if pipeline is not None:
            pipeline.close()
        elif engine is not None:
            engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
