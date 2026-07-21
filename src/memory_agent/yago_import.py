"""Import local, incremental et borne des dumps YAGO 4.5.

Le module vise le sous-ensemble Turtle/Turtle-star effectivement publie par
YAGO 4.5. Il ne remplace pas un parseur RDF generaliste : les constructions
non necessaires aux dumps YAGO sont comptees comme non prises en charge (ou
font echouer un import ``strict``).

Les triplets restent dans une base SQLite de reference separee. Cela evite de
transformer des millions de faits externes en souvenirs personnels et permet
de reprendre un import interrompu sans dupliquer les affirmations.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import codecs
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
from typing import Any, BinaryIO, Iterator, Sequence
from urllib.parse import unquote, urljoin, urlsplit
import zipfile


PARSER_VERSION = "yago-turtle-stream-v1"
SCHEMA_VERSION = 1
YAGO_DATASET_NAME = "YAGO 4.5"
YAGO_LICENSE = "CC BY-SA 3.0"
YAGO_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"
YAGO_DOWNLOAD_URL = "https://yago-knowledge.org/downloads/yago-4-5"
YAGO_PAPER_URL = "https://doi.org/10.1145/3626772.3657876"
LINEAGE_ID = "wikidata-schemaorg--yago-4.5"

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_SCHEMA_START = "http://schema.org/startDate"
_SCHEMA_END = "http://schema.org/endDate"
_SCHEMA_STARTS = frozenset({_SCHEMA_START, "https://schema.org/startDate"})
_SCHEMA_ENDS = frozenset({_SCHEMA_END, "https://schema.org/endDate"})
_PROV_PREFIX = "http://www.w3.org/ns/prov#"
_SUPPORTED_SUFFIXES = frozenset({".ttl", ".nt", ".ntx", ".rdf", ".txt"})
_IGNORABLE_NAMES = frozenset({"readme", "readme.txt", "license", "license.txt"})
_NUMERIC_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?\Z")
_LANG_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z0-9]+)*\Z")


class YAGOImportError(ValueError):
    """Le dump est invalide, non pris en charge ou depasse une limite."""


@dataclass(frozen=True, slots=True)
class ImportLimits:
    """Bornes de securite, ajustables explicitement pour un gros essai."""

    max_archive_bytes: int = 512 * 1024 * 1024
    max_members: int = 128
    max_member_bytes: int = 4 * 1024 * 1024 * 1024
    max_total_uncompressed_bytes: int = 8 * 1024 * 1024 * 1024
    max_compression_ratio: float = 250.0
    max_statement_chars: int = 4 * 1024 * 1024
    max_statements: int = 12_000_000
    max_prefixes_per_member: int = 4_096
    max_parse_errors: int = 1_000
    max_triples: int = 10_000_000
    batch_size: int = 2_000

    def __post_init__(self) -> None:
        integer_fields = (
            "max_archive_bytes",
            "max_members",
            "max_member_bytes",
            "max_total_uncompressed_bytes",
            "max_statement_chars",
            "max_statements",
            "max_prefixes_per_member",
            "max_parse_errors",
            "max_triples",
            "batch_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} doit etre un entier positif")
        if not isinstance(self.max_compression_ratio, (int, float)):
            raise TypeError("max_compression_ratio doit etre numerique")
        if not 1.0 <= float(self.max_compression_ratio) <= 10_000.0:
            raise ValueError("max_compression_ratio doit etre compris entre 1 et 10000")
        if self.batch_size > 100_000:
            raise ValueError("batch_size ne peut pas depasser 100000")


@dataclass(frozen=True, slots=True)
class RDFTerm:
    kind: str
    value: str
    datatype: str | None = None
    language: str | None = None

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class RDFTriple:
    subject: RDFTerm
    predicate: RDFTerm
    object: RDFTerm

    @property
    def statement_id(self) -> str:
        encoded = "\0".join(
            (self.subject.canonical(), self.predicate.canonical(), self.object.canonical())
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ParsedRDF:
    subject: RDFTerm | RDFTriple
    predicate: RDFTerm
    object: RDFTerm


@dataclass(slots=True)
class _Budget:
    total_bytes: int = 0


class _CountingReader:
    def __init__(
        self,
        raw: BinaryIO,
        *,
        member_limit: int,
        total_limit: int,
        shared: _Budget,
    ) -> None:
        self.raw = raw
        self.member_limit = member_limit
        self.total_limit = total_limit
        self.shared = shared
        self.member_bytes = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        self.member_bytes += len(chunk)
        self.shared.total_bytes += len(chunk)
        if self.member_bytes > self.member_limit:
            raise YAGOImportError("Un membre depasse la taille decompressee autorisee")
        if self.shared.total_bytes > self.total_limit:
            raise YAGOImportError("Le total decompresse depasse la limite autorisee")
        return chunk


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise YAGOImportError("Nom de membre ZIP invalide")
    portable = name.replace("\\", "/")
    path = PurePosixPath(portable)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise YAGOImportError(f"Chemin ZIP dangereux: {name!r}")
    if re.match(r"^[A-Za-z]:", portable):
        raise YAGOImportError(f"Chemin ZIP absolu interdit: {name!r}")
    return path.as_posix()


def _validate_zip_members(
    archive: zipfile.ZipFile,
    limits: ImportLimits,
) -> tuple[list[zipfile.ZipInfo], int]:
    infos = archive.infolist()
    if len(infos) > limits.max_members:
        raise YAGOImportError("Le ZIP contient trop de membres")
    selected: list[zipfile.ZipInfo] = []
    declared_total = 0
    for info in infos:
        safe_name = _safe_member_name(info.filename)
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise YAGOImportError(f"Lien symbolique interdit dans le ZIP: {safe_name}")
        if info.flag_bits & 0x1:
            raise YAGOImportError(f"Membre ZIP chiffre non pris en charge: {safe_name}")
        if info.is_dir():
            continue
        declared_total += info.file_size
        if info.file_size > limits.max_member_bytes:
            raise YAGOImportError(f"Membre ZIP trop grand: {safe_name}")
        if declared_total > limits.max_total_uncompressed_bytes:
            raise YAGOImportError("Taille decompressee declaree trop grande")
        if info.file_size:
            if info.compress_size <= 0:
                raise YAGOImportError(f"Ratio ZIP invalide: {safe_name}")
            ratio = info.file_size / info.compress_size
            if ratio > float(limits.max_compression_ratio):
                raise YAGOImportError(f"Ratio de compression suspect pour {safe_name}")
        basename = Path(safe_name).name.casefold()
        suffix = Path(safe_name).suffix.casefold()
        if basename in _IGNORABLE_NAMES:
            continue
        if suffix in _SUPPORTED_SUFFIXES:
            selected.append(info)
        else:
            # Aucun fichier n'est extrait. Les extras restent simplement ignores.
            continue
    if not selected:
        raise YAGOImportError("Le ZIP ne contient aucun fichier RDF/Turtle pris en charge")
    return selected, declared_total


@contextmanager
def _source_members(
    source_path: Path,
    limits: ImportLimits,
) -> Iterator[Iterator[tuple[str, _CountingReader, int | None]]]:
    """Ouvre sans extraction les membres RDF valides."""

    shared = _Budget()
    if source_path.stat().st_size > limits.max_archive_bytes:
        raise YAGOImportError("Le fichier source depasse la limite d'archive")
    if source_path.suffix.casefold() == ".zip":
        try:
            archive = zipfile.ZipFile(source_path, "r")
        except (OSError, zipfile.BadZipFile) as error:
            raise YAGOImportError("Archive ZIP invalide") from error
        with archive:
            members, _ = _validate_zip_members(archive, limits)

            def iterator() -> Iterator[tuple[str, _CountingReader, int | None]]:
                for info in members:
                    try:
                        with archive.open(info, "r") as raw:
                            reader = _CountingReader(
                                raw,
                                member_limit=limits.max_member_bytes,
                                total_limit=limits.max_total_uncompressed_bytes,
                                shared=shared,
                            )
                            yield _safe_member_name(info.filename), reader, info.file_size
                    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                        raise YAGOImportError(f"Lecture ZIP impossible: {info.filename}") from error

            yield iterator()
    else:
        if source_path.suffix.casefold() not in _SUPPORTED_SUFFIXES:
            raise YAGOImportError("Extension attendue: .ttl, .nt, .ntx, .rdf, .txt ou .zip")

        def iterator() -> Iterator[tuple[str, _CountingReader, int | None]]:
            with source_path.open("rb") as raw:
                yield (
                    source_path.name,
                    _CountingReader(
                        raw,
                        member_limit=limits.max_member_bytes,
                        total_limit=limits.max_total_uncompressed_bytes,
                        shared=shared,
                    ),
                    source_path.stat().st_size,
                )

        yield iterator()


def _iter_turtle_statements(
    stream: _CountingReader,
    *,
    max_statement_chars: int,
) -> Iterator[tuple[str, int]]:
    """Separe un flux UTF-8 en instructions sans confondre URI et decimales."""

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    buffer: list[str] = []
    line_number = 1
    statement_line = 1
    in_comment = False
    in_iri = False
    pending_lt = False
    quote: str | None = None
    escaped = False
    pending_dot = False

    def append(character: str) -> None:
        if not buffer and character.isspace():
            return
        buffer.append(character)
        if len(buffer) > max_statement_chars:
            raise YAGOImportError("Une instruction Turtle depasse la longueur autorisee")

    def finish() -> tuple[str, int] | None:
        nonlocal buffer, statement_line
        text = "".join(buffer).strip()
        start = statement_line
        buffer = []
        statement_line = line_number
        return (text, start) if text else None

    def process(text: str) -> Iterator[tuple[str, int]]:
        nonlocal line_number, statement_line, in_comment, in_iri
        nonlocal pending_lt, quote, escaped, pending_dot
        for character in text:
            again = True
            while again:
                again = False
                if in_comment:
                    if character == "\n":
                        in_comment = False
                        line_number += 1
                        if not buffer:
                            statement_line = line_number
                    continue
                if quote is not None:
                    append(character)
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == quote:
                        quote = None
                    if character == "\n":
                        line_number += 1
                    continue
                if in_iri:
                    append(character)
                    if character == ">":
                        in_iri = False
                    if character == "\n":
                        line_number += 1
                    continue
                if pending_lt:
                    pending_lt = False
                    append(character)
                    if character != "<":
                        in_iri = character != ">"
                    continue
                if pending_dot:
                    pending_dot = False
                    if character.isspace() or character == "#":
                        completed = finish()
                        if completed is not None:
                            yield completed
                        again = True
                        continue
                    append(".")
                    again = True
                    continue
                if character == "#":
                    in_comment = True
                    continue
                if character in {'"', "'"}:
                    if not buffer:
                        statement_line = line_number
                    append(character)
                    quote = character
                    continue
                if character == "<":
                    if not buffer:
                        statement_line = line_number
                    append(character)
                    pending_lt = True
                    continue
                if character == ".":
                    pending_dot = True
                    continue
                if character == "\n":
                    append(" ")
                    line_number += 1
                    continue
                if not buffer and not character.isspace():
                    statement_line = line_number
                append(character)

    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            yield from process(decoder.decode(chunk))
        yield from process(decoder.decode(b"", final=True))
    except UnicodeDecodeError as error:
        raise YAGOImportError("Le fichier RDF n'est pas un UTF-8 valide") from error
    if quote is not None or in_iri or pending_lt:
        raise YAGOImportError("Instruction Turtle tronquee")
    if pending_dot:
        completed = finish()
        if completed is not None:
            yield completed
    elif "".join(buffer).strip():
        raise YAGOImportError("Instruction Turtle sans point final")


def _tokenize(statement: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(statement)
    while index < length:
        character = statement[index]
        if character.isspace():
            index += 1
            continue
        if statement.startswith("<<", index) or statement.startswith(">>", index):
            tokens.append(statement[index : index + 2])
            index += 2
            continue
        if character in ";,[]()":
            tokens.append(character)
            index += 1
            continue
        if character == "<":
            end = index + 1
            escaped = False
            while end < length:
                current = statement[end]
                if current == ">" and not escaped:
                    break
                escaped = current == "\\" and not escaped
                if current != "\\":
                    escaped = False
                end += 1
            if end >= length:
                raise YAGOImportError("IRI Turtle non terminee")
            tokens.append(statement[index : end + 1])
            index = end + 1
            continue
        if character in {'"', "'"}:
            delimiter = character
            end = index + 1
            escaped = False
            while end < length:
                current = statement[end]
                if current == delimiter and not escaped:
                    break
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                end += 1
            if end >= length:
                raise YAGOImportError("Litteral Turtle non termine")
            end += 1
            if end < length and statement[end] == "@":
                end += 1
                while end < length and (statement[end].isalnum() or statement[end] == "-"):
                    end += 1
            elif statement.startswith("^^", end):
                end += 2
                if end < length and statement[end] == "<":
                    close = statement.find(">", end + 1)
                    if close < 0:
                        raise YAGOImportError("Datatype IRI non termine")
                    end = close + 1
                else:
                    while end < length and not statement[end].isspace() and statement[end] not in ";,":
                        end += 1
            tokens.append(statement[index:end])
            index = end
            continue
        end = index
        while end < length and not statement[end].isspace() and statement[end] not in ";,[]()":
            if statement.startswith("<<", end) or statement.startswith(">>", end):
                break
            end += 1
        if end == index:
            raise YAGOImportError(f"Jeton Turtle non pris en charge pres de {statement[index:index+20]!r}")
        tokens.append(statement[index:end])
        index = end
    return tokens


def _unescape_turtle(value: str) -> str:
    result: list[str] = []
    index = 0
    simple = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f", '"': '"', "'": "'", "\\": "\\"}
    while index < len(value):
        if value[index] != "\\":
            result.append(value[index])
            index += 1
            continue
        index += 1
        if index >= len(value):
            raise YAGOImportError("Echappement Turtle tronque")
        marker = value[index]
        if marker in simple:
            result.append(simple[marker])
            index += 1
            continue
        digits = 4 if marker == "u" else 8 if marker == "U" else 0
        if not digits or index + digits >= len(value):
            raise YAGOImportError(f"Echappement Turtle non pris en charge: \\{marker}")
        encoded = value[index + 1 : index + 1 + digits]
        try:
            result.append(chr(int(encoded, 16)))
        except (ValueError, OverflowError) as error:
            raise YAGOImportError("Echappement Unicode Turtle invalide") from error
        index += 1 + digits
    return "".join(result)


def _expand_resource(token: str, prefixes: dict[str, str], base: str | None) -> RDFTerm:
    if token == "a":
        return RDFTerm("iri", _RDF_TYPE)
    if token.startswith("<") and token.endswith(">"):
        raw = token[1:-1]
        return RDFTerm("iri", urljoin(base or "", _unescape_turtle(raw)))
    if token.startswith("_:") and len(token) > 2:
        return RDFTerm("bnode", token[2:])
    if ":" in token:
        prefix, local = token.split(":", 1)
        key = prefix + ":"
        if key not in prefixes:
            raise YAGOImportError(f"Prefixe Turtle inconnu: {key}")
        return RDFTerm("iri", prefixes[key] + _unescape_turtle(local))
    raise YAGOImportError(f"Ressource Turtle non prise en charge: {token!r}")


def _parse_literal(token: str, prefixes: dict[str, str], base: str | None) -> RDFTerm:
    delimiter = token[0]
    index = 1
    escaped = False
    while index < len(token):
        character = token[index]
        if character == delimiter and not escaped:
            break
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        index += 1
    if index >= len(token):
        raise YAGOImportError("Litteral Turtle invalide")
    lexical = _unescape_turtle(token[1:index])
    suffix = token[index + 1 :]
    language = None
    datatype = None
    if suffix.startswith("@"):
        language = suffix[1:].casefold()
        if not _LANG_RE.fullmatch(language):
            raise YAGOImportError("Langue de litteral Turtle invalide")
    elif suffix.startswith("^^"):
        datatype = _expand_resource(suffix[2:], prefixes, base).value
    elif suffix:
        raise YAGOImportError("Suffixe de litteral Turtle invalide")
    return RDFTerm("literal", lexical, datatype=datatype, language=language)


def _parse_term(
    tokens: Sequence[str],
    index: int,
    prefixes: dict[str, str],
    base: str | None,
) -> tuple[RDFTerm, int]:
    if index >= len(tokens):
        raise YAGOImportError("Terme Turtle manquant")
    token = tokens[index]
    if token[0:1] in {'"', "'"}:
        return _parse_literal(token, prefixes, base), index + 1
    if _NUMERIC_RE.fullmatch(token):
        datatype = (
            "http://www.w3.org/2001/XMLSchema#double"
            if "e" in token.casefold()
            else "http://www.w3.org/2001/XMLSchema#decimal"
            if "." in token
            else "http://www.w3.org/2001/XMLSchema#integer"
        )
        return RDFTerm("literal", token, datatype=datatype), index + 1
    if token in {"true", "false"}:
        return RDFTerm("literal", token, datatype="http://www.w3.org/2001/XMLSchema#boolean"), index + 1
    return _expand_resource(token, prefixes, base), index + 1


def _parse_subject(
    tokens: Sequence[str],
    index: int,
    prefixes: dict[str, str],
    base: str | None,
) -> tuple[RDFTerm | RDFTriple, int]:
    if index < len(tokens) and tokens[index] == "<<":
        subject, index = _parse_term(tokens, index + 1, prefixes, base)
        predicate, index = _parse_term(tokens, index, prefixes, base)
        object_term, index = _parse_term(tokens, index, prefixes, base)
        if index >= len(tokens) or tokens[index] != ">>":
            raise YAGOImportError("Triplet RDF-star non termine")
        if predicate.kind != "iri" or subject.kind == "literal":
            raise YAGOImportError("Triplet RDF-star invalide")
        return RDFTriple(subject, predicate, object_term), index + 1
    term, index = _parse_term(tokens, index, prefixes, base)
    if term.kind == "literal":
        raise YAGOImportError("Un sujet RDF ne peut pas etre un litteral")
    return term, index


def _parse_statement(
    statement: str,
    prefixes: dict[str, str],
    base: str | None,
) -> tuple[list[ParsedRDF], str | None]:
    tokens = _tokenize(statement)
    if not tokens:
        return [], base
    directive = tokens[0].casefold()
    if directive in {"@prefix", "prefix"}:
        if len(tokens) != 3 or not tokens[1].endswith(":"):
            raise YAGOImportError("Declaration @prefix invalide")
        iri = _expand_resource(tokens[2], prefixes, base)
        prefixes[tokens[1]] = iri.value
        return [], base
    if directive in {"@base", "base"}:
        if len(tokens) != 2:
            raise YAGOImportError("Declaration @base invalide")
        return [], _expand_resource(tokens[1], prefixes, base).value
    if any(token in {"[", "]", "(", ")"} for token in tokens):
        raise YAGOImportError("Collection ou noeud anonyme Turtle non pris en charge")

    subject, index = _parse_subject(tokens, 0, prefixes, base)
    parsed: list[ParsedRDF] = []
    while index < len(tokens):
        while index < len(tokens) and tokens[index] == ";":
            index += 1
        if index >= len(tokens):
            break
        predicate, index = _parse_term(tokens, index, prefixes, base)
        if predicate.kind != "iri":
            raise YAGOImportError("Le predicat RDF doit etre une IRI")
        need_object = True
        while need_object:
            object_term, index = _parse_term(tokens, index, prefixes, base)
            parsed.append(ParsedRDF(subject, predicate, object_term))
            if index < len(tokens) and tokens[index] == ",":
                index += 1
                continue
            need_object = False
        if index < len(tokens) and tokens[index] != ";":
            raise YAGOImportError(f"Jeton Turtle inattendu: {tokens[index]!r}")
    return parsed, base


def _component(entry_name: str) -> str:
    lowered = Path(entry_name).name.casefold()
    if "meta" in lowered or lowered.endswith(".ntx"):
        return "meta"
    if "schema" in lowered or "shape" in lowered:
        return "schema"
    if "taxonom" in lowered or "type" in lowered:
        return "taxonomy"
    if "beyond" in lowered:
        return "facts_beyond_wikipedia"
    return "facts"


def _display_iri(iri: str) -> str:
    parsed = urlsplit(iri)
    fragment = parsed.fragment
    tail = fragment or parsed.path.rstrip("/").rsplit("/", 1)[-1] or iri
    return " ".join(unquote(tail).replace("_u2013_", "-").replace("_", " ").split())


def _term_label(term: RDFTerm) -> str:
    return term.value if term.kind == "literal" else _display_iri(term.value)


def _confidence_basis(*, statement_provenance: bool = False, provenance_iri: str | None = None) -> str:
    value: dict[str, Any] = {
        "kind": "statement_provenance" if statement_provenance else "dataset_provenance",
        "numeric_probability": None,
        "dataset": YAGO_DATASET_NAME,
        "lineage_id": LINEAGE_ID,
        "lineage": ["Wikidata", "Schema.org", "YAGO 4.5"],
        "independence_warning": "YAGO et Wikidata partagent la meme lignee et ne valent pas deux confirmations independantes.",
    }
    if provenance_iri is not None:
        value["statement_provenance_iri"] = provenance_iri
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _temporal_field(predicate_iri: str) -> str | None:
    if predicate_iri in _SCHEMA_STARTS:
        return "temporal_start"
    if predicate_iri in _SCHEMA_ENDS:
        return "temporal_end"
    return None


class _YAGOStore:
    def __init__(self, db_path: Path, *, batch_size: int) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.batch_size = batch_size
        self.pending = 0
        self.in_transaction = False
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS imports (
                import_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_size INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL,
                dataset_name TEXT NOT NULL,
                dataset_release TEXT NOT NULL,
                lineage_id TEXT NOT NULL,
                parser_version TEXT NOT NULL,
                license_name TEXT NOT NULL,
                license_url TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                counts_json TEXT
            );
            CREATE TABLE IF NOT EXISTS entities (
                iri TEXT PRIMARY KEY,
                compact_label TEXT NOT NULL,
                preferred_label TEXT,
                preferred_language TEXT
            );
            CREATE TABLE IF NOT EXISTS relations (
                iri TEXT PRIMARY KEY,
                compact_label TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claims (
                statement_id TEXT PRIMARY KEY,
                subject_kind TEXT NOT NULL,
                subject_value TEXT NOT NULL,
                predicate_iri TEXT NOT NULL,
                object_kind TEXT NOT NULL,
                object_value TEXT NOT NULL,
                object_datatype TEXT,
                object_language TEXT,
                subject_label TEXT NOT NULL,
                predicate_label TEXT NOT NULL,
                object_label TEXT NOT NULL,
                temporal_start TEXT,
                temporal_end TEXT,
                epistemic_status TEXT NOT NULL,
                confidence_level TEXT NOT NULL,
                confidence_basis_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claims_subject_predicate
                ON claims(subject_value, predicate_iri);
            CREATE INDEX IF NOT EXISTS idx_claims_predicate
                ON claims(predicate_iri);
            CREATE TABLE IF NOT EXISTS claim_sources (
                statement_id TEXT NOT NULL,
                import_id TEXT NOT NULL REFERENCES imports(import_id),
                entry_name TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                component TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                dataset_release TEXT NOT NULL,
                lineage_id TEXT NOT NULL,
                PRIMARY KEY(statement_id, import_id, entry_name, line_number)
            );
            CREATE TABLE IF NOT EXISTS annotations (
                annotation_id TEXT PRIMARY KEY,
                statement_id TEXT NOT NULL,
                predicate_iri TEXT NOT NULL,
                object_kind TEXT NOT NULL,
                object_value TEXT NOT NULL,
                object_datatype TEXT,
                object_language TEXT,
                import_id TEXT NOT NULL REFERENCES imports(import_id),
                entry_name TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                component TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_annotations_statement
                ON annotations(statement_id, predicate_iri);
            CREATE TABLE IF NOT EXISTS entity_labels (
                iri TEXT NOT NULL,
                label TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT '',
                statement_id TEXT NOT NULL,
                PRIMARY KEY(iri, label, language, statement_id)
            );
            """
        )
        row = self.connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is not None and int(row["value"]) != SCHEMA_VERSION:
            raise YAGOImportError("Version SQLite YAGO incompatible")
        self.connection.execute(
            "INSERT OR IGNORE INTO schema_metadata(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    def begin_import(
        self,
        import_id: str,
        source_path: Path,
        *,
        release: str,
    ) -> None:
        metadata = source_path.stat()
        self.connection.execute(
            """
            INSERT INTO imports(
                import_id, source_path, source_size, source_mtime_ns,
                dataset_name, dataset_release, lineage_id, parser_version,
                license_name, license_url, started_at, status
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
            ON CONFLICT(import_id) DO UPDATE SET
                started_at = excluded.started_at,
                completed_at = NULL,
                status = 'running',
                counts_json = NULL
            """,
            (
                import_id,
                str(source_path),
                metadata.st_size,
                metadata.st_mtime_ns,
                YAGO_DATASET_NAME,
                release,
                LINEAGE_ID,
                PARSER_VERSION,
                YAGO_LICENSE,
                YAGO_LICENSE_URL,
                _utcnow(),
            ),
        )
        self.connection.commit()

    def _begin(self) -> None:
        if not self.in_transaction:
            self.connection.execute("BEGIN IMMEDIATE")
            self.in_transaction = True

    def _touch(self) -> None:
        self.pending += 1
        if self.pending >= self.batch_size:
            self.connection.commit()
            self.in_transaction = False
            self.pending = 0

    def _entity(self, term: RDFTerm) -> None:
        if term.kind != "iri":
            return
        self.connection.execute(
            "INSERT OR IGNORE INTO entities(iri, compact_label) VALUES(?, ?)",
            (term.value, _display_iri(term.value)),
        )

    def _relation(self, term: RDFTerm) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO relations(iri, compact_label) VALUES(?, ?)",
            (term.value, _display_iri(term.value)),
        )

    def add_claim(
        self,
        triple: RDFTriple,
        *,
        import_id: str,
        entry_name: str,
        line_number: int,
        component: str,
        release: str,
    ) -> tuple[bool, bool, bool]:
        self._begin()
        self._entity(triple.subject)
        self._relation(triple.predicate)
        self._entity(triple.object)
        statement_id = triple.statement_id
        temporal = self.connection.execute(
            """
            SELECT predicate_iri, object_value FROM annotations
            WHERE statement_id = ? AND predicate_iri IN (?, ?, ?, ?)
            ORDER BY rowid
            """,
            (statement_id, *sorted(_SCHEMA_STARTS | _SCHEMA_ENDS)),
        ).fetchall()
        start = next((row["object_value"] for row in temporal if row["predicate_iri"] in _SCHEMA_STARTS), None)
        end = next((row["object_value"] for row in temporal if row["predicate_iri"] in _SCHEMA_ENDS), None)
        provenance_row = self.connection.execute(
            """
            SELECT object_value FROM annotations
            WHERE statement_id = ? AND predicate_iri LIKE ?
            ORDER BY rowid LIMIT 1
            """,
            (statement_id, _PROV_PREFIX + "%"),
        ).fetchone()
        confidence_level = "statement-attributed" if provenance_row is not None else "dataset-attributed"
        confidence_basis = _confidence_basis(
            statement_provenance=provenance_row is not None,
            provenance_iri=None if provenance_row is None else provenance_row["object_value"],
        )
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO claims(
                statement_id, subject_kind, subject_value, predicate_iri,
                object_kind, object_value, object_datatype, object_language,
                subject_label, predicate_label, object_label,
                temporal_start, temporal_end, epistemic_status,
                confidence_level, confidence_basis_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'asserted-by-dataset',
                     ?, ?, ?)
            """,
            (
                statement_id,
                triple.subject.kind,
                triple.subject.value,
                triple.predicate.value,
                triple.object.kind,
                triple.object.value,
                triple.object.datatype,
                triple.object.language,
                _term_label(triple.subject),
                _term_label(triple.predicate),
                _term_label(triple.object),
                start,
                end,
                confidence_level,
                confidence_basis,
                _utcnow(),
            ),
        )
        claim_created = cursor.rowcount == 1
        if not claim_created and (start is not None or end is not None or provenance_row is not None):
            self.connection.execute(
                """
                UPDATE claims
                SET temporal_start = COALESCE(temporal_start, ?),
                    temporal_end = COALESCE(temporal_end, ?),
                    confidence_level = CASE WHEN ? = 1 THEN 'statement-attributed' ELSE confidence_level END,
                    confidence_basis_json = CASE WHEN ? = 1 THEN ? ELSE confidence_basis_json END
                WHERE statement_id = ?
                """,
                (
                    start,
                    end,
                    int(provenance_row is not None),
                    int(provenance_row is not None),
                    confidence_basis,
                    statement_id,
                ),
            )
        source_cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO claim_sources(
                statement_id, import_id, entry_name, line_number, component,
                dataset_name, dataset_release, lineage_id
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                statement_id,
                import_id,
                entry_name,
                line_number,
                component,
                YAGO_DATASET_NAME,
                release,
                LINEAGE_ID,
            ),
        )
        source_created = source_cursor.rowcount == 1
        label_created = False
        if triple.predicate.value == _RDFS_LABEL and triple.object.kind == "literal":
            label_cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO entity_labels(iri, label, language, statement_id)
                VALUES(?, ?, ?, ?)
                """,
                (
                    triple.subject.value,
                    triple.object.value,
                    triple.object.language or "",
                    statement_id,
                ),
            )
            label_created = label_cursor.rowcount == 1
            # Francais, puis anglais, puis toute etiquette sont preferes de facon stable.
            rank = {"fr": 0, "en": 1}.get(triple.object.language or "", 2)
            current = self.connection.execute(
                "SELECT preferred_language FROM entities WHERE iri = ?",
                (triple.subject.value,),
            ).fetchone()
            current_rank = 99 if current is None or current["preferred_language"] is None else {"fr": 0, "en": 1}.get(current["preferred_language"], 2)
            if rank < current_rank or current_rank == 99:
                self.connection.execute(
                    "UPDATE entities SET preferred_label = ?, preferred_language = ? WHERE iri = ?",
                    (triple.object.value, triple.object.language or "", triple.subject.value),
                )
        self._touch()
        return claim_created, source_created, label_created

    def add_annotation(
        self,
        quoted: RDFTriple,
        predicate: RDFTerm,
        object_term: RDFTerm,
        *,
        import_id: str,
        entry_name: str,
        line_number: int,
        component: str,
        release: str,
    ) -> tuple[bool, bool, bool, bool, bool]:
        temporal_field = _temporal_field(predicate.value)
        scoped_claim_created = False
        scoped_source_created = False
        if temporal_field is not None:
            # En mode RDF-star "separate assertions", le triplet cite avec sa
            # portee temporelle est le fait utile meme s'il n'existe pas comme
            # triplet RDF autonome dans le fichier facts.
            scoped_claim_created, scoped_source_created, _ = self.add_claim(
                quoted,
                import_id=import_id,
                entry_name=entry_name,
                line_number=line_number,
                component=component,
                release=release,
            )
        self._begin()
        statement_id = quoted.statement_id
        annotation_key = "\0".join(
            (
                statement_id,
                predicate.canonical(),
                object_term.canonical(),
                import_id,
                entry_name,
                str(line_number),
            )
        )
        annotation_id = hashlib.sha256(annotation_key.encode("utf-8")).hexdigest()
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO annotations(
                annotation_id, statement_id, predicate_iri, object_kind,
                object_value, object_datatype, object_language, import_id,
                entry_name, line_number, component, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                annotation_id,
                statement_id,
                predicate.value,
                object_term.kind,
                object_term.value,
                object_term.datatype,
                object_term.language,
                import_id,
                entry_name,
                line_number,
                component,
                _utcnow(),
            ),
        )
        created = cursor.rowcount == 1
        temporal = temporal_field is not None
        provenance = predicate.value.startswith(_PROV_PREFIX)
        if temporal:
            self.connection.execute(
                f"""
                UPDATE claims
                SET {temporal_field} = COALESCE({temporal_field}, ?),
                    epistemic_status = 'temporally-scoped-by-rdf-star'
                WHERE statement_id = ?
                """,
                (object_term.value, statement_id),
            )
        if provenance:
            self.connection.execute(
                """
                UPDATE claims
                SET confidence_level = 'statement-attributed', confidence_basis_json = ?
                WHERE statement_id = ?
                """,
                (_confidence_basis(statement_provenance=True, provenance_iri=object_term.value), statement_id),
            )
        self._touch()
        return created, temporal, provenance, scoped_claim_created, scoped_source_created

    def finish_import(self, import_id: str, counts: dict[str, Any], *, status: str) -> None:
        if self.in_transaction:
            self.connection.commit()
            self.in_transaction = False
            self.pending = 0
        self.connection.execute(
            "UPDATE imports SET completed_at = ?, status = ?, counts_json = ? WHERE import_id = ?",
            (
                _utcnow(),
                status,
                json.dumps(counts, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                import_id,
            ),
        )
        self.connection.commit()

    def variant_groups(self) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count FROM (
                SELECT subject_value, predicate_iri
                FROM claims
                GROUP BY subject_value, predicate_iri
                HAVING COUNT(DISTINCT object_kind || char(0) || object_value || char(0) ||
                    COALESCE(object_datatype, '') || char(0) || COALESCE(object_language, '')) > 1
            )
            """
        ).fetchone()
        return int(row["count"])

    def totals(self) -> dict[str, int]:
        return {
            f"{table}_total": int(
                self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in ("claims", "annotations", "entities", "relations", "claim_sources")
        }

    def close(self) -> None:
        if self.in_transaction:
            self.connection.commit()
        self.connection.close()


def _import_id(source_path: Path, release: str) -> str:
    metadata = source_path.stat()
    value = "\0".join(
        (
            str(source_path.resolve()),
            str(metadata.st_size),
            str(metadata.st_mtime_ns),
            release,
            PARSER_VERSION,
        )
    )
    return "yago-" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def import_yago(
    source: str | Path,
    database: str | Path | None = None,
    *,
    release: str = "4.5.0.2",
    dry_run: bool = False,
    strict: bool = False,
    limits: ImportLimits | None = None,
) -> dict[str, Any]:
    """Analyse ou importe un dump local YAGO sans reseau et sans extraction.

    Un ``dry_run`` ne cree ni ne modifie aucune base. Un import reel ecrit par
    petits lots dans une base SQLite dediee. Les identifiants de triplets sont
    deterministes : relancer le meme fichier reprend le travail sans doublons.
    """

    limits = limits or ImportLimits()
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise YAGOImportError(f"Source YAGO introuvable: {source_path}")
    if not isinstance(release, str) or not release.strip() or len(release) > 64:
        raise YAGOImportError("release doit etre une chaine courte non vide")
    if not dry_run and database is None:
        raise YAGOImportError("database est obligatoire hors dry-run")
    database_path = None if database is None else Path(database).expanduser().resolve()
    if database_path is not None and database_path == source_path:
        raise YAGOImportError("La base de destination doit etre separee du dump")

    run_id = _import_id(source_path, release.strip())
    counts: dict[str, Any] = {
        "members": 0,
        "members_completed": 0,
        "bytes_read": 0,
        "statements": 0,
        "directives": 0,
        "triples_seen": 0,
        "claims_created": 0,
        "claim_sources_created": 0,
        "annotations_created": 0,
        "labels_created": 0,
        "temporal_annotations": 0,
        "provenance_annotations": 0,
        "parse_errors": 0,
        "limited": False,
    }
    store = None if dry_run else _YAGOStore(database_path, batch_size=limits.batch_size)  # type: ignore[arg-type]
    if store is not None:
        store.begin_import(run_id, source_path, release=release.strip())
    stop = False
    try:
        with _source_members(source_path, limits) as members:
            try:
                for entry_name, reader, _declared_size in members:
                    counts["members"] += 1
                    component = _component(entry_name)
                    # Prefixes et @base ont une portee de document, pas d'archive.
                    prefixes: dict[str, str] = {}
                    base: str | None = None
                    for statement, line_number in _iter_turtle_statements(
                        reader, max_statement_chars=limits.max_statement_chars
                    ):
                        counts["statements"] += 1
                        if counts["statements"] > limits.max_statements:
                            raise YAGOImportError("Le nombre maximal d'instructions Turtle est depasse")
                        try:
                            parsed, base = _parse_statement(statement, prefixes, base)
                        except YAGOImportError:
                            counts["parse_errors"] += 1
                            if strict or counts["parse_errors"] > limits.max_parse_errors:
                                raise
                            continue
                        if len(prefixes) > limits.max_prefixes_per_member:
                            raise YAGOImportError("Le nombre maximal de prefixes Turtle est depasse")
                        if not parsed:
                            counts["directives"] += 1
                        for item in parsed:
                            if counts["triples_seen"] >= limits.max_triples:
                                counts["limited"] = True
                                stop = True
                                break
                            counts["triples_seen"] += 1
                            if store is None:
                                continue
                            if isinstance(item.subject, RDFTriple):
                                (
                                    created,
                                    temporal,
                                    provenance,
                                    scoped_claim_created,
                                    scoped_source_created,
                                ) = store.add_annotation(
                                    item.subject,
                                    item.predicate,
                                    item.object,
                                    import_id=run_id,
                                    entry_name=entry_name,
                                    line_number=line_number,
                                    component=component,
                                    release=release.strip(),
                                )
                                counts["annotations_created"] += int(created)
                                counts["temporal_annotations"] += int(created and temporal)
                                counts["provenance_annotations"] += int(created and provenance)
                                counts["claims_created"] += int(scoped_claim_created)
                                counts["claim_sources_created"] += int(scoped_source_created)
                            else:
                                claim_created, source_created, label_created = store.add_claim(
                                    RDFTriple(item.subject, item.predicate, item.object),
                                    import_id=run_id,
                                    entry_name=entry_name,
                                    line_number=line_number,
                                    component=component,
                                    release=release.strip(),
                                )
                                counts["claims_created"] += int(claim_created)
                                counts["claim_sources_created"] += int(source_created)
                                counts["labels_created"] += int(label_created)
                        if stop:
                            break
                    counts["bytes_read"] += reader.member_bytes
                    if not stop:
                        counts["members_completed"] += 1
                    if stop:
                        break
            finally:
                close = getattr(members, "close", None)
                if close is not None:
                    close()
        if store is not None:
            counts["variant_groups"] = store.variant_groups()
            counts.update(store.totals())
            store.finish_import(run_id, counts, status="limited" if stop else "completed")
    except Exception:
        if store is not None:
            try:
                store.finish_import(run_id, counts, status="failed")
            except sqlite3.Error:
                pass
        raise
    finally:
        if store is not None:
            store.close()

    return {
        "schema_version": "yago-import-report-v1",
        "import_id": run_id,
        "dry_run": dry_run,
        "source": str(source_path),
        "database": None if database_path is None else str(database_path),
        "dataset": {
            "name": YAGO_DATASET_NAME,
            "release": release.strip(),
            "license": YAGO_LICENSE,
            "license_url": YAGO_LICENSE_URL,
            "download_url": YAGO_DOWNLOAD_URL,
            "citation": YAGO_PAPER_URL,
        },
        "provenance": {
            "lineage_id": LINEAGE_ID,
            "lineage": ["Wikidata", "Schema.org", "YAGO 4.5"],
            "independent_sources": 1,
            "warning": "YAGO et Wikidata ne sont jamais comptes comme deux sources independantes.",
            "confidence_model": "provenance explicite, jamais probabilite inventee",
        },
        "counts": counts,
        "limits": asdict(limits),
        "resume_safe": not dry_run,
        "network_used": False,
        "archive_extracted": False,
        "parser_version": PARSER_VERSION,
    }


def iter_memory_batches(
    database: str | Path,
    *,
    batch_size: int = 500,
) -> Iterator[list[dict[str, Any]]]:
    """Expose les affirmations en lots bornes pour un espace de reference.

    Les enregistrements utilisent ``source.type = inferred`` : le dataset est
    externe et attribue, pas confirme personnellement par l'utilisateur. Ils
    peuvent etre injectes hors ligne dans un ``MemoryEngine`` puis monter cette
    base en espace ``reference`` du ``MemoryHub``.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 10_000:
        raise ValueError("batch_size doit etre compris entre 1 et 10000")
    path = Path(database).expanduser().resolve()
    if not path.is_file():
        raise YAGOImportError(f"Base YAGO introuvable: {path}")
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        last_rowid = 0
        while True:
            rows = connection.execute(
                """
                SELECT rowid, * FROM claims
                WHERE rowid > ?
                ORDER BY rowid
                LIMIT ?
                """,
                (last_rowid, batch_size),
            ).fetchall()
            if not rows:
                break
            batch: list[dict[str, Any]] = []
            for row in rows:
                last_rowid = int(row["rowid"])
                temporal = {
                    key: row[column]
                    for key, column in (("start", "temporal_start"), ("end", "temporal_end"))
                    if row[column] is not None
                }
                batch.append(
                    {
                        "text": f"{row['subject_label']} — {row['predicate_label']} → {row['object_label']}",
                        "episode_id": f"yago-{row['statement_id'][:32]}",
                        "context": {
                            "origin": "yago_4_5",
                            "category": "reference_knowledge_graph",
                            "statement_id": row["statement_id"],
                            "subject": row["subject_value"],
                            "predicate": row["predicate_iri"],
                            "object": {
                                "kind": row["object_kind"],
                                "value": row["object_value"],
                                "datatype": row["object_datatype"],
                                "language": row["object_language"],
                            },
                            "temporal": temporal,
                            "provenance": json.loads(row["confidence_basis_json"]),
                            "confidence_level": row["confidence_level"],
                        },
                        "source": {
                            "type": "inferred",
                            "medium": "yago_4_5_offline_import",
                            "lineage_id": LINEAGE_ID,
                        },
                        "idempotency_key": f"yago:{row['statement_id']}",
                    }
                )
            yield batch
    finally:
        connection.close()


__all__ = [
    "ImportLimits",
    "LINEAGE_ID",
    "PARSER_VERSION",
    "YAGOImportError",
    "import_yago",
    "iter_memory_batches",
]
