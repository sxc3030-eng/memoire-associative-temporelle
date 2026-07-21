#!/usr/bin/env python3
"""Download the pinned MAT-LM base model without third-party dependencies.

This downloader deliberately supports exactly one public repository, one immutable
revision and a small, explicit set of files.  It is intended as a fallback when
``huggingface_hub`` cannot run in the local environment; it is not a general URL
downloader.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable
import urllib.error
import urllib.parse
import urllib.request


REPOSITORY = "ibm-granite/granite-3.3-2b-instruct"
REVISION = "707f574c62054322f6b5b04b6d075f0a8f05e0f0"
ORIGIN = "https://huggingface.co"
MANIFEST_NAME = "download-manifest.json"
TOTAL_MAX_BYTES = 6 * 1024**3
CHUNK_BYTES = 4 * 1024**2
TIMEOUT_SECONDS = 90


@dataclasses.dataclass(frozen=True)
class FileSpec:
    name: str
    max_bytes: int
    required: bool = True
    expected_sha256: str | None = None


# The two weight hashes are the upstream safetensors SHA-256 values.  Small
# metadata files are still hashed into our local manifest and are pinned by the
# immutable repository revision in their download URL.
FILE_SPECS: tuple[FileSpec, ...] = (
    FileSpec("config.json", 2 * 1024**2),
    FileSpec("generation_config.json", 2 * 1024**2),
    FileSpec(
        "model-00001-of-00002.safetensors",
        11 * 512 * 1024**2,
        expected_sha256="12880d33c0ad4726af5cf8c07406905f9b496253c58ee46f52be8bde8ccf2254",
    ),
    FileSpec(
        "model-00002-of-00002.safetensors",
        256 * 1024**2,
        expected_sha256="a8757c5bf7627933e7fddbd9bab0533491b4dc0962820e0617f356ca1a379ffa",
    ),
    FileSpec("model.safetensors.index.json", 8 * 1024**2),
    FileSpec("tokenizer.json", 32 * 1024**2),
    FileSpec("tokenizer_config.json", 8 * 1024**2),
    # These tokenizer sidecars are present in some Transformers exports.  A 404
    # is acceptable because tokenizer.json is the required complete tokenizer.
    FileSpec("vocab.json", 16 * 1024**2, required=False),
    FileSpec("merges.txt", 16 * 1024**2, required=False),
    FileSpec("added_tokens.json", 4 * 1024**2, required=False),
    FileSpec("special_tokens_map.json", 4 * 1024**2, required=False),
    FileSpec("README.md", 4 * 1024**2, required=False),
    FileSpec("LICENSE", 2 * 1024**2, required=False),
)


class DownloadError(RuntimeError):
    """A safe-download invariant was not satisfied."""


_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_UNSATISFIED_RANGE = re.compile(r"^bytes \*/(\d+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REDIRECT_PATH_PREFIXES: dict[str, tuple[str, ...]] = {
    "cas-bridge.xethub.hf.co": ("/xet-bridge-us/", "/xet-bridge-eu/"),
    "transfer.xethub.hf.co": ("/xet-bridge-us/", "/xet-bridge-eu/"),
    "cdn-lfs.huggingface.co": ("/repos/",),
    "cdn-lfs-us-1.hf.co": ("/repos/",),
    "cdn-lfs-eu-1.hf.co": ("/repos/",),
}


def _allowed_names() -> frozenset[str]:
    return frozenset(spec.name for spec in FILE_SPECS)


def _validate_specs() -> None:
    names = [spec.name for spec in FILE_SPECS]
    if len(names) != len(set(names)):
        raise DownloadError("La liste fixe contient un fichier en double")
    if any(spec.max_bytes <= 0 for spec in FILE_SPECS):
        raise DownloadError("Une limite de fichier fixe est invalide")
    if sum(spec.max_bytes for spec in FILE_SPECS) > TOTAL_MAX_BYTES:
        raise DownloadError("Les limites de fichiers depassent la limite totale")
    if any(
        spec.expected_sha256 is not None and _SHA256.fullmatch(spec.expected_sha256) is None
        for spec in FILE_SPECS
    ):
        raise DownloadError("Une empreinte SHA-256 fixe est invalide")


def _validate_filename(name: str) -> None:
    if name not in _allowed_names():
        raise DownloadError(f"Fichier non autorise: {name!r}")
    candidate = Path(name)
    if candidate.name != name or candidate.is_absolute() or name in {".", ".."}:
        raise DownloadError(f"Chemin de fichier dangereux: {name!r}")


def _safe_target(output_dir: Path, name: str) -> Path:
    """Return a target proven to be an immediate child of *output_dir*."""

    _validate_filename(name)
    root = output_dir.resolve(strict=False)
    target = (root / name).resolve(strict=False)
    if target.parent != root:
        raise DownloadError(f"Le chemin sort du dossier de destination: {name!r}")
    return target


def _download_url(name: str) -> str:
    _validate_filename(name)
    encoded_name = urllib.parse.quote(name, safe="")
    return f"{ORIGIN}/{REPOSITORY}/resolve/{REVISION}/{encoded_name}?download=true"


def _clean_url_path(url: str) -> tuple[urllib.parse.SplitResult, str]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https":
        raise DownloadError("Seules les adresses HTTPS sont autorisees")
    if parsed.username is not None or parsed.password is not None:
        raise DownloadError("Une adresse avec identifiants est refusee")
    try:
        port = parsed.port
    except ValueError as exc:
        raise DownloadError("Port invalide dans l'adresse de telechargement") from exc
    if port not in (None, 443):
        raise DownloadError("Seul le port HTTPS standard est autorise")
    if parsed.fragment:
        raise DownloadError("Les fragments d'adresse sont refuses")
    decoded_path = urllib.parse.unquote(parsed.path)
    if "\\" in decoded_path or "\x00" in decoded_path:
        raise DownloadError("Chemin distant dangereux")
    if any(segment in {".", ".."} for segment in decoded_path.split("/")):
        raise DownloadError("Traversal distant refuse")
    return parsed, decoded_path


def _validate_remote_url(url: str, filename: str) -> None:
    """Allow only the pinned resolver URL and tightly scoped CDN redirects."""

    _validate_filename(filename)
    parsed, path = _clean_url_path(url)
    host = (parsed.hostname or "").lower()
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in REPOSITORY.split("/"))
    encoded_name = urllib.parse.quote(filename, safe="")
    resolver_path = f"/{quoted_repo}/resolve/{REVISION}/{encoded_name}"
    cache_path = f"/api/resolve-cache/models/{quoted_repo}/{REVISION}/{encoded_name}"

    if host == "huggingface.co":
        if parsed.path not in {resolver_path, cache_path}:
            raise DownloadError(f"Chemin Hugging Face non autorise: {path}")
        return

    prefixes = _REDIRECT_PATH_PREFIXES.get(host)
    if prefixes is None or not any(path.startswith(prefix) for prefix in prefixes):
        raise DownloadError(f"Redirection non autorisee vers {host or '<sans hote>'}{path}")


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects outside Hugging Face's fixed model-delivery hosts."""

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_remote_url(newurl, self.filename)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_opener(filename: str):
    return urllib.request.build_opener(SafeRedirectHandler(filename))


def _request(url: str, *, offset: int = 0) -> urllib.request.Request:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "MAT-LM-pinned-downloader/1.0",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers, method="GET")


def _response_status(response) -> int:  # noqa: ANN001
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    return int(status)


def _header_int(headers, name: str) -> int | None:  # noqa: ANN001
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise DownloadError(f"En-tete {name} invalide") from exc
    if value < 0:
        raise DownloadError(f"En-tete {name} negatif")
    return value


def _partial_total(headers, offset: int) -> tuple[int, int]:  # noqa: ANN001
    raw = headers.get("Content-Range")
    match = _CONTENT_RANGE.fullmatch(str(raw or ""))
    if match is None:
        raise DownloadError("Reponse partielle sans Content-Range valide")
    start, end, total = (int(value) for value in match.groups())
    if start != offset or end < start or total <= end:
        raise DownloadError("Content-Range incoherent")
    length = end - start + 1
    content_length = _header_int(headers, "Content-Length")
    if content_length is not None and content_length != length:
        raise DownloadError("Content-Length ne correspond pas a Content-Range")
    return length, total


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _check_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise DownloadError(f"Lien symbolique refuse pour {label}: {path}")
    if path.exists() and not path.is_file():
        raise DownloadError(f"Un fichier normal est attendu pour {label}: {path}")


def _inspect_output_dir(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise DownloadError("Le dossier de destination ne peut pas etre un lien symbolique")
    if output_dir.exists() and not output_dir.is_dir():
        raise DownloadError("La destination existe mais n'est pas un dossier")
    if not output_dir.exists():
        return

    allowed = set(_allowed_names())
    allowed.update(f"{name}.part" for name in _allowed_names())
    allowed.update({MANIFEST_NAME, f".{MANIFEST_NAME}.tmp"})
    unexpected = sorted(entry.name for entry in output_dir.iterdir() if entry.name not in allowed)
    if unexpected:
        joined = ", ".join(unexpected[:5])
        raise DownloadError(f"Le dossier contient des elements sans rapport: {joined}")

    for spec in FILE_SPECS:
        final = _safe_target(output_dir, spec.name)
        partial = _safe_target(output_dir, spec.name).with_name(f"{spec.name}.part")
        _check_regular_file(final, spec.name)
        _check_regular_file(partial, f"{spec.name}.part")
        if final.exists() and partial.exists():
            raise DownloadError(f"Fichier final et partiel presents ensemble: {spec.name}")


def _verify_local_file(path: Path, spec: FileSpec) -> dict[str, object]:
    size = path.stat().st_size
    if size <= 0 or size > spec.max_bytes:
        raise DownloadError(f"Taille locale invalide pour {spec.name}: {size} octets")
    digest = _sha256_file(path)
    if spec.expected_sha256 and digest != spec.expected_sha256:
        raise DownloadError(f"SHA-256 invalide pour {spec.name}")
    return {"name": spec.name, "bytes": size, "sha256": digest, "status": "present"}


def _finalize_complete_partial(partial: Path, target: Path, spec: FileSpec) -> dict[str, object]:
    record = _verify_local_file(partial, spec)
    os.replace(partial, target)
    return record


def _open_response(opener, request: urllib.request.Request):  # noqa: ANN001
    return opener.open(request, timeout=TIMEOUT_SECONDS)


def _download_one(output_dir: Path, spec: FileSpec, *, opener=None) -> dict[str, object] | None:  # noqa: ANN001
    target = _safe_target(output_dir, spec.name)
    partial = target.with_name(f"{spec.name}.part")
    _check_regular_file(target, spec.name)
    _check_regular_file(partial, f"{spec.name}.part")

    if target.exists():
        return _verify_local_file(target, spec)

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec.max_bytes:
        raise DownloadError(f"Fichier partiel trop grand: {spec.name}")

    url = _download_url(spec.name)
    active_opener = opener if opener is not None else _build_opener(spec.name)
    try:
        response = _open_response(active_opener, _request(url, offset=offset))
    except urllib.error.HTTPError as exc:
        _validate_remote_url(exc.geturl(), spec.name)
        if exc.code == 404 and not spec.required:
            return None
        if exc.code == 416 and offset:
            match = _UNSATISFIED_RANGE.fullmatch(str(exc.headers.get("Content-Range") or ""))
            remote_total = int(match.group(1)) if match else -1
            if remote_total == offset:
                return _finalize_complete_partial(partial, target, spec)
        raise DownloadError(f"Echec HTTP {exc.code} pour {spec.name}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"Echec reseau pour {spec.name}: {exc.reason}") from exc

    with response:
        final_url = response.geturl()
        _validate_remote_url(final_url, spec.name)
        status = _response_status(response)
        content_length = _header_int(response.headers, "Content-Length")

        if status == 206:
            expected_remaining, remote_total = _partial_total(response.headers, offset)
            if remote_total > spec.max_bytes:
                raise DownloadError(f"Fichier distant trop grand: {spec.name}")
            mode = "ab" if offset else "wb"
            starting_size = offset
        elif status == 200:
            # A server may ignore Range.  Restart the temporary file instead of
            # silently appending a complete response to it.
            expected_remaining = content_length
            remote_total = content_length
            if remote_total is not None and remote_total > spec.max_bytes:
                raise DownloadError(f"Fichier distant trop grand: {spec.name}")
            mode = "wb"
            starting_size = 0
        else:
            raise DownloadError(f"Statut HTTP inattendu {status} pour {spec.name}")

        written = 0
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if starting_size + written > spec.max_bytes:
                    raise DownloadError(f"Limite de taille depassee pour {spec.name}")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())

    if expected_remaining is not None and written != expected_remaining:
        raise DownloadError(
            f"Telechargement incomplet pour {spec.name}: {written}/{expected_remaining} octets"
        )

    final_size = partial.stat().st_size
    if final_size <= 0 or final_size > spec.max_bytes:
        raise DownloadError(f"Taille finale invalide pour {spec.name}: {final_size} octets")
    if remote_total is not None and final_size != remote_total:
        raise DownloadError(
            f"Taille finale incoherente pour {spec.name}: {final_size}/{remote_total} octets"
        )

    digest = _sha256_file(partial)
    if spec.expected_sha256 and digest != spec.expected_sha256:
        # The temporary artifact is unsafe to resume because it is complete but
        # does not match the upstream immutable object.
        partial.unlink(missing_ok=True)
        raise DownloadError(f"SHA-256 invalide pour {spec.name}")
    os.replace(partial, target)
    return {"name": spec.name, "bytes": final_size, "sha256": digest, "status": "present"}


def _write_manifest(output_dir: Path, files: list[dict[str, object]]) -> Path:
    total = sum(int(item["bytes"]) for item in files if item.get("status") == "present")
    if total > TOTAL_MAX_BYTES:
        raise DownloadError(f"La taille totale depasse la limite de {TOTAL_MAX_BYTES} octets")
    manifest = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_bytes": total,
        "files": files,
    }
    target = output_dir / MANIFEST_NAME
    temporary = output_dir / f".{MANIFEST_NAME}.tmp"
    _check_regular_file(target, MANIFEST_NAME)
    _check_regular_file(temporary, f".{MANIFEST_NAME}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return target


def download_all(output_dir: Path, *, dry_run: bool = False, opener=None) -> dict[str, object]:  # noqa: ANN001
    _validate_specs()
    output_dir = Path(output_dir).expanduser()
    _inspect_output_dir(output_dir)
    plan = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "output_dir": str(output_dir.resolve(strict=False)),
        "max_total_bytes": TOTAL_MAX_BYTES,
        "files": [
            {
                "name": spec.name,
                "required": spec.required,
                "max_bytes": spec.max_bytes,
                "url": _download_url(spec.name),
            }
            for spec in FILE_SPECS
        ],
    }
    if dry_run:
        return plan

    output_dir.mkdir(parents=True, exist_ok=True)
    _inspect_output_dir(output_dir)
    records: list[dict[str, object]] = []
    running_total = 0
    for spec in FILE_SPECS:
        record = _download_one(output_dir, spec, opener=opener)
        if record is None:
            records.append({"name": spec.name, "status": "not-present"})
            continue
        running_total += int(record["bytes"])
        if running_total > TOTAL_MAX_BYTES:
            raise DownloadError(f"La taille totale depasse la limite de {TOTAL_MAX_BYTES} octets")
        records.append(record)
    manifest = _write_manifest(output_dir, records)
    return {**plan, "manifest": str(manifest), "downloaded": records}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Telecharge de facon sure la revision epinglee de Granite 3.3 2B Instruct."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Dossier local du modele")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le plan fixe sans creer de dossier ni ouvrir le reseau",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = download_all(args.output_dir, dry_run=args.dry_run)
    except (DownloadError, OSError) as exc:
        print(f"ERREUR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
