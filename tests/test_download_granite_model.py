from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.request


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_granite_model.py"
SPEC = importlib.util.spec_from_file_location("download_granite_model", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, *, url: str, status: int = 200, headers=None):
        super().__init__(body)
        self._url = url
        self.status = status
        self.headers = dict(headers or {})

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status


class QueueOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout=None):  # noqa: ANN001
        self.requests.append((request, timeout))
        if not self.responses:
            raise AssertionError("unexpected network request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def fixture_spec(body: bytes, *, required: bool = True, max_bytes: int = 1024):
    return downloader.FileSpec(
        "config.json",
        max_bytes=max_bytes,
        required=required,
        expected_sha256=hashlib.sha256(body).hexdigest(),
    )


class GraniteDownloaderTests(unittest.TestCase):
    def test_dry_run_does_not_create_directory_or_open_network(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "not-created"
            opener = QueueOpener()
            result = downloader.download_all(output, dry_run=True, opener=opener)

            self.assertFalse(output.exists())
            self.assertEqual(opener.requests, [])
            self.assertEqual(result["revision"], downloader.REVISION)
            self.assertEqual(len(result["files"]), len(downloader.FILE_SPECS))

    def test_fresh_download_is_atomic_and_manifest_contains_sha256(self):
        body = b'{"model_type":"granite"}'
        spec = fixture_spec(body)
        url = downloader._download_url(spec.name)
        opener = QueueOpener(
            FakeResponse(body, url=url, headers={"Content-Length": str(len(body))})
        )
        with tempfile.TemporaryDirectory() as parent, mock.patch.object(
            downloader, "FILE_SPECS", (spec,)
        ):
            output = Path(parent) / "model"
            result = downloader.download_all(output, opener=opener)

            self.assertEqual((output / spec.name).read_bytes(), body)
            self.assertFalse((output / f"{spec.name}.part").exists())
            manifest = json.loads((output / downloader.MANIFEST_NAME).read_text("utf-8"))
            self.assertEqual(manifest["revision"], downloader.REVISION)
            self.assertEqual(manifest["total_bytes"], len(body))
            self.assertEqual(manifest["files"][0]["sha256"], hashlib.sha256(body).hexdigest())
            self.assertEqual(result["manifest"], str(output / downloader.MANIFEST_NAME))

    def test_resume_sends_range_and_appends_valid_partial_response(self):
        body = b"abcdef"
        spec = fixture_spec(body)
        url = downloader._download_url(spec.name)
        opener = QueueOpener(
            FakeResponse(
                b"def",
                url=url,
                status=206,
                headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6"},
            )
        )
        with tempfile.TemporaryDirectory() as parent, mock.patch.object(
            downloader, "FILE_SPECS", (spec,)
        ):
            output = Path(parent)
            (output / f"{spec.name}.part").write_bytes(b"abc")
            downloader.download_all(output, opener=opener)

            self.assertEqual((output / spec.name).read_bytes(), body)
            self.assertEqual(opener.requests[0][0].get_header("Range"), "bytes=3-")

    def test_server_ignoring_range_restarts_partial_file(self):
        body = b"fresh"
        spec = fixture_spec(body)
        url = downloader._download_url(spec.name)
        opener = QueueOpener(
            FakeResponse(body, url=url, status=200, headers={"Content-Length": "5"})
        )
        with tempfile.TemporaryDirectory() as parent, mock.patch.object(
            downloader, "FILE_SPECS", (spec,)
        ):
            output = Path(parent)
            (output / f"{spec.name}.part").write_bytes(b"stale")
            downloader.download_all(output, opener=opener)

            self.assertEqual((output / spec.name).read_bytes(), body)

    def test_complete_partial_is_finalized_after_range_416(self):
        body = b"already-complete"
        spec = fixture_spec(body)
        url = downloader._download_url(spec.name)
        error = urllib.error.HTTPError(
            url,
            416,
            "range not satisfiable",
            {"Content-Range": f"bytes */{len(body)}"},
            None,
        )
        opener = QueueOpener(error)
        with tempfile.TemporaryDirectory() as parent, mock.patch.object(
            downloader, "FILE_SPECS", (spec,)
        ):
            output = Path(parent)
            (output / f"{spec.name}.part").write_bytes(body)
            downloader.download_all(output, opener=opener)

            self.assertEqual((output / spec.name).read_bytes(), body)
            self.assertFalse((output / f"{spec.name}.part").exists())

    def test_rejects_oversized_response_before_writing(self):
        spec = fixture_spec(b"tiny", max_bytes=4)
        url = downloader._download_url(spec.name)
        opener = QueueOpener(
            FakeResponse(b"12345", url=url, headers={"Content-Length": "5"})
        )
        with tempfile.TemporaryDirectory() as parent, mock.patch.object(
            downloader, "FILE_SPECS", (spec,)
        ):
            with self.assertRaisesRegex(downloader.DownloadError, "trop grand"):
                downloader.download_all(Path(parent), opener=opener)
            self.assertFalse((Path(parent) / spec.name).exists())

    def test_bad_sha_never_becomes_final_file(self):
        spec = fixture_spec(b"expected")
        url = downloader._download_url(spec.name)
        wrong = b"wrong"
        opener = QueueOpener(
            FakeResponse(wrong, url=url, headers={"Content-Length": str(len(wrong))})
        )
        with tempfile.TemporaryDirectory() as parent, mock.patch.object(
            downloader, "FILE_SPECS", (spec,)
        ):
            with self.assertRaisesRegex(downloader.DownloadError, "SHA-256"):
                downloader.download_all(Path(parent), opener=opener)
            self.assertFalse((Path(parent) / spec.name).exists())
            self.assertFalse((Path(parent) / f"{spec.name}.part").exists())

    def test_optional_404_is_recorded_without_failure(self):
        spec = fixture_spec(b"unused", required=False)
        request_url = downloader._download_url(spec.name)
        error = urllib.error.HTTPError(request_url, 404, "not found", {}, None)
        opener = QueueOpener(error)
        with tempfile.TemporaryDirectory() as parent, mock.patch.object(
            downloader, "FILE_SPECS", (spec,)
        ):
            downloader.download_all(Path(parent), opener=opener)
            manifest = json.loads(
                (Path(parent) / downloader.MANIFEST_NAME).read_text("utf-8")
            )
            self.assertEqual(manifest["files"], [{"name": spec.name, "status": "not-present"}])

    def test_rejects_unrelated_populated_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent)
            (output / "private-notes.txt").write_text("keep me", encoding="utf-8")
            with self.assertRaisesRegex(downloader.DownloadError, "sans rapport"):
                downloader.download_all(output, dry_run=True)

    def test_rejects_local_path_traversal(self):
        with tempfile.TemporaryDirectory() as parent:
            with self.assertRaisesRegex(downloader.DownloadError, "non autorise"):
                downloader._safe_target(Path(parent), "../config.json")

    def test_rejects_final_response_on_untrusted_host(self):
        body = b"safe?"
        spec = fixture_spec(body)
        opener = QueueOpener(
            FakeResponse(body, url="https://evil.example/config.json", headers={"Content-Length": "5"})
        )
        with tempfile.TemporaryDirectory() as parent, mock.patch.object(
            downloader, "FILE_SPECS", (spec,)
        ):
            with self.assertRaisesRegex(downloader.DownloadError, "Redirection non autorisee"):
                downloader.download_all(Path(parent), opener=opener)
            self.assertFalse((Path(parent) / spec.name).exists())

    def test_redirect_handler_rejects_http_userinfo_and_wrong_path(self):
        handler = downloader.SafeRedirectHandler("config.json")
        request = urllib.request.Request(downloader._download_url("config.json"))
        bad_urls = (
            "http://huggingface.co/ibm-granite/granite-3.3-2b-instruct/resolve/"
            f"{downloader.REVISION}/config.json",
            "https://user:password@huggingface.co/anything",
            "https://huggingface.co/other/repo/resolve/main/config.json",
            "https://cas-bridge.xethub.hf.co/xet-bridge-us/../secret",
        )
        for bad_url in bad_urls:
            with self.subTest(url=bad_url), self.assertRaises(downloader.DownloadError):
                handler.redirect_request(request, None, 302, "Found", {}, bad_url)

    def test_accepts_scoped_huggingface_and_xet_urls(self):
        name = "config.json"
        downloader._validate_remote_url(downloader._download_url(name), name)
        downloader._validate_remote_url(
            "https://cas-bridge.xethub.hf.co/xet-bridge-us/object-id?signature=fixed",
            name,
        )


if __name__ == "__main__":
    unittest.main()
