#!/usr/bin/env python3
"""Self-contained tests for the parts that do not need a running WebUI.

Run directly (``python tests/test_extended_lora_details.py``) or under pytest.
The Gradio layer is exercised separately against a real Forge checkout; these
cover CSV parsing, hash-based matching, the Civitai scan pipeline (with a stub
client) and the background job manager.
"""

from __future__ import annotations

import csv
import hashlib
import json
import struct
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extended_lora_details import civitai, hashing, images, jobs  # noqa: E402
from extended_lora_details.library import Library, parse_row  # noqa: E402


def make_safetensors(path: Path, payload: bytes, metadata: dict | None = None) -> str:
    """Write a minimal valid safetensors file; returns its full-file SHA256."""
    header = json.dumps({"__metadata__": metadata or {}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------- hashing


def test_hashes_match_reference_digests():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.safetensors"
        expected = make_safetensors(path, b"PAYLOAD", {"k": "v"})
        cache = hashing.HashCache(Path(tmp) / "cache.json")

        assert hashing.sha256_file(path, cache=cache) == expected
        assert hashing.addnet_sha256_file(path, cache=cache) == hashlib.sha256(b"PAYLOAD").hexdigest()

        # second read comes from the cache, and survives a save/load round trip
        assert cache.peek(path, "sha256") == expected
        cache.save()
        reloaded = hashing.HashCache(Path(tmp) / "cache.json")
        assert reloaded.peek(path, "sha256") == expected

        # touching the file invalidates the entry
        path.write_bytes(path.read_bytes() + b"more")
        assert reloaded.peek(path, "sha256") is None


def test_hashing_a_missing_file_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        cache = hashing.HashCache(Path(tmp) / "cache.json")
        assert hashing.sha256_file(Path(tmp) / "nope.safetensors", cache=cache) is None


# --------------------------------------------------------------------- parsing


def test_parse_row_recognises_aliased_columns():
    record = parse_row(
        {
            "File Name": "sub/My Lora.safetensors",
            "SHA-256": "AB" * 32,
            "Model Name": "Cool Style",
            "Trained Words": "one, two; three",
            "prompt_2": "second",
            "prompt_1": "first",
            "image_url_1": "https://img/one.jpeg",
            "Notes": "kept as an extra",
            "Blank": "",
        },
        source="x.csv",
        row_number=2,
    )
    assert record.sha256 == "ab" * 32
    assert record.stem == "my lora"
    assert record.rel_path == "sub/my lora.safetensors"
    assert record.model_name == "Cool Style"
    assert record.trigger_words == ("one", "two", "three")
    assert record.prompts == ("first", "second")          # ordered by column index
    assert record.entries[0].image_url == "https://img/one.jpeg"
    assert record.entries[1].image_url == ""
    assert record.extras == {"Notes": "kept as an extra"}  # unknown columns survive
    assert "Blank" not in record.extras


def test_parse_row_ignores_a_row_with_nothing_in_it():
    assert parse_row({"a": "", "b": None}, source="x.csv", row_number=2) is None


def test_prompt_columns_sort_numerically_not_lexically():
    row = {f"positive_prompt_{i}": f"p{i}" for i in (1, 2, 10, 11)}
    record = parse_row(row, source="x.csv", row_number=2)
    assert record.prompts == ("p1", "p2", "p10", "p11")


def test_an_empty_prompt_cell_does_not_shift_images_onto_the_wrong_prompt():
    record = parse_row(
        {
            "positive_prompt_1": "first",
            "prompt_image_url_1": "https://img/1.jpeg",
            "positive_prompt_2": "",                       # dropped
            "prompt_image_url_2": "https://img/2.jpeg",    # must go with it
            "positive_prompt_3": "third",
            "prompt_image_url_3": "https://img/3.jpeg",
            "prompt_image_id_3": "333",
        },
        source="x.csv",
        row_number=2,
    )
    assert record.prompts == ("first", "third")
    assert [entry.image_url for entry in record.entries] == ["https://img/1.jpeg", "https://img/3.jpeg"]
    assert record.entries[1].image_id == "333"


# -------------------------------------------------------------------- matching


def _library_with_rows(directory: Path, rows: list[dict], columns: list[str], name="lib.csv") -> Library:
    with (directory / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    library = Library(directory)
    library.reload()
    return library


def test_lookup_prefers_sha256_then_falls_back():
    sha = "cd" * 32
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        library = _library_with_rows(
            directory,
            [
                {"safetensor_file": "sub/Hashed.safetensors", "sha256": sha, "civitai_model_name": "Hashed"},
                {"safetensor_file": "sub/Named.safetensors", "sha256": "", "civitai_model_name": "Named"},
            ],
            ["safetensor_file", "sha256", "civitai_model_name"],
        )

        records, how = library.lookup(sha256=sha.upper())
        assert how == "SHA256" and records[0].model_name == "Hashed"

        # a 12-char prefix of a full digest also resolves
        records, how = library.lookup(short_hashes=[sha[:12]])
        assert how == "short hash" and records[0].model_name == "Hashed"

        # backslash paths from a Windows CSV still match
        records, how = library.lookup(rel_paths=["sub\\Named.safetensors"])
        assert how == "file path" and records[0].model_name == "Named"

        # bare file name, and then stem-only, as last resorts
        records, how = library.lookup(filename="/anywhere/Named.safetensors")
        assert how == "file name" and records[0].model_name == "Named"

        assert library.lookup(sha256="00" * 32, filename="absent.safetensors") == ([], "")


def test_matches_are_pooled_across_csv_files():
    sha = "ef" * 32
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _library_with_rows(
            directory,
            [{"safetensor_file": "a.safetensors", "sha256": sha, "positive_prompt_1": "from one"}],
            ["safetensor_file", "sha256", "positive_prompt_1"],
            name="one.csv",
        )
        library = _library_with_rows(
            directory,
            [{"safetensor_file": "a.safetensors", "sha256": sha, "positive_prompt_1": "from two"}],
            ["safetensor_file", "sha256", "positive_prompt_1"],
            name="two.csv",
        )
        records, how = library.lookup(sha256=sha)
        assert how == "SHA256" and len(records) == 2
        assert {record.source for record in records} == {"one.csv", "two.csv"}


def test_a_broken_csv_does_not_take_the_library_down():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "empty.csv").write_text("", encoding="utf-8")
        (directory / "binary.csv").write_bytes(b"\xff\xfe\x00\x01\x00")
        library = _library_with_rows(
            directory,
            [{"safetensor_file": "good.safetensors", "civitai_model_name": "Good"}],
            ["safetensor_file", "civitai_model_name"],
        )
        assert library.lookup(filename="good.safetensors")[0][0].model_name == "Good"


def test_remove_csv_refuses_to_escape_the_library_directory():
    with tempfile.TemporaryDirectory() as tmp:
        library = Library(Path(tmp))
        library.reload()
        ok, message = library.remove_csv("../../etc/passwd")
        assert not ok and "outside the library directory" in message


# ----------------------------------------------------------------- base models


def test_base_model_filter_is_forgiving_about_spelling():
    assert civitai.matches_base_model("Krea2", "Krea 2")
    assert civitai.matches_base_model("Krea-2-Turbo", "Krea 2")
    assert not civitai.matches_base_model("Flux.1 D", "Krea 2")
    assert civitai.matches_base_model("anything at all", "")  # empty filter accepts everything


def test_a_version_whose_hashes_disagree_is_rejected():
    version = {"files": [{"name": "f", "hashes": {"SHA256": "11" * 32}}]}
    allowed, reason = civitai.version_file_hash_state(version, "22" * 32)
    assert not allowed and "inconsistent" in reason

    allowed, _ = civitai.version_file_hash_state(version, "11" * 32)
    assert allowed

    # no SHA256 exposed at all: the by-hash lookup is the identity evidence
    allowed, _ = civitai.version_file_hash_state({"files": [{"name": "f"}]}, "11" * 32)
    assert allowed


def test_the_negative_prompt_is_never_used_as_a_positive_one():
    assert civitai.extract_positive_prompt({"meta": {"negativePrompt": "bad"}}) == ""
    assert civitai.extract_positive_prompt({"meta": {"prompt": "good", "negativePrompt": "bad"}}) == "good"
    assert civitai.extract_positive_prompt({"meta": json.dumps({"prompt": "stringified"})}) == "stringified"


# -------------------------------------------------------------------- scanning


class StubClient:
    """Stands in for CivitaiClient; records every hash it was asked about."""

    def __init__(self, known: set[str], **_kwargs):
        self.known = known
        self.lookups: list[str] = []

    def close(self):
        pass

    def version_by_hash(self, sha256):
        self.lookups.append(sha256)
        if sha256 not in self.known:
            return None
        return {
            "id": 1,
            "modelId": 2,
            "name": "v1",
            "baseModel": "SDXL",
            "trainedWords": ["trig"],
            "files": [{"name": "f", "hashes": {"SHA256": sha256}}],
            "model": {"name": "Stub Model", "type": "LORA"},
        }

    def model_info(self, version):
        return version["model"]

    def gallery_prompts(self, _version_id, *, embedded_images, max_images):
        return (civitai.PromptImage("a prompt", "https://image.example/1.jpeg", "1"),), None


def _stub_scan(monkey_target, known):
    created = {}

    def factory(**kwargs):
        client = StubClient(known, **kwargs)
        created["client"] = client
        return client

    original = civitai.CivitaiClient
    civitai.CivitaiClient = factory
    return original, created


def test_scan_writes_a_hash_column_and_refetches_only_new_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hashing.CACHE = hashing.HashCache(root / "hashes.json")
        loras = root / "loras"
        sha_a = make_safetensors(loras / "alpha.safetensors", b"A")
        sha_b = make_safetensors(loras / "sub" / "beta.safetensors", b"B")
        output = root / "out.csv"

        original, created = _stub_scan(civitai, {sha_a})
        try:
            report = civitai.scan_folder(loras, output, log=lambda _m: None)
            assert report.total == 2 and report.resolved == 1
            assert len(created["client"].lookups) == 2

            written = list(csv.DictReader(output.open(encoding="utf-8")))
            assert {row["safetensor_file"] for row in written} == {"alpha.safetensors", "sub/beta.safetensors"}
            alpha = next(row for row in written if row["safetensor_file"] == "alpha.safetensors")
            assert alpha["sha256"] == sha_a, "the SHA256 column is what makes hash lookup work"
            assert alpha["civitai_model_name"] == "Stub Model"
            assert alpha["positive_prompt_1"] == "a prompt"
            assert alpha["prompt_image_url_1"] == "https://image.example/1.jpeg"
            assert alpha["prompt_image_id_1"] == "1"

            # the library indexes what was just written, by hash
            library = Library(root)
            library.reload()
            records, how = library.lookup(sha256=sha_a)
            assert how == "SHA256" and records[0].model_name == "Stub Model"
            assert records[0].entries[0].image_url == "https://image.example/1.jpeg"

            # a refresh of the same folder costs nothing and keeps every row
            report = civitai.scan_folder(loras, output, skip_hashes=library.known_sha256(), log=lambda _m: None)
            assert created["client"].lookups == [], "a refresh must not re-query anything already recorded"
            assert report.skipped == 2 and len(report.rows) == 2

            # opting in to retries picks the unresolved one back up, and only it
            retry = library.known_sha256(resolved_only=True)
            report = civitai.scan_folder(loras, output, skip_hashes=retry, log=lambda _m: None)
            assert created["client"].lookups == [sha_b], created["client"].lookups

            # only a genuinely new file is fetched
            sha_c = make_safetensors(loras / "gamma.safetensors", b"C")
            library.reload()
            report = civitai.scan_folder(loras, output, skip_hashes=library.known_sha256(), log=lambda _m: None)
            assert created["client"].lookups == [sha_c], "only the genuinely new LoRA is looked up"
            assert len(report.rows) == 3
        finally:
            civitai.CivitaiClient = original


def test_rewriting_from_scratch_never_drops_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hashing.CACHE = hashing.HashCache(root / "hashes.json")
        loras = root / "loras"
        sha_a = make_safetensors(loras / "alpha.safetensors", b"A")
        output = root / "out.csv"

        original, created = _stub_scan(civitai, {sha_a})
        try:
            civitai.scan_folder(loras, output, log=lambda _m: None)
            # "skip what we have" + "rewrite from scratch" would otherwise lose rows
            report = civitai.scan_folder(
                loras,
                output,
                skip_hashes={sha_a},
                keep_existing_rows=False,
                log=lambda _m: None,
            )
            assert report.skipped == 0 and len(report.rows) == 1
        finally:
            civitai.CivitaiClient = original


def test_the_csv_is_replaced_only_after_a_complete_write():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "out.csv"
        civitai.write_csv(output, [{"safetensor_file": "a", "sha256": "b"}])
        before = output.read_text(encoding="utf-8")
        class Unwritable:
            def __str__(self):
                raise RuntimeError("boom halfway through the write")

        try:
            civitai.write_csv(output, [{"safetensor_file": "a"}, {"safetensor_file": Unwritable()}])
        except RuntimeError:
            pass
        else:
            raise AssertionError("the write should have failed")
        assert output.read_text(encoding="utf-8") == before
        assert not list(Path(tmp).glob("*.tmp")), "the temp file must be cleaned up"


# ---------------------------------------------------------------------- images


class _ImageServer:
    """A tiny local HTTP server, so the download path is exercised for real."""

    BODY = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    def __enter__(self):
        import http.server
        import socketserver
        import threading

        body = self.BODY

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path == "/missing":
                    self.send_response(404)
                    self.end_headers()
                    return
                content_type = "image/webp" if self.path == "/noext" else "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()


def test_an_image_downloads_once_and_reappears_as_new_when_deleted():
    with tempfile.TemporaryDirectory() as tmp, _ImageServer() as server:
        directory = Path(tmp)
        url = f"{server.base}/cat.jpeg"

        assert images.find_local(url, "42", directory=directory) is None

        first = images.download(url, "42", directory=directory)
        assert first.ok and not first.skipped
        assert first.path.name == "civitai-42.jpeg", first.path.name

        second = images.download(url, "42", directory=directory)
        assert second.ok and second.skipped, "an already-present image must not be re-fetched"

        # the whole point: presence is answered from disk, never from a manifest
        first.path.unlink()
        assert images.find_local(url, "42", directory=directory) is None
        third = images.download(url, "42", directory=directory)
        assert third.ok and not third.skipped


def test_image_names_are_stable_without_a_civitai_id():
    with tempfile.TemporaryDirectory() as tmp, _ImageServer() as server:
        directory = Path(tmp)
        url = f"{server.base}/noext"

        result = images.download(url, directory=directory)
        assert result.ok
        assert result.path.name.startswith("url-") and result.path.suffix == ".webp"
        assert images.find_local(url, directory=directory) == result.path
        assert images.stem_for(url) == images.stem_for(url), "the name must not drift between runs"


def test_image_download_refuses_what_it_should():
    with tempfile.TemporaryDirectory() as tmp, _ImageServer() as server:
        directory = Path(tmp)

        assert not images.download("", directory=directory).ok
        assert "no image URL" in images.download("", directory=directory).message

        refused = images.download("file:///etc/passwd", directory=directory)
        assert not refused.ok and "non-HTTP" in refused.message

        missing = images.download(f"{server.base}/missing", "7", directory=directory)
        assert not missing.ok and "404" in missing.message

        capped = images.download(f"{server.base}/big.jpeg", "8", directory=directory, max_bytes=8)
        assert not capped.ok and "limit" in capped.message

        assert not list(directory.glob("*.part")), "a failed download must leave no partial file"
        assert images.stats(directory)["files"] == 0


def test_purge_empties_the_image_folder():
    with tempfile.TemporaryDirectory() as tmp, _ImageServer() as server:
        directory = Path(tmp)
        images.download(f"{server.base}/a.jpeg", "1", directory=directory)
        images.download(f"{server.base}/b.jpeg", "2", directory=directory)
        assert images.stats(directory)["files"] == 2

        removed, message = images.purge(directory)
        assert removed == 2 and "Deleted 2" in message
        assert images.stats(directory)["files"] == 0


# ------------------------------------------------------------------------ jobs


def test_jobs_report_progress_failure_and_cancellation():
    manager = jobs.JobManager()

    def counting(job):
        for index in range(4):
            if job.cancelled:
                break
            job.set_progress(index + 1, 4, f"step {index + 1}")
            job.log_line(f"step {index + 1}")
            time.sleep(0.02)
        return "counted"

    manager.submit("test", "counting", counting)
    deadline = time.time() + 5
    while manager.snapshots()[0].active and time.time() < deadline:
        time.sleep(0.02)

    snapshot = manager.snapshots()[0]
    assert snapshot.state == jobs.DONE and snapshot.percent == 100
    assert snapshot.summary == "counted" and len(snapshot.log) == 4

    def failing(_job):
        raise ValueError("nope")

    manager.submit("test", "failing", failing)
    deadline = time.time() + 5
    while manager.snapshots()[0].active and time.time() < deadline:
        time.sleep(0.02)
    assert manager.snapshots()[0].state == jobs.FAILED
    assert "ValueError: nope" in manager.snapshots()[0].error

    def looping(job):
        while not job.cancelled:
            time.sleep(0.02)
        return "stopped"

    manager.submit("test", "looping", looping)
    deadline = time.time() + 5
    while not manager.has_active() and time.time() < deadline:
        time.sleep(0.02)
    assert manager.cancel_all() == 1
    deadline = time.time() + 5
    while manager.has_active() and time.time() < deadline:
        time.sleep(0.02)
    assert manager.snapshots()[0].state == jobs.CANCELLED

    assert manager.clear_finished() == 3 and manager.snapshots() == []


def test_bulk_image_download_reports_what_it_did():
    with tempfile.TemporaryDirectory() as tmp, _ImageServer() as server:
        original_dir = images.image_dir
        images.image_dir = lambda: Path(tmp)
        try:
            job = jobs.submit_image_downloads(
                [
                    (f"{server.base}/a.jpeg", "1"),
                    (f"{server.base}/a.jpeg", "1"),   # duplicate -> already present
                    (f"{server.base}/missing", "2"),  # 404 -> failed
                    ("", "3"),                        # no url -> dropped before queueing
                ],
                label="Demo",
            )
            deadline = time.time() + 15
            while jobs.MANAGER.get(job.id).snapshot().active and time.time() < deadline:
                time.sleep(0.05)

            snapshot = jobs.MANAGER.get(job.id).snapshot()
            assert snapshot.state == jobs.DONE, snapshot.state
            assert "1 downloaded" in snapshot.summary
            assert "1 already present" in snapshot.summary
            assert "1 failed" in snapshot.summary
        finally:
            images.image_dir = original_dir


def test_jobs_run_one_at_a_time():
    manager = jobs.JobManager()
    concurrent = []
    running = []

    def watcher(_job):
        running.append(1)
        concurrent.append(len(running))
        time.sleep(0.05)
        running.pop()
        return ""

    for _ in range(4):
        manager.submit("test", "watcher", watcher)

    deadline = time.time() + 10
    while manager.has_active() and time.time() < deadline:
        time.sleep(0.02)
    assert concurrent == [1, 1, 1, 1], concurrent


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
