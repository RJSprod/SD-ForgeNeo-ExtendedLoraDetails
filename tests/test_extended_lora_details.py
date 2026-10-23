#!/usr/bin/env python3
"""Self-contained tests for the parts that do not need a running WebUI.

Run directly (``python tests/test_extended_lora_details.py``) or under pytest.
The Gradio layer is exercised separately against a real Forge checkout; these
cover CSV parsing, hash-based matching, the Civitai scan pipeline (with a stub
client) and the background job manager.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import html as html_module
import json
import os
import re
import stat
import struct
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extended_lora_details import civitai, descriptions, hashing, images, jobs, layouts, network_filter, preset_folders, render, state  # noqa: E402
from extended_lora_details.library import Library, parse_row  # noqa: E402


@contextlib.contextmanager
def patched_store(store, *, preset: str):
    """Point the filter at a throwaway store and a chosen UI Preset."""
    original_store = preset_folders.get_store
    original_preset = preset_folders.current_preset
    preset_folders.get_store = lambda: store
    preset_folders.current_preset = lambda: preset
    try:
        yield store
    finally:
        preset_folders.get_store = original_store
        preset_folders.current_preset = original_preset


@contextlib.contextmanager
def patched_state(path):
    """Point the remembered-values helpers at a throwaway file."""
    store = state.LastUsedStore(path)
    original = state.get_store
    state.get_store = lambda: store
    try:
        yield store
    finally:
        state.get_store = original


@contextlib.contextmanager
def patched_options(**values):
    """Stand in for ``modules.shared.opts``, so option-gated code can be tested."""
    import types

    originals = {name: sys.modules.get(name) for name in ("modules", "modules.shared")}
    package = types.ModuleType("modules")
    shared = types.SimpleNamespace(opts=types.SimpleNamespace(**values))
    package.shared = shared
    sys.modules["modules"] = package
    sys.modules["modules.shared"] = shared
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


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
    """Stands in for CivitaiClient; records every hash it was asked about.

    Shaped like the real API: the version response embeds a model record with a
    name and type but no description, so asking for one costs a model-page fetch.
    """

    def __init__(self, known: set[str], **_kwargs):
        self.known = known
        self.lookups: list[str] = []
        self.model_pages: list[int] = []

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
            "description": "<p>Version <b>notes</b>.</p>",
            "trainedWords": ["trig"],
            "files": [{"name": "f", "hashes": {"SHA256": sha256}}],
            "model": {"name": "Stub Model", "type": "LORA"},
        }

    def model_info(self, version, *, with_description=False):
        if not with_description:
            return version["model"]
        self.model_pages.append(version["modelId"])
        return {**version["model"], "description": "<p>About <b>this</b> model.</p>", "creator": {"username": "maker"}}

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


# --------------------------------------------------------------- descriptions


def test_html_descriptions_become_readable_text():
    text = descriptions.html_to_text(
        "<h2>My LoRA</h2><p>Trained on <b>500</b> images.<br>Use <code>trigger</code>.</p>"
        "<ul><li>one</li><li>two <a href=\"/models/5\">the page</a></li></ul>"
        "<script>steal()</script><img src=\"https://img.example/x.png\">"
    )
    assert text == (
        "My LoRA\n\n"
        "Trained on 500 images.\n"
        "Use trigger.\n\n"
        "• one\n"
        "• two the page (https://civitai.com/models/5)"
    ), text
    assert "<" not in text, "no markup may survive the conversion"


def test_description_conversion_keeps_lists_and_drops_nothing_else():
    assert descriptions.html_to_text("<ol><li>first</li><li>second<ul><li>nested</li></ul></li><li>third</li></ol>") == (
        "1. first\n2. second\n  • nested\n3. third"
    )
    # plain text passes through with its paragraphs and entities intact
    assert descriptions.html_to_text("Plain.\n\nSecond &amp; last.") == "Plain.\n\nSecond & last."
    assert descriptions.html_to_text(None) == "" and descriptions.html_to_text("   ") == ""
    # a link whose text is already the URL is not spelled out twice
    assert descriptions.html_to_text('<a href="https://x.example/y">https://x.example/y</a>') == "https://x.example/y"


def test_a_long_description_is_capped_visibly():
    capped = descriptions.html_to_text("<p>" + "word " * 5000 + "</p>", limit=100)
    assert len(capped) == 100 and capped.endswith("…")


def test_rendered_descriptions_escape_text_and_link_urls():
    html = render.rich_text("A <script>alert(1)</script> line\nsecond\n\nSee https://civitai.com/models/7?a=1&b=2.")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert html.count("<p>") == 2 and "<br>" in html
    assert '<a href="https://civitai.com/models/7?a=1&amp;b=2"' in html
    assert html.endswith(".</p>"), "sentence punctuation stays outside the link"


def test_description_columns_are_read_and_not_repeated_as_extras():
    record = parse_row(
        {
            "sha256": "cd" * 32,
            "model_description": "About the model",
            "version_description": "About the version",
            "Notes": "kept",
        },
        source="x.csv",
        row_number=2,
    )
    assert record.description == "About the model"
    assert record.version_description == "About the version"
    assert record.has_description
    assert record.extras == {"Notes": "kept"}

    # an unrelated tool's column naming works just as well
    aliased = parse_row({"sha256": "ef" * 32, "Description": "text"}, source="y.csv", row_number=2)
    assert aliased.description == "text" and aliased.extras == {}

    assert not parse_row({"sha256": "ab" * 32, "civitai_model_name": "x"}, source="z.csv", row_number=2).has_description


def test_a_fetch_stores_the_description_and_backfills_rows_that_lack_one():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hashing.CACHE = hashing.HashCache(root / "hashes.json")
        loras = root / "loras"
        sha_a = make_safetensors(loras / "alpha.safetensors", b"A")
        sha_b = make_safetensors(loras / "beta.safetensors", b"B")   # Civitai knows nothing about it
        output = root / "out.csv"

        original, created = _stub_scan(civitai, {sha_a})
        try:
            civitai.scan_folder(loras, output, log=lambda _m: None)
            # a fresh stub client is built per scan, so each assertion below
            # reads the one that scan used
            assert created["client"].model_pages == [2], "the description costs one model-page fetch"

            row = next(r for r in csv.DictReader(output.open(encoding="utf-8")) if r["sha256"] == sha_a)
            assert row["model_description"] == "About this model."
            assert row["version_description"] == "Version notes."
            assert row["civitai_creator"] == "maker", "the model page names the creator"

            library = Library(root)
            library.reload()
            records, _how = library.lookup(sha256=sha_a)
            assert records[0].description == "About this model."

            # a refresh has nothing to fill in, so it costs no requests at all
            civitai.scan_folder(
                loras,
                output,
                skip_hashes=library.known_sha256(require_description=True),
                log=lambda _m: None,
            )
            assert created["client"].lookups == [], "nothing is missing, so nothing is re-queried"

            # a row written before descriptions existed is looked up again...
            stripped = list(csv.DictReader(output.open(encoding="utf-8")))
            for entry in stripped:
                entry["model_description"] = entry["version_description"] = ""
            civitai.write_csv(output, stripped)
            library.reload()
            assert library.known_sha256() == {sha_a, sha_b}, "plain refresh still skips everything recorded"

            civitai.scan_folder(
                loras,
                output,
                skip_hashes=library.known_sha256(require_description=True),
                log=lambda _m: None,
            )
            assert created["client"].lookups == [sha_a], "only the resolved row missing its text is re-fetched"

            # ...and the description comes back
            row = next(r for r in csv.DictReader(output.open(encoding="utf-8")) if r["sha256"] == sha_a)
            assert row["model_description"] == "About this model."
        finally:
            civitai.CivitaiClient = original


def test_turning_descriptions_off_costs_no_extra_request():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hashing.CACHE = hashing.HashCache(root / "hashes.json")
        loras = root / "loras"
        sha_a = make_safetensors(loras / "alpha.safetensors", b"A")
        output = root / "out.csv"

        original, created = _stub_scan(civitai, {sha_a})
        try:
            civitai.scan_folder(loras, output, fetch_descriptions=False, log=lambda _m: None)
            assert created["client"].model_pages == []
            row = next(iter(csv.DictReader(output.open(encoding="utf-8"))))
            assert row["model_description"] == "" and row["civitai_model_name"] == "Stub Model"
            assert row["civitai_creator"] == "", "only the model page names the creator"
        finally:
            civitai.CivitaiClient = original


def test_a_refetch_keeps_hand_written_columns_but_not_stale_prompts():
    fresh = {"safetensor_file": "a.safetensors", "sha256": "ab" * 32, "positive_prompt_1": "new"}
    existing = {
        "safetensor_file": "a.safetensors",
        "sha256": "ab" * 32,
        "civitai_model_name": "Old Name",
        "positive_prompt_1": "old",
        "positive_prompt_2": "dropped with the version it came from",
        "My Rating": "5 stars",
    }
    merged = civitai.merge_row(existing, fresh)
    assert merged["My Rating"] == "5 stars", "a column this tool does not own survives"
    assert merged["positive_prompt_1"] == "new"
    assert "positive_prompt_2" not in merged
    assert merged.get("civitai_model_name", "") == ""
    assert civitai.merge_row(None, fresh) == fresh


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



# ------------------------------------------------------- remembered values


def test_the_key_and_folder_of_the_last_fetch_are_remembered():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "loras"
        folder.mkdir()
        path = Path(tmp) / "last_used.json"

        with patched_state(path) as store:
            state.remember_api_key("  secret-key  ")
            state.remember_scan_folder(folder)

            assert state.remembered_api_key() == "secret-key"
            assert state.remembered_scan_folder() == str(folder)

            # a fresh reader sees both - which is all the next session does
            reread = state.LastUsedStore(path)
            assert reread.get(state.API_KEY) == "secret-key"
            assert reread.get(state.SCAN_FOLDER) == str(folder)

            # a run without a key never clears the one that worked
            state.remember_api_key("")
            assert store.get(state.API_KEY) == "secret-key"

        if os.name == "posix":
            assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, "the file holds an API key"


def test_a_folder_that_has_gone_away_is_not_offered_but_is_kept():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "loras"
        folder.mkdir()

        with patched_state(Path(tmp) / "last_used.json") as store:
            state.remember_scan_folder(folder)
            folder.rmdir()

            # an unplugged drive must not become a default that cannot be scanned
            assert state.remembered_scan_folder() == ""
            # but the choice is kept, and is the default again when it comes back
            assert store.get(state.SCAN_FOLDER) == str(folder)
            folder.mkdir()
            assert state.remembered_scan_folder() == str(folder)


def test_turning_remembering_off_hides_the_values_and_drops_the_key():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "loras"
        folder.mkdir()

        with patched_state(Path(tmp) / "last_used.json") as store:
            state.remember_api_key("secret-key")
            state.remember_scan_folder(folder)

            with patched_options(eld_remember_api_key=False, eld_remember_scan_folder=False):
                assert state.remembered_api_key() == ""
                assert state.remembered_scan_folder() == ""
                state.remember_api_key("another-key")
                assert store.get(state.API_KEY) == "secret-key", "nothing is stored while it is off"
                state.forget_api_key()  # what turning the setting off does

            assert state.remembered_api_key() == ""
            assert state.remembered_scan_folder() == str(folder), "only the key is forgotten"


def test_only_a_key_civitai_answered_counts_as_used():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hashing.CACHE = hashing.HashCache(root / "hashes.json")
        loras = root / "loras"
        sha_a = make_safetensors(loras / "alpha.safetensors", b"A")

        original, _created = _stub_scan(civitai, {sha_a})
        try:
            report = civitai.scan_folder(loras, root / "out.csv", log=lambda _m: None)
            assert report.api_answered, "an answered lookup is what proves the key was accepted"

            report = civitai.scan_folder(loras, root / "out.csv", skip_hashes={sha_a}, log=lambda _m: None)
            assert not report.api_answered, "a run that asked nothing proves nothing"

            class Rejecting(StubClient):
                def version_by_hash(self, sha256):
                    raise civitai.CivitaiError("Civitai returned HTTP 401; check the API key")

            civitai.CivitaiClient = lambda **kwargs: Rejecting(set(), **kwargs)
            report = civitai.scan_folder(loras, root / "rejected.csv", log=lambda _m: None)
            assert report.processed == 1 and not report.api_answered
        finally:
            civitai.CivitaiClient = original


# ------------------------------------------------------- preset folder filter


def make_network_tree(root: Path) -> dict:
    """A models folder with two top-level folders and one nested folder."""
    layout = {
        "root": root,
        "zit": root / "zit",
        "zit_faces": root / "zit" / "faces",
        "xl": root / "xl",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    for name, path in layout.items():
        (path / f"{name}.safetensors").write_bytes(b"x")
    return layout


def test_a_folder_without_subfolders_covers_only_itself():
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_network_tree(Path(tmp) / "Lora")
        rules = [preset_folders.Rule(str(tree["zit"]), subfolders=False)]

        assert preset_folders.matches(rules, tree["zit"])
        assert not preset_folders.matches(rules, tree["zit_faces"])
        assert not preset_folders.matches(rules, tree["xl"])
        assert not preset_folders.matches(rules, tree["root"])

        assert preset_folders.file_allowed(tree["zit"] / "zit.safetensors", rules)
        assert not preset_folders.file_allowed(tree["zit_faces"] / "zit_faces.safetensors", rules)


def test_including_subfolders_covers_the_whole_subtree():
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_network_tree(Path(tmp) / "Lora")
        rules = [preset_folders.Rule(str(tree["zit"]), subfolders=True)]

        assert preset_folders.matches(rules, tree["zit"])
        assert preset_folders.matches(rules, tree["zit_faces"])
        assert not preset_folders.matches(rules, tree["xl"])

        # a sibling whose name merely starts the same way is not "inside"
        assert not preset_folders.matches(rules, str(tree["zit"]) + "-extra")


def test_assignments_survive_a_reload_and_drop_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_network_tree(Path(tmp) / "Lora")
        config = Path(tmp) / "preset_folders.json"
        store = preset_folders.PresetFolderStore(config)

        store.set_rules(
            "zit",
            "lora",
            [
                preset_folders.Rule(str(tree["zit"]), subfolders=True),
                preset_folders.Rule(str(tree["zit"]) + os.sep, subfolders=False),  # same folder
                preset_folders.Rule(str(tree["xl"]), subfolders=False),
            ],
        )

        reopened = preset_folders.PresetFolderStore(config)
        rules = reopened.rules("zit", "lora")
        assert [rule.subfolders for rule in rules] == [True, False]
        assert preset_folders.normalize_dir(rules[0].path) == preset_folders.normalize_dir(tree["zit"])
        assert reopened.rules("xl", "lora") == []
        assert reopened.presets_in_use() == ["zit"]

        # an empty assignment removes the page, and the preset with it
        reopened.set_rules("zit", "lora", [])
        assert preset_folders.PresetFolderStore(config).presets_in_use() == []


def test_a_bare_string_entry_is_read_as_folder_only():
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "preset_folders.json"
        config.write_text(
            json.dumps({"version": 1, "presets": {"zit": {"lora": ["/models/Lora/zit"]}}}),
            encoding="utf-8",
        )
        rules = preset_folders.PresetFolderStore(config).rules("zit", "lora")
        assert len(rules) == 1
        assert rules[0].subfolders is False


def test_folder_choices_list_the_root_and_every_subfolder():
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_network_tree(Path(tmp) / "Lora")
        (tree["root"] / ".hidden").mkdir()

        choices = preset_folders.folder_choices([str(tree["root"])])
        labels = [label for label, _ in choices]
        values = [preset_folders.normalize_dir(value) for _, value in choices]

        assert "Lora/" in labels
        assert "Lora/zit/" in labels
        assert "Lora/zit/faces/" in labels
        assert "Lora/.hidden/" not in labels
        assert preset_folders.normalize_dir(tree["zit_faces"]) in values
        assert len(values) == len(set(values))


def test_the_filter_only_runs_for_presets_with_folders_assigned():
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_network_tree(Path(tmp) / "Lora")
        store = preset_folders.PresetFolderStore(Path(tmp) / "preset_folders.json")
        store.set_rules("zit", "lora", [preset_folders.Rule(str(tree["zit"]), subfolders=False)])

        class Page:
            extra_networks_tabname = "lora"

        with patched_store(store, preset="zit"):
            rules = network_filter.active_rules(Page())
            assert rules and rules[0].subfolders is False

        # a preset with nothing assigned is left alone
        with patched_store(store, preset="xl"):
            assert network_filter.active_rules(Page()) is None

        # ... and so is a page with nothing assigned
        class Checkpoints:
            extra_networks_tabname = "checkpoints"

        with patched_store(store, preset="zit"):
            assert network_filter.active_rules(Checkpoints()) is None


def test_items_outside_the_assigned_folders_are_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_network_tree(Path(tmp) / "Lora")
        rules = [preset_folders.Rule(str(tree["zit"]), subfolders=False)]

        inside = {"filename": str(tree["zit"] / "zit.safetensors")}
        nested = {"filename": str(tree["zit_faces"] / "zit_faces.safetensors")}
        elsewhere = {"filename": str(tree["xl"] / "xl.safetensors")}

        assert network_filter.item_allowed(inside, rules)
        assert not network_filter.item_allowed(nested, rules)
        assert not network_filter.item_allowed(elsewhere, rules)

        # an empty rule list means "the preset owns nothing here": hide it all
        assert not network_filter.item_allowed(inside, [])


def test_folder_buttons_are_kept_only_for_assigned_folders():
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_network_tree(Path(tmp) / "Lora")
        markup = "".join(
            f"""<button class='lg secondary gradio-button custom-button{" search-all" if subdir == "" else ""}'
            onclick='extraNetworksSearchButton("txt2img", "lora", event)'>
            {subdir if subdir else "all"}
            </button>"""
            for subdir in ("", "zit/", "zit/faces/", "xl/")
        )
        roots = [str(tree["root"])]

        kept = network_filter.filter_dirs_html(markup, roots, [preset_folders.Rule(str(tree["zit"]), False)])
        assert "all" in kept
        assert "zit/" in kept
        assert "zit/faces/" not in kept
        assert "xl/" not in kept

        deep = network_filter.filter_dirs_html(markup, roots, [preset_folders.Rule(str(tree["zit"]), True)])
        assert "zit/faces/" in deep
        assert "xl/" not in deep

        # backslash-separated labels (Windows) resolve to the same folders
        windows = markup.replace("zit/faces/", "\\zit\\faces\\")
        assert "zit\\faces" in network_filter.filter_dirs_html(windows, roots, [preset_folders.Rule(str(tree["zit"]), True)])

        # markup without buttons is handed back untouched
        assert network_filter.filter_dirs_html("<div>none</div>", roots, []) == "<div>none</div>"


def test_the_summary_names_the_scope_of_every_folder():
    html = render.preset_folder_summary_html(
        "zit",
        [("Lora", [("/models/Lora/zit", True, True), ("/models/Lora/gone", False, False)]), ("Checkpoints", [])],
    )
    assert "zit" in html
    assert "this folder and its subfolders" in html
    assert "this folder only" in html
    assert "folder is missing" in html

    assert "Nothing is assigned" in render.preset_folder_summary_html("xl", [("Lora", [])])


def test_checkpoint_titles_outside_the_assigned_folders_are_dropped():
    import types

    with tempfile.TemporaryDirectory() as tmp:
        tree = make_network_tree(Path(tmp) / "Stable-diffusion")
        rules = [preset_folders.Rule(str(tree["zit"]), subfolders=False)]

        def info(name, path):
            return types.SimpleNamespace(name=name, short_title=name, filename=str(path))

        checkpoints = {
            "zit.safetensors": info("zit.safetensors", tree["zit"] / "zit.safetensors"),
            "nested.safetensors": info("nested.safetensors", tree["zit_faces"] / "nested.safetensors"),
            "xl.safetensors": info("xl.safetensors", tree["xl"] / "xl.safetensors"),
        }
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(sd_checkpoint_dropdown_use_short=False, sd_model_checkpoint="xl.safetensors"))
        sd_models = types.SimpleNamespace(checkpoints_list=checkpoints)

        original = sys.modules.get("modules")
        package = types.ModuleType("modules")
        package.sd_models = sd_models
        package.shared = shared
        sys.modules["modules"] = package
        sys.modules["modules.sd_models"] = sd_models
        sys.modules["modules.shared"] = shared
        try:
            kept = network_filter.filter_checkpoint_tiles(
                ["zit.safetensors", "nested.safetensors", "xl.safetensors", "mystery.safetensors"],
                rules,
            )
        finally:
            for name in ("modules", "modules.sd_models", "modules.shared"):
                sys.modules.pop(name, None)
            if original is not None:
                sys.modules["modules"] = original

        # inside the folder: kept. nested: dropped. loaded checkpoint: always kept.
        # unknown title: kept, so nothing becomes unloadable.
        assert kept == ["zit.safetensors", "xl.safetensors", "mystery.safetensors"]


# ------------------------------------------------------- descriptions, again


def test_the_extractor_keeps_its_state_out_of_htmlparsers_way():
    # Python 3.13 gave HTMLParser a ``_pending`` buffer; an extractor that also
    # used the name broke close(), and every description fell back to stripping.
    from html.parser import HTMLParser

    own = set(vars(descriptions._TextExtractor())) - set(vars(HTMLParser()))
    assert own and all(name.startswith("_text_") for name in own), own
    assert descriptions.html_to_text("<p>one</p><ul><li>two</li></ul>") == "one\n\n• two"


def test_the_description_fallback_never_leaks_a_script():
    original = descriptions._TextExtractor.feed

    def broken(self, data):
        raise ValueError("unparseable")

    descriptions._TextExtractor.feed = broken
    try:
        text = descriptions.html_to_text("<p>Kept</p><script>steal()</script><style>p{}</style><b>too</b>")
    finally:
        descriptions._TextExtractor.feed = original
    assert "Kept" in text and "too" in text
    assert "steal" not in text and "p{}" not in text, text


# ------------------------------------------------------------- MOSS-Image


MOSS_NAMES = (
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_A.weight",
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_B.weight",
)


def make_lora(path: Path, names, metadata: dict | None = None) -> None:
    """A safetensors file with real tensor entries in its header."""
    header: dict = {"__metadata__": metadata or {}}
    for index, name in enumerate(names):
        header[name] = {"dtype": "F16", "shape": [1], "data_offsets": [index * 2, index * 2 + 2]}
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * (2 * len(names)))


def test_several_base_models_can_be_accepted():
    assert civitai.base_model_choices("Krea 2, MOSS Image") == ["Krea 2", "MOSS Image"]
    assert civitai.base_model_choices(" ,; ") == []
    assert civitai.matches_base_model("MOSS Image", "Krea 2, MOSS Image")
    assert civitai.matches_base_model("Krea2", "Krea 2; MOSS")
    assert not civitai.matches_base_model("Flux.1 D", "Krea 2, MOSS Image")

    assessment = civitai.assess_base_model({"baseModel": "MOSS Image"}, {}, wanted="Krea 2, MOSS Image", require_explicit=True)
    assert assessment.allowed and assessment.explicit


def test_a_variant_type_is_not_a_conflicting_base_model():
    # "Standard" says which variant a version is, not which family: a LoRA
    # labelled "Other" must not be rejected because of it.
    other = civitai.assess_base_model({"baseModel": "Other", "baseModelType": "Standard"}, {}, wanted="MOSS Image", require_explicit=False)
    assert other.allowed and not other.explicit, other

    krea = civitai.assess_base_model({"baseModel": "Krea 2", "baseModelType": "Standard"}, {}, wanted="MOSS Image", require_explicit=False)
    assert not krea.allowed and "Krea 2" in krea.reason and "Standard" not in krea.reason

    assert civitai.family_label({"baseModelType": "Standard"}) == ""
    assert civitai.family_label({"baseModel": "MOSS Image", "baseModelType": "Standard"}) == "MOSS Image"


def test_a_moss_lora_is_recognised_by_its_tensor_names():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "moss.safetensors"
        make_lora(path, MOSS_NAMES, {"ss_network_module": "networks.lora"})
        metadata, names = civitai.read_safetensors_header(path)
        assert metadata == {"ss_network_module": "networks.lora"}
        assert civitai.local_architecture(names) == "MOSS Image"
        assert civitai.read_safetensors_metadata(path) == metadata

    assert civitai.local_architecture(["lora_unet_layers_3_proj_ffn_down.lora_up.weight"]) == "MOSS Image"
    for name in (
        "diffusion_model.blocks.0.attn.qkv.lora_A.weight",
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
        "transformer.layers.0.attention.qkv.lora_A.weight",
        "lora_unet_layers_0_feed_forward_w1.lora_down.weight",
        "head.proj.0.weight",
    ):
        assert civitai.local_architecture([name]) == "", name

    # Civitai's label is generic: the names are explicit evidence for MOSS...
    moss = civitai.assess_base_model({"baseModel": "Other"}, {}, wanted="MOSS Image", require_explicit=True, tensor_names=MOSS_NAMES)
    assert moss.allowed and moss.explicit and "tensor names" in moss.reason
    # ...and never let a MOSS LoRA through a filter for something else.
    krea = civitai.assess_base_model({"baseModel": "Other"}, {}, wanted="Krea 2", require_explicit=True, tensor_names=MOSS_NAMES)
    assert not krea.allowed


def test_a_fetch_uses_the_tensor_names_of_the_file():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hashing.CACHE = hashing.HashCache(root / "hashes.json")
        loras = root / "loras"
        make_lora(loras / "moss.safetensors", MOSS_NAMES)
        sha = hashlib.sha256((loras / "moss.safetensors").read_bytes()).hexdigest()

        class OtherLabel(StubClient):
            def version_by_hash(self, sha256):
                version = super().version_by_hash(sha256)
                if version is not None:
                    version["baseModel"] = "Other"
                return version

        original = civitai.CivitaiClient
        civitai.CivitaiClient = lambda **kwargs: OtherLabel({sha}, **kwargs)
        try:
            report = civitai.scan_folder(
                loras,
                root / "out.csv",
                base_model_filter="MOSS Image",
                require_explicit_label=True,
                log=lambda _m: None,
            )
        finally:
            civitai.CivitaiClient = original
        assert report.resolved == 1, report.counts
        row = next(iter(csv.DictReader((root / "out.csv").open(encoding="utf-8"))))
        assert row["status"] == "ok" and "tensor names" in row["note"]


def test_a_moss_layout_prompt_is_read_and_kept_verbatim():
    layout = {
        "summary": {"caption": "A cozy nook (dusk)", "meta": ["photo"], "canvas": "single"},
        "items": [
            {"label": "object", "caption": "armchair", "bbox": [-10, -10, 10, 10]},
            {"label": "text", "caption": "a sign", "text": "READ", "bbox": [0, 0, 5, 5]},
        ],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    compiled = layouts.escape_parens(json.dumps(layout, ensure_ascii=False, separators=(",", ":")))
    assert "\\(dusk\\)" in compiled

    parsed = layouts.parse(compiled)
    assert parsed.caption == "A cozy nook (dusk)"
    assert [item.label for item in parsed.items] == ["object", "text"] and parsed.items[1].text == "READ"
    assert parsed.describe() == "1 object, 1 text"
    assert layouts.label(compiled, 60).startswith("[layout] A cozy nook")
    for plain in ("a plain prompt (with parens)", "{not json}", '{"a": 1}', ""):
        assert layouts.parse(plain) is None, plain
    assert layouts.label("a plain prompt", 60) == "a plain prompt"

    # the API passing the layout on as an object gets Forge's compiled line back
    assert civitai.extract_positive_prompt({"meta": {"prompt": layout}}) == compiled
    assert civitai.extract_positive_prompt({"meta": {"prompt": {"caption": "just (this)"}}}) == "just \\(this\\)"
    # one that arrives as text is passed through untouched
    assert civitai.extract_positive_prompt({"meta": {"prompt": compiled}}) == compiled
    # and an object that is no layout is still ignored
    assert civitai.extract_positive_prompt({"meta": {"prompt": {"nodes": [1, 2]}}}) == ""


def test_moss_is_a_known_preset_without_forge():
    assert "moss" in preset_folders.FALLBACK_PRESETS and "qwen21" in preset_folders.FALLBACK_PRESETS


# ------------------------------------------------------------ the panel


MOSS_ROW = {
    "safetensor_file": "moss/mossLight_v1.safetensors",
    "sha256": "ab" * 32,
    "civitai_model_name": "Moss light",
    "civitai_version_name": "v1.0",
    "base_model": "MOSS Image",
    "civitai_creator": "kohaku",
    "civitai_model_id": "11",
    "civitai_version_id": "22",
    "civitai_url": "https://civitai.com/models/11?modelVersionId=22",
    "trigger_words": "rim light, backlit",
    "status": "ok",
    "note": "fine",
    "model_description": "About it.",
    "version_description": "First release.",
    "negative_prompt": "flat",
    "positive_prompt_1": "a prompt",
}


def test_the_record_carries_the_facts_the_header_shows():
    record = parse_row(dict(MOSS_ROW), source="civitai-moss.csv", row_number=2)
    assert (record.version_name, record.base_model, record.creator) == ("v1.0", "MOSS Image", "kohaku")
    assert (record.model_id, record.version_id, record.status, record.note) == ("11", "22", "ok", "fine")
    assert record.resolved and record.title == "Moss light"
    # read for the header, and still listed with the row's other fields
    assert record.extras["civitai_version_name"] == "v1.0" and record.extras["base_model"] == "MOSS Image"

    unresolved = parse_row(
        {"safetensor_file": "krea2\\funnyFaces_v1.safetensors", "sha256": "cd" * 32, "status": "not_found"},
        source="civitai-krea2.csv",
        row_number=3,
    )
    assert not unresolved.resolved
    assert unresolved.title == "funnyFaces_v1", "the file's own spelling, not the matching key"


def test_the_panel_header_and_sidebar_render_the_record():
    record = parse_row(dict(MOSS_ROW), source="civitai-moss.csv", row_number=2)

    header = render.header_html(records=[record], how="SHA256")
    assert "Moss light" in header and "Open on Civitai" in header
    assert all(text in header for text in ("v1.0", "MOSS Image", "by kohaku", "civitai-moss.csv (row 2)"))

    facts = render.facts_html([record], identity={"SHA256": "ab" * 32, "Forge hash": "abcdef123456"}, architecture="MOSS Image")
    assert "Creator" in facts and "tensor names" in facts and "abcdef123456" in facts

    chips = render.triggers_html(["rim light", "<b>x</b>"], target="t_activation", active={"rim light"})
    assert 'data-eld-target="t_activation"' in chips and chips.count("eld-word-on") == 1
    assert "<b>x</b>" not in chips and "&lt;b&gt;x&lt;/b&gt;" in chips
    assert 'data-eld-action="copy-word"' in render.triggers_html(["a"]), "without a target a chip copies"

    side = render.descriptions_html([(record.title, record.source, record.description, record.version_description)])
    assert "Description" in side and "Version notes" in side and "eld-clamp" not in side
    long_text = "word " * 400
    assert "eld-clamp" in render.text_section_html("Description", long_text) and "Show more" in render.text_section_html("Description", long_text)
    assert 'data-eld-action="append-negative"' in render.negative_html(record.negative_prompts, tab="img2img")
    assert render.fields_html([record]).startswith("<details") and "My" not in render.fields_html([record])

    unresolved = parse_row({"safetensor_file": "x.safetensors", "sha256": "cd" * 32, "status": "not_found", "note": "SHA256 not found on Civitai"}, source="c.csv", row_number=2)
    html = render.header_html(records=[unresolved], how="SHA256")
    assert "Civitai had nothing" in html and "not_found" in html and "SHA256 not found" in html
    assert "eld-head-title" not in html, "no title for a LoRA Civitai did not name"
    assert "No CSV row matched" in render.header_html(records=[], how="")


def test_prompt_cards_carry_the_exact_prompt_and_escape_it():
    layout_prompt = layouts.escape_parens(json.dumps({"summary": {"caption": "Two friends (laughing)"}, "items": []}, separators=(",", ":")))
    cards = [
        {"index": 0, "prompt": "a <b>bold</b> (prompt)", "thumb": "", "saved": False, "video": False, "url": "http://x/1.jpeg"},
        {"index": 1, "prompt": layout_prompt, "thumb": "data:image/webp;base64,AAAA", "saved": True, "video": False, "url": "http://x/2.jpeg"},
    ]
    html = render.prompt_cards_html(cards, total=5, tab="txt2img", bridge="pick_id")

    assert html.count('class="eld-card"') == 2
    assert 'data-eld-bridge="pick_id"' in html and 'data-eld-tab="txt2img"' in html
    assert "<b>bold</b>" not in html and "&lt;b&gt;bold&lt;/b&gt;" in html
    assert "Two friends (laughing)" in html and "eld-badge-inline" in html, "a layout shows its caption"
    assert 'data-eld-action="download"' in html, "with images around, an unsaved card offers its download"
    assert "…and 3 more" in html

    sources = [html_module.unescape(text) for text in re.findall(r'<textarea class="eld-copy-source"[^>]*>(.*?)</textarea>', html, re.S)]
    assert sources == [cards[0]["prompt"], layout_prompt], "Copy/Append/Replace act on the prompt exactly"

    # with no image saved anywhere, the cards stay text-only
    bare = render.prompt_cards_html([dict(cards[0])], total=1, tab="txt2img", bridge="pick_id")
    assert "eld-card-media" not in bare


def test_thumbnails_are_made_once_and_embedded():
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships with the WebUI
        return

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "civitai-1.png"
        Image.new("RGB", (900, 1200), (10, 20, 30)).save(source)
        cache = Path(tmp) / "thumbs"

        first = images.thumbnail(source, directory=cache)
        assert first is not None and first.suffix == ".webp"
        with Image.open(first) as thumb:
            assert max(thumb.size) <= images.THUMB_SIZE
        assert images.thumbnail(source, directory=cache) == first, "made once"
        assert images.thumbnail_data_uri(source, directory=cache).startswith("data:image/webp;base64,")

        # a replaced image gets a fresh thumbnail
        Image.new("RGB", (100, 100), (200, 0, 0)).save(source)
        stamp = time.time() + 5
        os.utime(source, (stamp, stamp))
        assert images.thumbnail(source, directory=cache) != first

        (Path(tmp) / "clip.mp4").write_bytes(b"\x00" * 16)
        assert images.thumbnail(Path(tmp) / "clip.mp4", directory=cache) is None
        assert images.thumbnail_data_uri(Path(tmp) / "missing.png", directory=cache) == ""


def test_the_panel_moves_up_below_the_header_row():
    try:
        from extended_lora_details import details_ui
    except ImportError:  # needs Gradio; exercised inside the WebUI
        return

    class Block:
        def __init__(self, parent=None):
            self.parent = parent
            self.children = []
            if parent is not None:
                parent.children.append(self)

    box = Block()
    hidden = Block(box)
    header_row = Block(box)
    name = Block(Block(header_row))
    field = Block(box)
    panel = Block(box)

    editor = types.SimpleNamespace(box=box, edit_name=name)
    assert details_ui._place_after_header(editor, panel)
    assert box.children == [hidden, header_row, panel, field]

    # anything unexpected leaves the layout as it is
    assert not details_ui._place_after_header(types.SimpleNamespace(box=box, edit_name=None), panel)
    assert not details_ui._place_after_header(types.SimpleNamespace(box=None, edit_name=name), panel)
    assert box.children == [hidden, header_row, panel, field]


def test_the_fetch_button_appears_only_when_it_has_work():
    try:
        from extended_lora_details import details_ui
    except ImportError:  # needs Gradio; exercised inside the WebUI
        return

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "krea2"
        lora = folder / "x.safetensors"
        folder.mkdir()
        lora.write_bytes(b"x")

        plan = details_ui._fetch_plan(str(lora), [])
        assert plan["folder"] == str(folder) and plan["label"] == "krea2/" and not plan["retry"]

        unresolved = parse_row({"safetensor_file": "x.safetensors", "sha256": "ab" * 32, "status": "not_found"}, source="c.csv", row_number=2)
        assert details_ui._fetch_plan(str(lora), [unresolved])["retry"], "a miss is retried"

        resolved = parse_row(dict(MOSS_ROW), source="c.csv", row_number=2)
        assert details_ui._fetch_plan(str(lora), [resolved]) == {}
        assert details_ui._fetch_plan("", []) == {}


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
