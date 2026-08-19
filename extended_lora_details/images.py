"""Downloading gallery images for a prompt.

Everything lands in one flat folder (``<extension>/images`` by default) so the
collection can be managed in one place. A file's name is derived from the image's
Civitai id, falling back to a digest of its URL, which makes "have I already got
this one?" a plain filesystem question — delete a file and the UI offers to
download it again, exactly like it had never been fetched.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .common import EXTENSION_ROOT, clean_cell, human_bytes, log, opt

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_BYTES = 32 * 1024 * 1024
CHUNK_SIZE = 64 * 1024

# Extensions we are willing to write. Anything else is stored as .img rather
# than trusting a remote-supplied suffix.
KNOWN_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp", ".mp4", ".webm"}

CONTENT_TYPE_SUFFIXES = {
    "image/jpeg": ".jpeg",
    "image/jpg": ".jpeg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
    "image/bmp": ".bmp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def default_image_dir() -> Path:
    return EXTENSION_ROOT / "images"


def image_dir() -> Path:
    """The configured image folder, created on demand."""
    configured = str(opt("eld_image_dir", "") or "").strip()
    path = Path(configured).expanduser() if configured else default_image_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log(f"could not create the image directory {path}: {exc}; using the default")
        path = default_image_dir()
        path.mkdir(parents=True, exist_ok=True)
    return path


def _suffix_from_url(url: str) -> str:
    name = unquote(urlparse(url).path).rsplit("/", 1)[-1]
    suffix = os.path.splitext(name)[1].lower()
    return suffix if suffix in KNOWN_SUFFIXES else ""


def _suffix_from_content_type(content_type: str) -> str:
    return CONTENT_TYPE_SUFFIXES.get(content_type.split(";", 1)[0].strip().lower(), "")


def stem_for(url: str, image_id: str = "") -> str:
    """A stable, filesystem-safe base name for one gallery image."""
    identifier = _SAFE_ID_RE.sub("", clean_cell(image_id))
    if identifier:
        return f"civitai-{identifier}"
    digest = hashlib.sha256(clean_cell(url).encode("utf-8")).hexdigest()[:16]
    return f"url-{digest}"


def find_local(url: str, image_id: str = "", *, directory: Path | None = None) -> Path | None:
    """The downloaded file for this image, or ``None`` if it is not on disk.

    Answered from the filesystem every time, never from a manifest, so a file
    the user deleted simply reads as "not downloaded".
    """
    if not clean_cell(url):
        return None

    folder = directory or image_dir()
    stem = stem_for(url, image_id)

    preferred = _suffix_from_url(url)
    candidates = [preferred, *KNOWN_SUFFIXES, ".img"] if preferred else [*KNOWN_SUFFIXES, ".img"]
    for suffix in candidates:
        candidate = folder / f"{stem}{suffix}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    path: Path | None
    message: str
    skipped: bool = False


def download(url: str, image_id: str = "", *, directory: Path | None = None, timeout: float | None = None, max_bytes: int | None = None, cancel=None) -> DownloadResult:
    """Fetch one gallery image into the image folder.

    Already-present files are reported as skipped rather than re-fetched.
    """
    url = clean_cell(url)
    if not url:
        return DownloadResult(False, None, "This prompt has no image URL in the library.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return DownloadResult(False, None, f"Refusing to fetch a non-HTTP URL: {url[:120]}")

    folder = directory or image_dir()
    existing = find_local(url, image_id, directory=folder)
    if existing is not None:
        return DownloadResult(True, existing, f"Already downloaded: {existing.name}", skipped=True)

    timeout = float(timeout if timeout is not None else opt("eld_image_timeout", DEFAULT_TIMEOUT))
    max_bytes = int(max_bytes if max_bytes is not None else opt("eld_image_max_mb", 32) * 1024 * 1024)
    max_bytes = max(1, max_bytes)

    try:
        import requests
    except ImportError:  # pragma: no cover - requests ships with the WebUI
        return DownloadResult(False, None, "The requests package is not installed.")

    headers = {"User-Agent": "SD-ForgeNeo-ExtendedLoraDetails/1.0", "Accept": "image/*,video/*;q=0.8,*/*;q=0.5"}
    api_key = str(opt("eld_civitai_api_key", "") or "").strip()
    if api_key and parsed.hostname and parsed.hostname.lower().endswith("civitai.com"):
        headers["Authorization"] = f"Bearer {api_key}"

    temp_name: str | None = None
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
            if response.status_code != 200:
                return DownloadResult(False, None, f"HTTP {response.status_code} from {parsed.netloc}")

            content_type = response.headers.get("Content-Type", "")
            suffix = _suffix_from_content_type(content_type) or _suffix_from_url(url) or ".img"

            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                return DownloadResult(False, None, f"Image is larger than the {human_bytes(max_bytes)} limit.")

            folder.mkdir(parents=True, exist_ok=True)
            written = 0
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".eld-", suffix=".part", delete=False) as handle:
                temp_name = handle.name
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if cancel is not None and cancel.is_set():
                        return DownloadResult(False, None, "Cancelled.")
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        return DownloadResult(False, None, f"Image exceeded the {human_bytes(max_bytes)} limit.")
                    handle.write(chunk)

            if written == 0:
                return DownloadResult(False, None, "The server returned an empty response.")

            destination = folder / f"{stem_for(url, image_id)}{suffix}"
            os.replace(temp_name, destination)
            temp_name = None
            return DownloadResult(True, destination, f"Saved {destination.name} ({human_bytes(written)})")

    except requests.RequestException as exc:
        return DownloadResult(False, None, f"Download failed: {exc}")
    except OSError as exc:
        return DownloadResult(False, None, f"Could not write the file: {exc}")
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def stats(directory: Path | None = None) -> dict:
    """Count and total size of the image folder."""
    folder = directory or image_dir()
    count = 0
    total = 0
    try:
        for entry in os.scandir(folder):
            if entry.is_file() and not entry.name.startswith(".eld-"):
                count += 1
                total += entry.stat().st_size
    except OSError:
        pass
    return {"directory": str(folder), "files": count, "bytes": total}


def purge(directory: Path | None = None) -> tuple[int, str]:
    """Delete every downloaded image. Returns ``(removed, message)``."""
    folder = directory or image_dir()
    removed = 0
    failed = 0
    try:
        entries = list(os.scandir(folder))
    except OSError as exc:
        return 0, f"Could not read the image folder: {exc}"

    for entry in entries:
        if not entry.is_file():
            continue
        try:
            os.unlink(entry.path)
            removed += 1
        except OSError:
            failed += 1

    message = f"Deleted {removed} file(s) from {folder}."
    if failed:
        message += f" {failed} could not be removed."
    return removed, message
