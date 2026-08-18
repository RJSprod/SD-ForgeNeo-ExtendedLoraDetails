"""Paths, logging and small shared helpers."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

PREFIX = "[Extended LoRA Details]"

EXTENSION_ROOT = Path(__file__).resolve().parent.parent

SAFETENSORS_EXTENSIONS = (".safetensors", ".safetensor")
NETWORK_EXTENSIONS = (".safetensors", ".safetensor", ".pt", ".ckpt")

_print_lock = threading.Lock()


def log(message: str) -> None:
    with _print_lock:
        print(f"{PREFIX} {message}", flush=True)


def report(message: str, exc_info: bool = True) -> None:
    """Route an error through the WebUI reporter when available."""
    try:
        from modules import errors

        errors.report(f"{PREFIX} {message}", exc_info=exc_info)
    except Exception:
        import traceback

        log(f"ERROR: {message}")
        if exc_info:
            traceback.print_exc()


def default_library_dir() -> Path:
    return EXTENSION_ROOT / "library"


def cache_dir() -> Path:
    path = EXTENSION_ROOT / ".cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_library_dir() -> Path:
    """Library directory from settings, falling back to ``<extension>/library``."""
    configured = ""
    try:
        from modules import shared

        configured = (getattr(shared.opts, "eld_library_dir", "") or "").strip()
    except Exception:
        configured = ""

    path = Path(configured).expanduser() if configured else default_library_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log(f"could not create library directory {path}: {exc}; using default")
        path = default_library_dir()
        path.mkdir(parents=True, exist_ok=True)
    return path


def opt(name: str, fallback):
    """Read a WebUI setting without exploding when the UI is not up yet."""
    try:
        from modules import shared

        value = getattr(shared.opts, name, fallback)
        return fallback if value is None else value
    except Exception:
        return fallback


def lora_directories() -> list[str]:
    """Every directory Forge scans for LoRA networks."""
    try:
        from modules import shared

        dirs = [shared.cmd_opts.lora_dir, *getattr(shared.cmd_opts, "lora_dirs", [])]
    except Exception:
        return []

    seen: list[str] = []
    for entry in dirs:
        if not entry:
            continue
        resolved = os.path.abspath(os.path.expanduser(str(entry)))
        if resolved not in seen:
            seen.append(resolved)
    return seen


def normalize_key(value) -> str:
    """Lower-case a column name / label down to ``[a-z0-9_]``."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def clean_cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


_HEX_RE = re.compile(r"^[0-9a-f]+$")


def normalize_hash(value) -> str:
    """Return a lower-case hex hash, or ``""`` when the value is not hex."""
    text = clean_cell(value).lower()
    if not text:
        return ""
    if text.startswith("0x"):
        text = text[2:]
    if len(text) < 8 or not _HEX_RE.match(text):
        return ""
    return text


def normalize_path(value) -> str:
    """Case-folded POSIX form of a path, for path-based CSV matching."""
    text = clean_cell(value).replace("\\", "/").strip("/")
    return text.casefold()


def path_stem(value) -> str:
    text = normalize_path(value)
    if not text:
        return ""
    base = text.rsplit("/", 1)[-1]
    for extension in NETWORK_EXTENSIONS:
        if base.endswith(extension):
            return base[: -len(extension)]
    return os.path.splitext(base)[0]


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
