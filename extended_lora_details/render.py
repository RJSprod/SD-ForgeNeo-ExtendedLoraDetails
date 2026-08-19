"""HTML fragments for the details panel.

Everything here styles itself with Gradio's own CSS custom properties
(``--body-text-color``, ``--border-color-primary``, ...) so a user's theme keeps
applying; there are no hard-coded colours.
"""

from __future__ import annotations

import html
from urllib.parse import urlparse

from .common import truncate

MAX_VALUE_CHARS = 4000


def escape(value) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _linkify(value: str) -> str:
    text = str(value).strip()
    if text.startswith(("http://", "https://")) and " " not in text:
        return f'<a href="{escape(text)}" target="_blank" rel="noopener noreferrer">{escape(truncate(text, 90))}</a>'
    return escape(value).replace("\n", "<br>")


def notice(message: str, *, kind: str = "info") -> str:
    return f'<div class="eld-notice eld-notice-{escape(kind)}">{message}</div>'


def summary_html(*, matched: bool, how: str, records, identity: dict) -> str:
    """The one-line status strip at the top of the panel."""
    rows = []
    for label, value in identity.items():
        if value:
            rows.append(f'<span class="eld-chip"><b>{escape(label)}</b> <code>{escape(value)}</code></span>')
    chips = f'<div class="eld-chips">{"".join(rows)}</div>' if rows else ""

    if not matched:
        return notice(
            "No CSV row matched this LoRA. Add a library CSV in "
            "<b>Settings → Extended LoRA Details</b>, or run a Civitai fetch for its folder "
            "from the <b>Extended LoRA Details</b> tab.",
            kind="empty",
        ) + chips

    names = []
    for record in records:
        title = escape(record.title)
        names.append(f'<a href="{escape(record.url)}" target="_blank" rel="noopener noreferrer">{title}</a>' if record.url else title)

    sources = ", ".join(sorted({escape(record.source) for record in records}))
    plural = "s" if len(records) != 1 else ""
    body = (
        f'<b>{" · ".join(names)}</b>'
        f'<span class="eld-muted"> — {len(records)} row{plural} matched by {escape(how)} '
        f"from {sources}</span>"
    )
    return notice(body, kind="ok") + chips


def _value_cell(value: str) -> str:
    text = str(value)
    if len(text) > MAX_VALUE_CHARS:
        text = text[:MAX_VALUE_CHARS] + "…"
    return _linkify(text)


def overview_html(records) -> str:
    """A field table per matched CSV row."""
    if not records:
        return ""

    blocks = []
    for record in records:
        fields: list[tuple[str, str]] = []
        if record.model_name:
            fields.append(("Model", record.model_name))
        if record.url:
            fields.append(("Civitai", record.url))
        if record.trigger_words:
            fields.append(("Trigger words", ", ".join(record.trigger_words)))
        fields.append(("Prompts", str(len(record.prompts))))
        if record.negative_prompts:
            fields.append(("Negative prompt", "\n".join(record.negative_prompts)))
        if record.rel_path:
            fields.append(("CSV file path", record.rel_path))
        if record.sha256:
            fields.append(("SHA256", record.sha256))
        for label, value in record.extras.items():
            fields.append((label, value))

        body = "".join(f"<tr><th>{escape(label)}</th><td>{_value_cell(value)}</td></tr>" for label, value in fields if str(value).strip())
        blocks.append(
            f'<div class="eld-source">'
            f'<div class="eld-source-title">{escape(record.source)} <span class="eld-muted">· row {record.row_number}</span></div>'
            f'<table class="eld-table">{body}</table>'
            f"</div>"
        )

    return "".join(blocks)


def prompt_meta_html(total: int, index: int, sources: list[str]) -> str:
    if total <= 0:
        return notice("This LoRA has no gallery prompts in the library.", kind="empty")
    origin = f' <span class="eld-muted">· from {escape(", ".join(sorted(set(sources))))}</span>' if sources else ""
    return f'<div class="eld-count">Prompt <b>{index + 1}</b> of <b>{total}</b>{origin}</div>'


def image_status_html(*, downloaded: bool, name: str, url: str) -> str:
    """The line above the download button, in the Prompts tab."""
    source = urlparse(url).netloc if url else ""
    origin = f' <span class="eld-muted">from {escape(source)}</span>' if source else ""

    if downloaded:
        return notice(f"Image saved as <code>{escape(name)}</code>.{origin}", kind="ok")
    return notice(f"This prompt's gallery image has not been downloaded yet.{origin}", kind="info")


def prompt_list_html(prompts: list[str], *, limit: int = 50) -> str:
    if not prompts:
        return ""
    shown = prompts[: max(1, limit)]
    items = "".join(
        f'<li><span class="eld-prompt-index">{index}</span><span class="eld-prompt-text">{escape(prompt)}</span></li>'
        for index, prompt in enumerate(shown, start=1)
    )
    remainder = ""
    if len(prompts) > len(shown):
        remainder = f'<div class="eld-muted eld-more">…and {len(prompts) - len(shown)} more — use the selector above to reach them.</div>'
    return f'<ol class="eld-prompt-list">{items}</ol>{remainder}'


def library_summary_html(stats: dict) -> str:
    errors = stats.get("errors") or []
    rows = [
        ("Library directory", stats.get("directory", "")),
        ("CSV files", str(stats.get("files", 0))),
        ("Rows", str(stats.get("rows", 0))),
        ("Rows with a SHA256", str(stats.get("hashed_rows", 0))),
        ("Prompts", str(stats.get("prompts", 0))),
    ]
    body = "".join(f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows)
    table = f'<table class="eld-table">{body}</table>'

    if not stats.get("files"):
        table += notice("The library is empty. Upload a CSV, or run a Civitai fetch on a folder.", kind="empty")
    for source in errors:
        table += notice(f"<b>{escape(source.name)}</b>: {escape(source.error)}", kind="error")
    return table


def image_library_html(stats: dict) -> str:
    from .common import human_bytes

    rows = [
        ("Image folder", stats.get("directory", "")),
        ("Downloaded images", str(stats.get("files", 0))),
        ("Total size", human_bytes(stats.get("bytes", 0))),
    ]
    body = "".join(f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows)
    table = f'<table class="eld-table">{body}</table>'
    if not stats.get("files"):
        table += notice(
            "Nothing downloaded yet. Open a LoRA's details dialog, go to <b>Prompts</b>, and use "
            "<b>Download this image</b>.",
            kind="empty",
        )
    return table


def sources_html(sources) -> str:
    if not sources:
        return ""
    from .common import human_bytes

    rows = "".join(
        "<tr>"
        f"<td>{escape(source.name)}</td>"
        f'<td class="eld-num">{source.rows}</td>'
        f'<td class="eld-num">{escape(human_bytes(source.size))}</td>'
        f"<td>{escape(source.error) if source.error else ''}</td>"
        "</tr>"
        for source in sources
    )
    return (
        '<table class="eld-table eld-grid">'
        "<thead><tr><th>File</th><th>Rows</th><th>Size</th><th>Problem</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _progress_bar(percent: int) -> str:
    return f'<div class="eld-bar"><div class="eld-bar-fill" style="width:{max(0, min(100, percent))}%"></div></div>'


def job_html(snapshots, *, log_lines: int = 18) -> str:
    """The job monitor: newest first, with the running job's log tailed."""
    if not snapshots:
        return notice("No scans have been started yet.", kind="empty")

    state_labels = {
        "queued": "Queued",
        "running": "Running",
        "done": "Finished",
        "failed": "Failed",
        "cancelled": "Cancelled",
    }

    blocks = []
    for snapshot in snapshots:
        header = (
            f'<div class="eld-job-head">'
            f'<span class="eld-badge eld-badge-{escape(snapshot.state)}">{escape(state_labels.get(snapshot.state, snapshot.state))}</span>'
            f'<b>#{snapshot.id} {escape(snapshot.title)}</b>'
            f'<span class="eld-muted">{snapshot.elapsed:.0f}s</span>'
            f"</div>"
        )
        body = [header]
        if snapshot.active:
            body.append(_progress_bar(snapshot.percent))
        body.append(f'<div class="eld-muted eld-job-msg">{escape(snapshot.message)}</div>')
        if snapshot.summary:
            body.append(f'<div class="eld-job-summary">{escape(snapshot.summary)}</div>')
        if snapshot.error:
            body.append(notice(escape(snapshot.error), kind="error"))
        if snapshot.log and (snapshot.active or snapshot.state == "failed"):
            tail = snapshot.log[-log_lines:]
            body.append(f'<pre class="eld-log">{escape(chr(10).join(tail))}</pre>')
        blocks.append(f'<div class="eld-job">{"".join(body)}</div>')

    return "".join(blocks)
