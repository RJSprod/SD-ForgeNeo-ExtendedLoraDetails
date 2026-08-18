"""The panel appended to Forge's LoRA "edit metadata" dialog.

The dialog is built by ``LoraUserMetadataEditor.create_editor()``, which calls
``create_default_editor_elems()`` → the LoRA fields → ``create_default_buttons()``.
Patching ``create_default_buttons`` therefore lets us insert a panel *below the
existing fields but above Cancel/Save*, without rewriting any upstream code and
without touching the checkpoint or embedding editors.

Population is wired as a **second** listener on the same ``button_edit`` click
that upstream already uses, so the original outputs are untouched: if anything
in here breaks, the stock dialog still works.
"""

from __future__ import annotations

import json
import os
import re

import gradio as gr

from . import hashing, render
from .common import lora_directories, normalize_path, opt, report, truncate
from .library import get_library

TRIGGER_SPLIT_RE = re.compile(r" *, *")


# ------------------------------------------------------------------ gathering


def _relative_candidates(filename: str) -> list[str]:
    """Every path form a CSV might have used for this file."""
    if not filename:
        return []

    absolute = os.path.abspath(filename)
    candidates: list[str] = []

    for directory in lora_directories():
        try:
            if os.path.commonpath([absolute, directory]) == directory:
                candidates.append(os.path.relpath(absolute, directory))
        except ValueError:
            continue

    parent = os.path.dirname(absolute)
    candidates.append(os.path.basename(absolute))
    if parent:
        candidates.append(os.path.join(os.path.basename(parent), os.path.basename(absolute)))
    candidates.append(absolute)

    seen: list[str] = []
    for candidate in candidates:
        key = normalize_path(candidate)
        if key and key not in seen:
            seen.append(key)
    return seen


def _short_hash_candidates(item: dict) -> list[str]:
    """Cheap hashes already known to Forge - no file reads involved."""
    metadata = item.get("metadata") or {}
    candidates = [
        item.get("shorthash") or "",
        metadata.get("sshs_model_hash") or "",
        metadata.get("sshs_legacy_hash") or "",
    ]

    filename = item.get("filename") or ""
    if filename:
        cached_addnet = hashing.CACHE.peek(filename, "addnet")
        if cached_addnet:
            candidates.append(cached_addnet)

    return [candidate for candidate in candidates if candidate]


def _resolve_sha256(filename: str) -> tuple[str, bool]:
    """Return ``(sha256, computed_now)``; honours the two hashing settings."""
    if not filename or not opt("eld_hash_lookup", True):
        return "", False

    cached = hashing.known_hash(filename)
    if cached:
        return cached, False

    if not opt("eld_hash_on_open", True):
        return "", False

    value = hashing.sha256_file(filename) or ""
    if value:
        hashing.flush()
    return value, bool(value)


def gather(page, name: str) -> dict:
    """Everything the panel needs for one card, as plain data."""
    item = (page.items.get(name, {}) if page is not None else {}) or {}
    filename = item.get("filename") or ""

    sha256, computed = _resolve_sha256(filename)
    short_hashes = _short_hash_candidates(item)
    rel_paths = _relative_candidates(filename)

    library = get_library()
    records, how = library.lookup(
        sha256=sha256,
        short_hashes=short_hashes,
        filename=filename,
        rel_paths=rel_paths,
        name=name,
    )

    prompts: list[str] = []
    prompt_sources: list[str] = []
    triggers: list[tuple[str, str]] = []
    seen_triggers: set[str] = set()
    for record in records:
        for prompt in record.prompts:
            if prompt not in prompts:
                prompts.append(prompt)
                prompt_sources.append(record.source)
        for word in record.trigger_words:
            if word not in seen_triggers:
                seen_triggers.add(word)
                triggers.append((word, record.source))

    identity = {
        "SHA256": sha256 or ("not computed" if filename else ""),
        "Forge hash": item.get("shorthash") or "",
    }

    return {
        "records": records,
        "how": how,
        "prompts": prompts,
        "prompt_sources": prompt_sources,
        "triggers": triggers,
        "identity": identity,
        "computed": computed,
        "filename": filename,
    }


# ------------------------------------------------------------------- helpers


def _prompt_choices(prompts: list[str]) -> list[tuple[str, int]]:
    width = max(1, int(opt("eld_prompt_label_width", 90)))
    return [(f"{index + 1}. {truncate(prompt, width)}", index) for index, prompt in enumerate(prompts)]


def _clamp(index, total: int) -> int:
    try:
        value = int(index)
    except (TypeError, ValueError):
        value = 0
    if total <= 0:
        return 0
    return max(0, min(total - 1, value))


def _raw_json(records) -> str:
    if not records:
        return "{}"
    payload = [
        {
            "source": record.source,
            "row": record.row_number,
            "fields": record.raw,
        }
        for record in records
    ]
    try:
        return json.dumps(payload if len(payload) > 1 else payload[0], indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


# ---------------------------------------------------------------- the panel


def build_panel(editor) -> None:
    """Create the panel and wire it into ``editor.button_edit``."""
    open_by_default = bool(opt("eld_open_by_default", False))
    list_limit = max(1, int(opt("eld_prompt_list_limit", 50)))
    tabname = getattr(editor, "tabname", "txt2img")

    prompts_state = gr.State([])

    with gr.Accordion("Extended details", open=open_by_default, elem_classes=["eld-panel"]) as panel:
        summary = gr.HTML(elem_classes=["eld-summary"])

        with gr.Tabs(elem_classes=["eld-tabs"]):
            with gr.TabItem("Overview", elem_classes=["eld-tab"]):
                overview = gr.HTML(elem_classes=["eld-overview"])

            with gr.TabItem("Trigger words", elem_classes=["eld-tab"]):
                trigger_tags = gr.HighlightedText(
                    label="From the library — click a word to add it to the activation text",
                    visible=False,
                )
                trigger_text = gr.Textbox(
                    label="All trigger words",
                    lines=2,
                    max_lines=4,
                    interactive=False,
                    show_copy_button=True,
                )
                with gr.Row():
                    use_triggers = gr.Button("Use as activation text", size="sm")
                    append_triggers = gr.Button("Append to activation text", size="sm")

            with gr.TabItem("Prompts", elem_classes=["eld-tab"]):
                prompt_meta = gr.HTML(elem_classes=["eld-prompt-meta"])
                with gr.Row():
                    with gr.Column(scale=8):
                        prompt_selector = gr.Dropdown(
                            label="Prompt",
                            choices=[],
                            value=None,
                            interactive=True,
                            filterable=True,
                        )
                    with gr.Column(scale=1, min_width=110):
                        with gr.Row():
                            previous_button = gr.Button("‹", size="sm")
                            next_button = gr.Button("›", size="sm")
                prompt_box = gr.Textbox(
                    label="Positive prompt",
                    lines=6,
                    max_lines=14,
                    interactive=False,
                    show_copy_button=True,
                )
                with gr.Row():
                    append_prompt = gr.Button("Append to prompt", size="sm")
                    replace_prompt = gr.Button("Replace prompt", size="sm", variant="primary")

                with gr.Accordion("All prompts", open=False, elem_classes=["eld-all-prompts"]):
                    prompt_list = gr.HTML()

            with gr.TabItem("Raw CSV row", elem_classes=["eld-tab"]):
                raw_row = gr.Code(label="Matched row(s)", language="json", interactive=False)

    # ------------------------------------------------------------- populate

    def populate(name):
        blank = [
            gr.update(label="Extended details", open=open_by_default),
            "",
            "",
            gr.update(value=[], visible=False),
            "",
            gr.update(choices=[], value=None),
            "",
            "",
            "",
            "{}",
            [],
        ]

        try:
            data = gather(editor.page, name)
        except Exception:
            report(f'could not gather extended details for "{name}"')
            blank[1] = render.notice("Extended details could not be loaded — see the console for details.", kind="error")
            return blank

        records = data["records"]
        prompts = data["prompts"]
        triggers = data["triggers"]

        label = "Extended details"
        if records:
            pieces = [f"{len(records)} row{'s' if len(records) != 1 else ''}"]
            if triggers:
                pieces.append(f"{len(triggers)} trigger word{'s' if len(triggers) != 1 else ''}")
            if prompts:
                pieces.append(f"{len(prompts)} prompt{'s' if len(prompts) != 1 else ''}")
            label = "Extended details — " + ", ".join(pieces)
        else:
            label = "Extended details — no library match"

        choices = _prompt_choices(prompts)
        first_prompt = prompts[0] if prompts else ""
        should_open = open_by_default or (bool(records) and bool(opt("eld_open_on_match", False)))

        return [
            gr.update(label=label, open=should_open),
            render.summary_html(matched=bool(records), how=data["how"], records=records, identity=data["identity"]),
            render.overview_html(records),
            gr.update(value=list(triggers), visible=bool(triggers)),
            ", ".join(word for word, _ in triggers),
            gr.update(choices=choices, value=0 if choices else None),
            first_prompt,
            render.prompt_meta_html(len(prompts), 0, data["prompt_sources"][:1]),
            render.prompt_list_html(prompts, limit=list_limit),
            _raw_json(records),
            prompts,
        ]

    outputs = [
        panel,
        summary,
        overview,
        trigger_tags,
        trigger_text,
        prompt_selector,
        prompt_box,
        prompt_meta,
        prompt_list,
        raw_row,
        prompts_state,
    ]

    # A separate listener on the existing trigger: the stock dialog keeps working
    # even if this one raises.
    editor.button_edit.click(fn=populate, inputs=[editor.edit_name_input], outputs=outputs, show_progress=False)

    # ------------------------------------------------------------ navigation

    def select_prompt(index, prompts):
        prompts = prompts or []
        position = _clamp(index, len(prompts))
        text = prompts[position] if prompts else ""
        return text, render.prompt_meta_html(len(prompts), position, [])

    prompt_selector.change(
        fn=select_prompt,
        inputs=[prompt_selector, prompts_state],
        outputs=[prompt_box, prompt_meta],
        show_progress=False,
    )

    def step(index, prompts, delta):
        prompts = prompts or []
        if not prompts:
            return gr.update(), "", render.prompt_meta_html(0, 0, [])
        position = (_clamp(index, len(prompts)) + delta) % len(prompts)
        return position, prompts[position], render.prompt_meta_html(len(prompts), position, [])

    previous_button.click(
        fn=lambda index, prompts: step(index, prompts, -1),
        inputs=[prompt_selector, prompts_state],
        outputs=[prompt_selector, prompt_box, prompt_meta],
        show_progress=False,
    )
    next_button.click(
        fn=lambda index, prompts: step(index, prompts, 1),
        inputs=[prompt_selector, prompts_state],
        outputs=[prompt_selector, prompt_box, prompt_meta],
        show_progress=False,
    )

    # -------------------------------------------------------- trigger words

    activation = getattr(editor, "edit_activation_text", None)
    if activation is not None:

        def add_tag(current, event: gr.SelectData):
            word = event.value[0] if isinstance(event.value, (list, tuple)) else event.value
            words = [piece for piece in TRIGGER_SPLIT_RE.split(current or "") if piece.strip()]
            if word in words:
                return ", ".join(piece for piece in words if piece != word)
            return ", ".join([*words, word])

        trigger_tags.select(fn=add_tag, inputs=[activation], outputs=[activation], show_progress=False)

        use_triggers.click(fn=lambda text: text or "", inputs=[trigger_text], outputs=[activation], show_progress=False)

        def append_to_activation(current, text):
            words = [piece for piece in TRIGGER_SPLIT_RE.split(current or "") if piece.strip()]
            for word in TRIGGER_SPLIT_RE.split(text or ""):
                if word.strip() and word not in words:
                    words.append(word)
            return ", ".join(words)

        append_triggers.click(
            fn=append_to_activation,
            inputs=[activation, trigger_text],
            outputs=[activation],
            show_progress=False,
        )
    else:
        # Non-LoRA editors have no activation text field; hide the buttons.
        use_triggers.visible = False
        append_triggers.visible = False

    # -------------------------------------------------------- prompt actions

    append_prompt.click(
        fn=None,
        _js=f"function(text){{ eldAppendPrompt({json.dumps(tabname)}, text); return []; }}",
        inputs=[prompt_box],
        outputs=[],
        show_progress=False,
    )
    replace_prompt.click(
        fn=None,
        _js=f"function(text){{ eldReplacePrompt({json.dumps(tabname)}, text); return []; }}",
        inputs=[prompt_box],
        outputs=[],
        show_progress=False,
    )
