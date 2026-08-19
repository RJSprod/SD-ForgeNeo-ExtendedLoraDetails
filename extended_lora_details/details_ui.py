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

from . import hashing, images, jobs, render
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
    prompt_images: list[tuple[str, str]] = []
    prompt_sources: list[str] = []
    triggers: list[tuple[str, str]] = []
    seen_triggers: set[str] = set()
    for record in records:
        for entry in record.entries:
            if entry.text and entry.text not in prompts:
                prompts.append(entry.text)
                prompt_images.append((entry.image_url, entry.image_id))
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
        "prompt_images": prompt_images,
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


def _image_state(prompt_images: list, index: int):
    """Status HTML plus the preview update for the prompt at ``index``.

    Presence is a filesystem check, so an image the user deleted goes straight
    back to being offered as a download.
    """
    url, image_id = ("", "")
    if 0 <= index < len(prompt_images):
        url, image_id = prompt_images[index]

    if not url:
        return (
            render.notice("This prompt has no image URL in the library. Re-run a Civitai fetch to pick one up.", kind="empty"),
            gr.update(value=None, visible=False),
            gr.update(interactive=False),
        )

    local = images.find_local(url, image_id)
    if local is not None:
        return (
            render.image_status_html(downloaded=True, name=local.name, url=url),
            gr.update(value=str(local), visible=True),
            gr.update(interactive=False, value="Downloaded"),
        )

    return (
        render.image_status_html(downloaded=False, name="", url=url),
        gr.update(value=None, visible=False),
        gr.update(interactive=True, value="Download this image"),
    )


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
    images_state = gr.State([])

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
                with gr.Row(elem_classes=["eld-actions"]):
                    use_triggers = gr.Button("Use as activation text")
                    append_triggers = gr.Button("Append to activation text")

            with gr.TabItem("Prompts", elem_classes=["eld-tab"]):
                prompt_meta = gr.HTML(elem_classes=["eld-prompt-meta"])

                prompt_selector = gr.Dropdown(
                    label="Prompt",
                    choices=[],
                    value=None,
                    interactive=True,
                    filterable=True,
                )
                with gr.Row(elem_classes=["eld-actions", "eld-stepper"]):
                    previous_button = gr.Button("‹  Previous")
                    next_button = gr.Button("Next  ›")

                prompt_box = gr.Textbox(
                    label="Positive prompt",
                    lines=8,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True,
                )
                with gr.Row(elem_classes=["eld-actions"]):
                    append_prompt = gr.Button("Append to prompt")
                    replace_prompt = gr.Button("Replace prompt", variant="primary")

                with gr.Group(elem_classes=["eld-image-block"]):
                    image_status = gr.HTML(elem_classes=["eld-image-status"])
                    with gr.Row(elem_classes=["eld-actions"]):
                        download_image = gr.Button("Download this image", variant="primary")
                        download_all_images = gr.Button("Download all images for this LoRA")
                    image_preview = gr.Image(
                        label="Gallery image",
                        visible=False,
                        interactive=False,
                        show_download_button=True,
                        elem_classes=["eld-image-preview"],
                    )

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
            [],
            "",
            gr.update(value=None, visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
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

        prompt_images = data["prompt_images"]
        status, preview, download_button = _image_state(prompt_images, 0) if prompts else ("", gr.update(value=None, visible=False), gr.update(interactive=False))
        downloadable = sum(1 for url, _ in prompt_images if url)
        bulk = gr.update(
            interactive=bool(downloadable),
            value=f"Download all {downloadable} images" if downloadable else "No images in the library",
        )

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
            prompt_images,
            status,
            preview,
            download_button,
            bulk,
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
        images_state,
        image_status,
        image_preview,
        download_image,
        download_all_images,
    ]

    # A separate listener on the existing trigger: the stock dialog keeps working
    # even if this one raises.
    editor.button_edit.click(fn=populate, inputs=[editor.edit_name_input], outputs=outputs, show_progress=False)

    # ------------------------------------------------------------ navigation

    def select_prompt(index, prompts, prompt_images):
        prompts = prompts or []
        position = _clamp(index, len(prompts))
        text = prompts[position] if prompts else ""
        status, preview, button = _image_state(prompt_images or [], position)
        return text, render.prompt_meta_html(len(prompts), position, []), status, preview, button

    navigation_outputs = [prompt_box, prompt_meta, image_status, image_preview, download_image]

    prompt_selector.change(
        fn=select_prompt,
        inputs=[prompt_selector, prompts_state, images_state],
        outputs=navigation_outputs,
        show_progress=False,
    )

    def step(index, prompts, prompt_images, delta):
        prompts = prompts or []
        if not prompts:
            return (gr.update(), "", render.prompt_meta_html(0, 0, []), "", gr.update(visible=False), gr.update(interactive=False))
        position = (_clamp(index, len(prompts)) + delta) % len(prompts)
        status, preview, button = _image_state(prompt_images or [], position)
        return position, prompts[position], render.prompt_meta_html(len(prompts), position, []), status, preview, button

    previous_button.click(
        fn=lambda index, prompts, prompt_images: step(index, prompts, prompt_images, -1),
        inputs=[prompt_selector, prompts_state, images_state],
        outputs=[prompt_selector, *navigation_outputs],
        show_progress=False,
    )
    next_button.click(
        fn=lambda index, prompts, prompt_images: step(index, prompts, prompt_images, 1),
        inputs=[prompt_selector, prompts_state, images_state],
        outputs=[prompt_selector, *navigation_outputs],
        show_progress=False,
    )

    # ------------------------------------------------------- image downloads

    def fetch_current_image(index, prompt_images):
        prompt_images = prompt_images or []
        position = _clamp(index, len(prompt_images))
        if not (0 <= position < len(prompt_images)):
            return "", gr.update(visible=False), gr.update(interactive=False)

        url, image_id = prompt_images[position]
        try:
            result = images.download(url, image_id)
        except Exception as exc:  # noqa: BLE001 - surfaced next to the button
            report(f"could not download {url}")
            return (
                render.notice(render.escape(f"{type(exc).__name__}: {exc}"), kind="error"),
                gr.update(visible=False),
                gr.update(interactive=True),
            )

        status, preview, button = _image_state(prompt_images, position)
        if not result.ok:
            status = render.notice(render.escape(result.message), kind="error") + status
        return status, preview, button

    download_image.click(
        fn=fetch_current_image,
        inputs=[prompt_selector, images_state],
        outputs=[image_status, image_preview, download_image],
    )

    def fetch_all_images(index, prompt_images, name):
        pairs = [(url, image_id) for url, image_id in (prompt_images or []) if url]
        if not pairs:
            return render.notice("There are no image URLs to download.", kind="empty"), gr.update(), gr.update()

        try:
            job = jobs.submit_image_downloads(pairs, label=str(name or ""))
        except Exception as exc:  # noqa: BLE001
            report("could not queue the image downloads")
            return render.notice(render.escape(f"{type(exc).__name__}: {exc}"), kind="error"), gr.update(), gr.update()

        message = render.notice(
            f"Queued job #{job.id} for {len(pairs)} image(s). "
            "Progress is on the <b>Extended LoRA Details</b> tab; reopen this dialog to see them.",
            kind="ok",
        )
        return message, gr.update(), gr.update()

    download_all_images.click(
        fn=fetch_all_images,
        inputs=[prompt_selector, images_state, editor.edit_name_input],
        outputs=[image_status, image_preview, download_image],
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
