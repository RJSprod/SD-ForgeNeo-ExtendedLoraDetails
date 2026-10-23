"""The panel added to Forge's LoRA "edit metadata" dialog.

The dialog is built by ``LoraUserMetadataEditor.create_editor()``, which calls
``create_default_editor_elems()`` → the LoRA fields → ``create_default_buttons()``.
Patching ``create_default_buttons`` therefore lets us add a panel without
rewriting any upstream code and without touching the checkpoint or embedding
editors. It is created there - below the stock fields - and then moved up to sit
right under the dialog's header row, so what Civitai knows about the LoRA comes
before the fields you edit (a setting puts it back at the bottom).

The layout follows the Inspect view of the Wan2GP LoRA browser: a title strip; a
sidebar with the facts, trigger words, description and version notes; and the
gallery prompts - the selected one large, beside its image, and every one of them
as a card.

Population is wired as a **second** listener on the same ``button_edit`` click
that upstream already uses, so the original outputs are untouched: if anything
in here breaks, the stock dialog still works.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from . import civitai, hashing, images, jobs, layouts, render, state
from .common import SAFETENSORS_EXTENSIONS, lora_directories, normalize_path, opt, report
from .library import get_library

# Stock fields the stylesheet arranges more compactly when the panel is present:
# the weight's class finds the row it shares with the activation text and the
# negative prompt. Only a class is added; the components are not touched.
STOCK_FIELD_CLASSES = {
    "slider_preferred_weight": "eld-stock-weight",
}


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


_ARCHITECTURES: dict[tuple[str, int, int], str] = {}


def _architecture(filename: str) -> str:
    """What the file's tensor names say it is - read from its header only."""
    if not filename or not filename.lower().endswith(SAFETENSORS_EXTENSIONS):
        return ""
    try:
        info = os.stat(filename)
    except OSError:
        return ""
    key = (os.path.abspath(filename), info.st_size, info.st_mtime_ns)
    if key not in _ARCHITECTURES:
        if len(_ARCHITECTURES) > 4096:
            _ARCHITECTURES.clear()
        _metadata, names = civitai.read_safetensors_header(Path(filename))
        _ARCHITECTURES[key] = civitai.local_architecture(names)
    return _ARCHITECTURES[key]


def _fetch_plan(filename: str, records) -> dict:
    """What the panel's fetch button would do, or ``{}`` when it has nothing to do.

    A LoRA the library has never seen is fetched with its folder; one Civitai
    could not resolve before is retried. A resolved one needs no button.
    """
    folder = os.path.dirname(os.path.abspath(filename)) if filename else ""
    if not folder or not os.path.isdir(folder):
        return {}
    if records and any(record.resolved for record in records):
        return {}

    label = os.path.basename(folder) or folder
    for root in lora_directories():
        try:
            if os.path.commonpath([folder, root]) == root:
                relative = os.path.relpath(folder, root)
                label = os.path.basename(root) if relative == "." else relative.replace(os.sep, "/")
                break
        except ValueError:
            continue
    return {"folder": folder, "label": label + "/", "retry": bool(records)}


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
    descriptions: list[tuple[str, str, str, str]] = []
    negatives: list[str] = []
    seen_triggers: set[str] = set()
    for record in records:
        if record.has_description:
            descriptions.append((record.title, record.source, record.description, record.version_description))
        for entry in record.entries:
            if entry.text and entry.text not in prompts:
                prompts.append(entry.text)
                prompt_images.append((entry.image_url, entry.image_id))
                prompt_sources.append(record.source)
        for word in record.trigger_words:
            if word not in seen_triggers:
                seen_triggers.add(word)
                triggers.append((word, record.source))
        for negative in record.negative_prompts:
            if negative not in negatives:
                negatives.append(negative)

    identity = {
        "SHA256": sha256 or ("not computed" if filename else ""),
        "Forge hash": item.get("shorthash") or "",
    }

    user_metadata = item.get("user_metadata") or {}

    return {
        "records": records,
        "how": how,
        "descriptions": descriptions,
        "prompts": prompts,
        "prompt_images": prompt_images,
        "prompt_sources": prompt_sources,
        "triggers": triggers,
        "negatives": negatives,
        "identity": identity,
        "computed": computed,
        "filename": filename,
        "architecture": _architecture(filename),
        "activation": str(user_metadata.get("activation text", "") or ""),
        "fetch": _fetch_plan(filename, records),
    }


# ------------------------------------------------------------------- helpers


def _prompt_choices(prompts: list[str]) -> list[tuple[str, int]]:
    width = max(1, int(opt("eld_prompt_label_width", 90)))
    return [(f"{index + 1}. {layouts.label(prompt, width)}", index) for index, prompt in enumerate(prompts)]


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
            gr.update(interactive=False, value="Download this image"),
        )

    local = images.find_local(url, image_id)
    if local is not None:
        done = gr.update(interactive=False, value="Downloaded", variant="secondary")
        if images.is_video(local):
            return (
                render.image_status_html(downloaded=True, name=local.name, url=url),
                gr.update(value=None, visible=False),
                done,
            )
        return (
            render.image_status_html(downloaded=True, name=local.name, url=url),
            gr.update(value=str(local), visible=True),
            done,
        )

    return (
        render.image_status_html(downloaded=False, name="", url=url),
        gr.update(value=None, visible=False),
        gr.update(interactive=True, value="Download this image", variant="primary"),
    )


def _cards(prompts: list[str], prompt_images: list) -> tuple[list[dict], int]:
    """One entry per card shown, plus how many prompts have their image saved."""
    limit = max(1, int(opt("eld_prompt_list_limit", 50)))
    cards: list[dict] = []
    saved = 0
    for index, prompt in enumerate(prompts):
        url, image_id = prompt_images[index] if index < len(prompt_images) else ("", "")
        local = images.find_local(url, image_id) if url else None
        if local is not None:
            saved += 1
        if index >= limit:
            continue
        video = local is not None and images.is_video(local)
        cards.append(
            {
                "index": index,
                "prompt": prompt,
                "url": url,
                "saved": local is not None,
                "video": video,
                "thumb": images.thumbnail_data_uri(local) if local is not None and not video else "",
            }
        )
    return cards, saved


def _gallery(prompts: list[str], prompt_images: list, *, tabname: str, bridge: str) -> tuple[str, str]:
    """The gallery's heading and its cards, both of which change on a download."""
    cards, saved = _cards(prompts, prompt_images)
    return (
        render.prompts_head_html(len(prompts), saved),
        render.prompt_cards_html(cards, total=len(prompts), tab=tabname, bridge=bridge),
    )


def _source_for(sources: list, index: int) -> str:
    """Which CSV a prompt came from - worth saying only when there are several."""
    if len(set(sources or [])) <= 1 or not (0 <= index < len(sources)):
        return ""
    return f"from {sources[index]}"


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


def _activation_target(editor) -> str:
    """The element id of the dialog's activation text box, giving it one if it
    has none, so the trigger-word chips can edit it from the page."""
    activation = getattr(editor, "edit_activation_text", None)
    if activation is None:
        return ""
    if not getattr(activation, "elem_id", None):
        activation.elem_id = f"{getattr(editor, 'id_part', 'eld')}_activation_text"
    return str(activation.elem_id)


def _tag_stock_fields(editor) -> None:
    for attribute, css_class in STOCK_FIELD_CLASSES.items():
        component = getattr(editor, attribute, None)
        if component is None:
            continue
        classes = getattr(component, "elem_classes", None) or []
        if isinstance(classes, str):
            classes = [classes]
        if css_class not in classes:
            component.elem_classes = [*classes, css_class]


def _place_after_header(editor, panel) -> bool:
    """Move the panel up to sit right below the dialog's header row - the row
    with the name, description, file table and preview."""
    box = getattr(editor, "box", None)
    children = getattr(box, "children", None)
    if not isinstance(children, list) or panel not in children:
        return False

    node = getattr(editor, "edit_name", None)
    while node is not None and getattr(node, "parent", None) is not box:
        node = getattr(node, "parent", None)
    if node is None or node not in children:
        return False

    children.remove(panel)
    children.insert(children.index(node) + 1, panel)
    return True


def _api_key() -> str:
    """The key the dialog's fetch uses: the remembered one, the setting, the
    environment - the same order the fetch tab fills its field in."""
    return (
        state.remembered_api_key()
        or str(opt("eld_civitai_api_key", "") or "").strip()
        or os.environ.get("CIVITAI_API_KEY", "")
    )


# ---------------------------------------------------------------- the panel


def build_panel(editor) -> None:
    """Create the panel and wire it into ``editor.button_edit``."""
    open_by_default = bool(opt("eld_open_by_default", False))
    tabname = getattr(editor, "tabname", "txt2img")
    id_part = getattr(editor, "id_part", f"{tabname}_lora_edit_user_metadata")
    bridge = f"{id_part}_eld_pick"
    target = _activation_target(editor)

    prompts_state = gr.State([])
    images_state = gr.State([])
    sources_state = gr.State([])
    fetch_state = gr.State({})

    with gr.Accordion("Extended details", open=open_by_default, elem_classes=["eld-panel"]) as panel:
        header = gr.HTML(elem_classes=["eld-header"])

        with gr.Row(equal_height=False, elem_classes=["eld-body"]):
            # ---------------------------------------------------- sidebar
            with gr.Column(scale=4, min_width=280, elem_classes=["eld-side"]):
                facts = gr.HTML(elem_classes=["eld-facts-html"])
                with gr.Column(visible=False, elem_classes=["eld-fetch"]) as fetch_group:
                    fetch_button = gr.Button("Fetch from Civitai", size="sm", variant="primary")
                    fetch_note = gr.HTML(elem_classes=["eld-fetch-note"])
                side = gr.HTML(elem_classes=["eld-side-html"])

            # ---------------------------------------------- gallery prompts
            with gr.Column(scale=7, min_width=480, visible=False, elem_classes=["eld-main"]) as main:
                prompts_head = gr.HTML(elem_classes=["eld-prompts-head"])
                with gr.Row(equal_height=True, elem_classes=["eld-picker"]):
                    prompt_selector = gr.Dropdown(
                        label="Prompt",
                        show_label=False,
                        container=False,
                        choices=[],
                        value=None,
                        interactive=True,
                        filterable=True,
                        scale=10,
                        min_width=160,
                        elem_classes=["eld-selector"],
                    )
                    previous_button = gr.Button("‹", size="sm", scale=0, min_width=48, elem_classes=["eld-step"])
                    next_button = gr.Button("›", size="sm", scale=0, min_width=48, elem_classes=["eld-step"])

                with gr.Row(equal_height=False, elem_classes=["eld-viewer"]):
                    with gr.Column(scale=4, min_width=220, elem_classes=["eld-viewer-media"]):
                        image_preview = gr.Image(
                            label="Gallery image",
                            show_label=False,
                            visible=False,
                            interactive=False,
                            show_download_button=True,
                            elem_classes=["eld-image-preview"],
                        )
                        image_status = gr.HTML(elem_classes=["eld-image-status"])
                        with gr.Row(elem_classes=["eld-actions"]):
                            download_image = gr.Button("Download this image", size="sm", variant="primary")
                            download_all_images = gr.Button("Download all images", size="sm")

                    with gr.Column(scale=5, min_width=260, elem_classes=["eld-viewer-text"]):
                        prompt_meta = gr.HTML(elem_classes=["eld-prompt-meta"])
                        prompt_box = gr.Textbox(
                            label="Positive prompt",
                            show_label=False,
                            lines=7,
                            max_lines=18,
                            interactive=False,
                            show_copy_button=True,
                            elem_classes=["eld-prompt-box"],
                        )
                        with gr.Row(elem_classes=["eld-actions"]):
                            append_prompt = gr.Button("Append to prompt", size="sm")
                            replace_prompt = gr.Button("Replace prompt", size="sm", variant="primary")

                cards = gr.HTML(elem_classes=["eld-cards-html"])

        # The cards' "view" and "download" reach Python through these two, the
        # same way Forge's own cards open this dialog.
        pick_text = gr.Textbox(visible=False, elem_id=bridge)
        pick_button = gr.Button(visible=False, elem_id=f"{bridge}_button")

    # ------------------------------------------------------------- populate

    outputs = [
        panel,
        header,
        facts,
        fetch_group,
        fetch_button,
        fetch_note,
        side,
        main,
        prompts_head,
        prompt_selector,
        prompt_meta,
        prompt_box,
        image_status,
        image_preview,
        download_image,
        download_all_images,
        cards,
        prompts_state,
        images_state,
        sources_state,
        fetch_state,
    ]

    def blank(message: str = "") -> list:
        return [
            gr.update(label="Extended details", open=open_by_default),
            message,
            "",
            gr.update(visible=False),
            gr.update(),
            "",
            "",
            gr.update(visible=False),
            "",
            gr.update(choices=[], value=None),
            "",
            "",
            "",
            gr.update(value=None, visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            [],
            [],
            [],
            {},
        ]

    def populate(name):
        try:
            data = gather(editor.page, name)
        except Exception:
            report(f'could not gather extended details for "{name}"')
            return blank(render.notice("Extended details could not be loaded — see the console for details.", kind="error"))

        records = data["records"]
        prompts = data["prompts"]
        triggers = data["triggers"]
        prompt_images = data["prompt_images"]
        sources = data["prompt_sources"]
        plan = data["fetch"]

        if records:
            pieces = []
            if triggers:
                pieces.append(f"{len(triggers)} trigger word{'s' if len(triggers) != 1 else ''}")
            if prompts:
                pieces.append(f"{len(prompts)} prompt{'s' if len(prompts) != 1 else ''}")
            if data["descriptions"]:
                pieces.append("description")
            resolved = [record for record in records if record.resolved]
            label = f"Extended details — {resolved[0].title}" if resolved else "Extended details — no Civitai details yet"
            if pieces:
                label += " · " + ", ".join(pieces)
        else:
            label = "Extended details — no library match"

        should_open = open_by_default or (bool(records) and bool(opt("eld_open_on_match", False)))

        active = {piece.strip() for piece in data["activation"].split(",") if piece.strip()}
        texts = render.descriptions_html(data["descriptions"])
        if not texts and any(record.resolved for record in records):
            texts = render.notice(
                "No description in the library yet. A fetch with <b>fill in missing text details</b> ticked "
                "picks it up.",
                kind="empty",
            )
        side_html = "".join(
            [
                render.triggers_html([word for word, _ in triggers], target=target, active=active),
                texts,
                render.negative_html(data["negatives"], tab=tabname),
                render.fields_html(records),
                render.raw_html(_raw_json(records)),
            ]
        )

        if plan:
            fetch_label = "Retry on Civitai" if plan["retry"] else "Fetch from Civitai"
            what = (
                f"Looks this LoRA up again, with the others in <code>{render.escape(plan['label'])}</code> "
                "Civitai could not resolve before."
                if plan["retry"]
                else f"Looks up the LoRAs in <code>{render.escape(plan['label'])}</code> the library does not have "
                "yet, this one included."
            )
            note = f'<div class="eld-muted">{what} It runs in the background; reopen this dialog when it is done.</div>'
        else:
            fetch_label, note = "Fetch from Civitai", ""

        choices = _prompt_choices(prompts)
        if prompts:
            status, preview, download_button = _image_state(prompt_images, 0)
        else:
            status, preview, download_button = "", gr.update(value=None, visible=False), gr.update(interactive=False)
        downloadable = sum(1 for url, _ in prompt_images if url)
        bulk = gr.update(
            interactive=bool(downloadable),
            value=f"Download all {downloadable} images" if downloadable else "No images in the library",
        )
        head, gallery = _gallery(prompts, prompt_images, tabname=tabname, bridge=bridge)

        return [
            gr.update(label=label, open=should_open),
            render.header_html(records=records, how=data["how"]),
            render.facts_html(records, identity=data["identity"], architecture=data["architecture"]),
            gr.update(visible=bool(plan)),
            gr.update(value=fetch_label, interactive=bool(plan)),
            note,
            side_html,
            gr.update(visible=bool(prompts)),
            head,
            gr.update(choices=choices, value=0 if choices else None),
            render.prompt_meta_html(len(prompts), 0, _source_for(sources, 0), layouts.parse(prompts[0]) if prompts else None),
            prompts[0] if prompts else "",
            status,
            preview,
            download_button,
            bulk,
            gallery,
            prompts,
            prompt_images,
            sources,
            plan,
        ]

    # A separate listener on the existing trigger: the stock dialog keeps working
    # even if this one raises.
    editor.button_edit.click(fn=populate, inputs=[editor.edit_name_input], outputs=outputs, show_progress=False)

    # ------------------------------------------------------------ navigation

    def viewer(prompts, prompt_images, sources, position):
        status, preview, button = _image_state(prompt_images or [], position)
        text = prompts[position] if prompts else ""
        meta = render.prompt_meta_html(len(prompts), position, _source_for(sources, position), layouts.parse(text) if text else None)
        return text, meta, status, preview, button

    def select_prompt(index, prompts, prompt_images, sources):
        prompts = prompts or []
        return viewer(prompts, prompt_images, sources, _clamp(index, len(prompts)))

    navigation_outputs = [prompt_box, prompt_meta, image_status, image_preview, download_image]

    prompt_selector.change(
        fn=select_prompt,
        inputs=[prompt_selector, prompts_state, images_state, sources_state],
        outputs=navigation_outputs,
        show_progress=False,
    )

    def step(index, prompts, prompt_images, sources, delta):
        prompts = prompts or []
        if not prompts:
            return (gr.update(), "", render.prompt_meta_html(0, 0), "", gr.update(visible=False), gr.update(interactive=False))
        position = (_clamp(index, len(prompts)) + delta) % len(prompts)
        return (position, *viewer(prompts, prompt_images, sources, position))

    previous_button.click(
        fn=lambda index, prompts, prompt_images, sources: step(index, prompts, prompt_images, sources, -1),
        inputs=[prompt_selector, prompts_state, images_state, sources_state],
        outputs=[prompt_selector, *navigation_outputs],
        show_progress=False,
    )
    next_button.click(
        fn=lambda index, prompts, prompt_images, sources: step(index, prompts, prompt_images, sources, 1),
        inputs=[prompt_selector, prompts_state, images_state, sources_state],
        outputs=[prompt_selector, *navigation_outputs],
        show_progress=False,
    )

    # ------------------------------------------------------- image downloads

    def download_at(position, prompt_images):
        """Download one prompt's image; returns an error notice, or ``""``."""
        url, image_id = prompt_images[position]
        try:
            result = images.download(url, image_id)
        except Exception as exc:  # noqa: BLE001 - surfaced next to the button
            report(f"could not download {url}")
            return render.notice(render.escape(f"{type(exc).__name__}: {exc}"), kind="error")
        return "" if result.ok else render.notice(render.escape(result.message), kind="error")

    def fetch_current_image(index, prompts, prompt_images):
        prompts = prompts or []
        prompt_images = prompt_images or []
        position = _clamp(index, len(prompt_images))
        if not (0 <= position < len(prompt_images)):
            return "", gr.update(visible=False), gr.update(interactive=False), gr.update(), gr.update()

        problem = download_at(position, prompt_images)
        status, preview, button = _image_state(prompt_images, position)
        head, gallery = _gallery(prompts, prompt_images, tabname=tabname, bridge=bridge)
        return problem + status, preview, button, head, gallery

    download_image.click(
        fn=fetch_current_image,
        inputs=[prompt_selector, prompts_state, images_state],
        outputs=[image_status, image_preview, download_image, prompts_head, cards],
    )

    def fetch_all_images(prompt_images, name):
        pairs = [(url, image_id) for url, image_id in (prompt_images or []) if url]
        if not pairs:
            return render.notice("There are no image URLs to download.", kind="empty")

        try:
            job = jobs.submit_image_downloads(pairs, label=str(name or ""))
        except Exception as exc:  # noqa: BLE001
            report("could not queue the image downloads")
            return render.notice(render.escape(f"{type(exc).__name__}: {exc}"), kind="error")

        return render.notice(
            f"Queued job #{job.id} for {len(pairs)} image(s). "
            "Progress is on the <b>Extended LoRA Details</b> tab; reopen this dialog to see them.",
            kind="ok",
        )

    download_all_images.click(
        fn=fetch_all_images,
        inputs=[images_state, editor.edit_name_input],
        outputs=[image_status],
        show_progress=False,
    )

    # ------------------------------------------------- the cards' actions

    def pick(value, selected, prompts, prompt_images):
        """``view:<n>`` shows prompt n above; ``download:<n>`` saves its image."""
        prompts = prompts or []
        prompt_images = prompt_images or []
        nothing = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        action, _, raw_index = str(value or "").partition(":")
        try:
            position = int(raw_index)
        except ValueError:
            return nothing
        if not (0 <= position < len(prompts)):
            return nothing

        if action == "view":
            return (position, gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

        if action == "download" and position < len(prompt_images):
            problem = download_at(position, prompt_images)
            head, gallery = _gallery(prompts, prompt_images, tabname=tabname, bridge=bridge)
            if _clamp(selected, len(prompts)) == position:
                status, preview, button = _image_state(prompt_images, position)
                return (gr.update(), problem + status, preview, button, head, gallery)
            note = problem or gr.update()
            return (gr.update(), note, gr.update(), gr.update(), head, gallery)

        return nothing

    pick_button.click(
        fn=pick,
        inputs=[pick_text, prompt_selector, prompts_state, images_state],
        outputs=[prompt_selector, image_status, image_preview, download_image, prompts_head, cards],
        show_progress=False,
    )

    # ------------------------------------------------------ fetch from here

    def fetch_folder(plan):
        plan = plan or {}
        folder = plan.get("folder") or ""
        if not folder or not os.path.isdir(folder):
            return render.notice("This LoRA's folder could not be found.", kind="error"), gr.update()

        key = _api_key()
        try:
            job = jobs.submit_scan(
                folder,
                api_key=key,
                recursive=False,
                incremental=True,
                retry_unresolved=bool(plan.get("retry")) or bool(opt("eld_retry_unresolved", False)),
                backfill_text=bool(opt("eld_backfill_text", True)),
                fetch_descriptions=bool(opt("eld_fetch_descriptions", True)),
                base_model_filter=str(opt("eld_base_model_filter", "") or "").strip(),
                require_explicit_label=bool(opt("eld_require_explicit_label", False)),
                request_delay=float(opt("eld_request_delay", 0.15)),
                timeout=float(opt("eld_request_timeout", 30.0)),
                max_images=int(opt("eld_max_images", 0)),
                keep_existing_rows=True,
            )
        except Exception as exc:  # noqa: BLE001
            report("could not queue the fetch")
            return render.notice(render.escape(f"{type(exc).__name__}: {exc}"), kind="error"), gr.update()

        key_note = "" if key else " No API key is set, so only public models are found."
        message = render.notice(
            f"Queued job #{job.id} for <code>{render.escape(plan.get('label') or folder)}</code>. Progress is on the "
            f"<b>Extended LoRA Details</b> tab; reopen this dialog when it has finished.{render.escape(key_note)}",
            kind="ok",
        )
        return message, gr.update(interactive=False, value="Fetch queued")

    fetch_button.click(fn=fetch_folder, inputs=[fetch_state], outputs=[fetch_note, fetch_button], show_progress=False)

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

    # ------------------------------------------------------------- placement

    _tag_stock_fields(editor)
    if opt("eld_panel_first", True):
        try:
            _place_after_header(editor, panel)
        except Exception:
            report("could not move the extended details panel up; it stays at the bottom")
