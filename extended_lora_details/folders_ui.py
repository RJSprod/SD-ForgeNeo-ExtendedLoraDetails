"""The "Preset folders" tab: assign folders to a UI Preset.

One accordion per extra-network page (Lora, Checkpoints, Textual Inversion, and
anything else registered). Inside each, two lists:

* **Folders** - every folder under that network's roots; tick the ones the preset
  should show.
* **…and include their subfolders** - only the folders just ticked, so taking a
  whole subtree is always a separate, explicit choice rather than something
  implied by picking a parent folder.

Edits are written to ``preset_folders.json`` as they happen; the network
browsers pick them up on their next rebuild - automatically when the UI Preset
changes, or straight away via *Apply to the network browsers* here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import gradio as gr

from . import preset_folders, render
from .common import opt, report
from .preset_folders import Rule


@dataclass
class PageEntry:
    """One extra-network page and the components editing its assignments."""

    key: str
    title: str
    roots: list[str]
    folders: object = None
    recursive: object = None
    clear: object = None
    choices: list = field(default_factory=list)


def registered_pages() -> list:
    try:
        from modules import ui_extra_networks

        return list(ui_extra_networks.extra_pages)
    except Exception:
        return []


def _roots_for(page) -> list[str]:
    try:
        roots = page.allowed_directories_for_previews() or []
    except Exception:
        report(f"could not read the folders of the {getattr(page, 'title', '?')} page")
        return []

    found: list[str] = []
    for root in roots:
        text = str(root or "").strip()
        if not text:
            continue
        absolute = os.path.abspath(os.path.expanduser(text))
        if absolute not in found and os.path.isdir(absolute):
            found.append(absolute)
    return found


def page_entries() -> list[PageEntry]:
    entries: list[PageEntry] = []
    for page in registered_pages():
        key = preset_folders.page_key(page)
        if not key or any(entry.key == key for entry in entries):
            continue
        entries.append(PageEntry(key=key, title=str(getattr(page, "title", key)), roots=_roots_for(page)))
    return entries


def _choices_for(roots, extra_paths=()) -> list[tuple[str, str]]:
    """Folder choices, plus any assigned folder that is no longer on disk."""
    try:
        choices = preset_folders.folder_choices(roots)
    except Exception:
        report("could not list the folders of a network type")
        choices = []
    known = {preset_folders.normalize_dir(value) for _, value in choices}
    for path in extra_paths:
        if preset_folders.normalize_dir(path) not in known:
            choices.append((f"{path}  (missing)", path))
    return choices


def _state_for(preset: str, entry: PageEntry):
    """``(choices, selected, subfolder choices, subfolder selection)`` for a page."""
    rules = preset_folders.get_store().rules(preset, entry.key)
    choices = _choices_for(entry.roots, [rule.path for rule in rules])
    entry.choices = choices

    canonical = {preset_folders.normalize_dir(value): value for _, value in choices}
    label_of = {value: label for label, value in choices}

    selected: list[str] = []
    recursive: list[str] = []
    for rule in rules:
        value = canonical.get(preset_folders.normalize_dir(rule.path), rule.path)
        selected.append(value)
        if rule.subfolders:
            recursive.append(value)

    sub_choices = [(label_of.get(value, value), value) for value in selected]
    return choices, selected, sub_choices, recursive


def _updates_for(preset: str, entry: PageEntry):
    choices, selected, sub_choices, recursive = _state_for(preset, entry)
    return (
        gr.update(choices=choices, value=selected),
        gr.update(choices=sub_choices, value=recursive),
    )


def _summary(preset: str, entries: list[PageEntry]) -> str:
    store = preset_folders.get_store()
    blocks = []
    for entry in entries:
        rows = [(rule.path, rule.subfolders, os.path.isdir(rule.path)) for rule in store.rules(preset, entry.key)]
        blocks.append((entry.title, rows))
    return render.preset_folder_summary_html(preset, blocks)


def _save(preset: str, entry: PageEntry, selected, recursive) -> str:
    preset = str(preset or "").strip()
    if not preset:
        return render.notice("Pick a UI Preset first.", kind="error")

    recursive_set = {preset_folders.normalize_dir(path) for path in (recursive or [])}
    rules = [Rule(path, preset_folders.normalize_dir(path) in recursive_set) for path in (selected or [])]

    try:
        preset_folders.get_store().set_rules(preset, entry.key, rules)
    except Exception as exc:  # noqa: BLE001 - shown to the user
        report("could not save the folder assignments")
        return render.notice(render.escape(f"{type(exc).__name__}: {exc}"), kind="error")

    if not rules:
        return render.notice(
            f"No folders assigned to <b>{render.escape(preset)}</b> for <b>{render.escape(entry.title)}</b> — "
            "that browser is left unfiltered for this preset.",
            kind="info",
        )

    subtrees = sum(1 for rule in rules if rule.subfolders)
    tail = f", {subtrees} with their subfolders" if subtrees else ", none including subfolders"
    return render.notice(
        f"Saved {len(rules)} folder(s) for <b>{render.escape(entry.title)}</b> under "
        f"<b>{render.escape(preset)}</b>{tail}.",
        kind="ok",
    )


def _active_note() -> str:
    active = preset_folders.current_preset()
    enabled = bool(opt("eld_folder_filter_enabled", True))
    strict = bool(opt("eld_folder_filter_strict", False))

    if active:
        parts = [f"Forge is on the <b>{render.escape(active)}</b> UI Preset."]
    else:
        parts = ["Forge has not reported a UI Preset yet."]

    if enabled:
        parts.append(
            "Network browsers show only the folders assigned to the active preset; a network type with "
            "no folders assigned "
            + ("shows nothing (strict mode is on)." if strict else "keeps showing everything.")
        )
    else:
        parts.append(
            "Filtering is <b>off</b> — turn it on in "
            "<b>Settings → Extended LoRA Details → UI Preset folder filter</b>."
        )
    return render.notice(" ".join(parts), kind="info" if enabled else "empty")


def create_ui() -> None:
    """Build the tab's contents inside the caller's Blocks / TabItem context."""
    entries = page_entries()

    gr.HTML(
        render.notice(
            "Assign folders to a <b>UI Preset</b>. While that preset is active, every network browser — "
            "Lora, Checkpoints, Textual Inversion, anything else installed — lists only those folders, "
            "and only their quick-navigation buttons.<br>"
            "Ticking a folder takes <b>that folder alone</b>; add it to the second list to take everything "
            "nested under it as well.",
            kind="info",
        )
    )
    active_note = gr.HTML(value=_active_note)

    if not entries:
        gr.HTML(
            render.notice(
                "No extra-network pages were found — open this tab again once the UI has finished loading.",
                kind="empty",
            )
        )
        return

    initial_preset = preset_folders.current_preset() or (preset_folders.preset_names() or [""])[0]

    with gr.Row():
        preset = gr.Dropdown(
            label="UI Preset to edit",
            choices=preset_folders.preset_names(),
            value=initial_preset or None,
            interactive=True,
            scale=3,
        )
        rescan = gr.Button("Rescan folders", size="sm", scale=1)
        apply_now = gr.Button("Apply to the network browsers", variant="primary", scale=2)

    status = gr.HTML()

    for index, entry in enumerate(entries):
        with gr.Accordion(entry.title, open=index == 0):
            if not entry.roots:
                gr.HTML(
                    render.notice(
                        f"No folders are configured for {render.escape(entry.title)}.",
                        kind="empty",
                    )
                )
                continue

            gr.HTML(
                render.notice(
                    "Roots: " + ", ".join(f"<code>{render.escape(root)}</code>" for root in entry.roots),
                    kind="empty",
                )
            )

            choices, selected, sub_choices, recursive = _state_for(initial_preset, entry)
            entry.folders = gr.CheckboxGroup(
                label="Folders shown for this preset",
                choices=choices,
                value=selected,
                interactive=True,
                info="nothing ticked leaves this browser unfiltered for the preset",
            )
            entry.recursive = gr.CheckboxGroup(
                label="…and include their subfolders",
                choices=sub_choices,
                value=recursive,
                interactive=True,
                info="a folder left unticked here contributes only the models sitting directly in it",
            )
            entry.clear = gr.Button(f"Clear {entry.title}", size="sm")

    summary = gr.HTML(value=lambda: _summary(initial_preset, entries))

    # ------------------------------------------------------------------ events

    editable = [entry for entry in entries if entry.folders is not None]

    def _bind(component, **kwargs):
        """Prefer ``.input`` (user edits only) and fall back to ``.change``."""
        handler = getattr(component, "input", None) or component.change
        handler(**kwargs)

    for entry in editable:

        def do_folders(preset_value, selected_value, recursive_value, entry=entry):
            keep = [path for path in (recursive_value or []) if path in (selected_value or [])]
            message = _save(preset_value, entry, selected_value, keep)
            label_of = {value: label for label, value in entry.choices}
            sub = [(label_of.get(value, value), value) for value in (selected_value or [])]
            return gr.update(choices=sub, value=keep), message, _summary(preset_value, entries)

        def do_recursive(preset_value, selected_value, recursive_value, entry=entry):
            return _save(preset_value, entry, selected_value, recursive_value), _summary(preset_value, entries)

        def do_clear(preset_value, entry=entry):
            message = _save(preset_value, entry, [], [])
            return (
                gr.update(value=[]),
                gr.update(choices=[], value=[]),
                message,
                _summary(preset_value, entries),
            )

        _bind(
            entry.folders,
            fn=do_folders,
            inputs=[preset, entry.folders, entry.recursive],
            outputs=[entry.recursive, status, summary],
            show_progress=False,
        )
        _bind(
            entry.recursive,
            fn=do_recursive,
            inputs=[preset, entry.folders, entry.recursive],
            outputs=[status, summary],
            show_progress=False,
        )
        entry.clear.click(
            fn=do_clear,
            inputs=[preset],
            outputs=[entry.folders, entry.recursive, status, summary],
            show_progress=False,
        )

    group_outputs = []
    for entry in editable:
        group_outputs.extend([entry.folders, entry.recursive])

    def do_preset(preset_value):
        updates = []
        for entry in editable:
            updates.extend(_updates_for(preset_value, entry))
        return [*updates, "", _summary(preset_value, entries), _active_note()]

    preset.change(
        fn=do_preset,
        inputs=[preset],
        outputs=[*group_outputs, status, summary, active_note],
        show_progress=False,
    )
    rescan.click(
        fn=do_preset,
        inputs=[preset],
        outputs=[*group_outputs, status, summary, active_note],
    )
    apply_now.click(
        fn=lambda: render.notice("Rebuilding the network browsers…", kind="ok"),
        outputs=[status],
        show_progress=False,
    ).then(fn=lambda: None, _js="eldRefreshExtraNetworks")
