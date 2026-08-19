"""The "Extended LoRA Details" settings page.

Most entries are ordinary options. The CSV uploader is an option whose
``component`` factory builds a real ``gr.File`` (plus its status line and a
couple of buttons) inside the settings Blocks, and wires its own event handlers
there. It is marked ``do_not_save`` so the settings machinery treats it as a
display-only widget and never tries to persist a temp path into ``config.json``.
"""

from __future__ import annotations

import gradio as gr

from . import render
from .common import default_library_dir, report
from .images import default_image_dir
from .library import get_library

SECTION = ("extended_lora_details", "Extended LoRA Details")


def _library_status_html() -> str:
    try:
        return render.library_summary_html(get_library().stats())
    except Exception:
        report("could not read the library status")
        return render.notice("The library could not be read — see the console for details.", kind="error")


def _uploader_component(**kwargs):
    """Factory used as ``OptionInfo.component``; runs inside the settings page."""
    # The settings builder passes the option's stored value; a File component
    # must not be seeded with one.
    kwargs.pop("value", None)
    label = kwargs.pop("label", "") or "Add a CSV to the library"
    elem_id = kwargs.pop("elem_id", None)

    with gr.Column(elem_classes=["eld-settings-block"]):
        gr.HTML(
            render.notice(
                "Drop a CSV here to add it to the library. Rows are matched to a LoRA by "
                "<b>SHA256</b> when the CSV has a hash column, otherwise by file path or file name.",
                kind="info",
            )
        )
        uploader = gr.File(
            label=label,
            file_count="multiple",
            file_types=[".csv", "text/csv"],
            type="filepath",
            elem_id=elem_id,
            **kwargs,
        )
        status = gr.HTML(elem_classes=["eld-upload-status"])
        with gr.Row():
            refresh = gr.Button("Reload library from disk", size="sm")
            open_note = gr.Button("Show library contents", size="sm")
        library_status = gr.HTML(value=_library_status_html, elem_classes=["eld-library-status"])
        sources = gr.HTML(visible=False)

    def do_upload(paths):
        if not paths:
            return "", _library_status_html(), None

        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        messages = []
        for entry in paths:
            path = getattr(entry, "name", entry)
            try:
                ok, message = get_library().import_csv(path)
            except Exception as exc:  # noqa: BLE001 - reported to the user
                report(f"could not import {path}")
                ok, message = False, f"{type(exc).__name__}: {exc}"
            messages.append(render.notice(render.escape(message), kind="ok" if ok else "error"))

        return "".join(messages), _library_status_html(), None

    uploader.upload(
        fn=do_upload,
        inputs=[uploader],
        outputs=[status, library_status, uploader],
        show_progress=True,
    )

    def do_refresh():
        try:
            get_library().reload()
            message = render.notice("Library reloaded.", kind="ok")
        except Exception as exc:  # noqa: BLE001
            report("could not reload the library")
            message = render.notice(render.escape(f"{type(exc).__name__}: {exc}"), kind="error")
        return message, _library_status_html()

    refresh.click(fn=do_refresh, outputs=[status, library_status], show_progress=False)

    def do_list():
        try:
            html = render.sources_html(get_library().sources)
        except Exception:
            report("could not list the library sources")
            html = render.notice("The library could not be listed.", kind="error")
        return gr.update(value=html, visible=True)

    open_note.click(fn=do_list, outputs=[sources], show_progress=False)

    # The settings page collects the returned component into its inputs list.
    return uploader


def register() -> None:
    """Called from ``on_ui_settings``."""
    from modules import shared
    from modules.options import OptionHTML, OptionInfo, options_section

    def option(default, label, **kwargs):
        return OptionInfo(default, label, **kwargs)

    uploader = OptionInfo([], "Add a CSV to the library", component=_uploader_component, section=SECTION)
    uploader.do_not_save = True

    options = {
        "eld_intro": OptionHTML(
            "Show CSV-sourced Civitai details — model name, trigger words and gallery prompts — "
            "inside the LoRA card's <b>edit metadata</b> dialog. Run fetches and manage the library "
            "from the <b>Extended LoRA Details</b> tab."
        ),
        "eld_enabled": option(True, "Show the extended details panel in the LoRA dialog").needs_reload_ui(),
        "eld_library_dir": option(
            str(default_library_dir()),
            "Library directory (CSV files are read from here, recursively)",
        ).info("takes effect immediately"),
        "eld_upload_csv": uploader,
        "eld_lookup_header": OptionHTML("<h3>Matching</h3>"),
        "eld_hash_lookup": option(True, "Match by SHA256 first (falls back to file path, then file name)"),
        "eld_hash_on_open": option(
            True,
            "Compute a missing SHA256 when the dialog opens (first open of a large LoRA takes a moment)",
        ).info("turn this off and use “Precompute hashes” on the extension tab instead"),
        "eld_display_header": OptionHTML("<h3>Panel</h3>"),
        "eld_open_by_default": option(False, "Open the extended details panel automatically"),
        "eld_open_on_match": option(True, "Open it automatically when the library has a match"),
        "eld_prompt_list_limit": OptionInfo(
            50,
            "Prompts rendered in the “All prompts” list",
            gr.Slider,
            {"minimum": 5, "maximum": 500, "step": 5},
        ).info("the selector above it always reaches every prompt"),
        "eld_prompt_label_width": OptionInfo(
            90,
            "Characters of each prompt shown in the selector",
            gr.Slider,
            {"minimum": 30, "maximum": 200, "step": 5},
        ),
        "eld_images_header": OptionHTML("<h3>Gallery images</h3>"),
        "eld_image_dir": option(
            str(default_image_dir()),
            "Folder downloaded gallery images are saved to",
        ).info("one flat folder, so the whole collection is managed in one place"),
        "eld_image_max_mb": OptionInfo(
            32,
            "Largest image to download (MB)",
            gr.Slider,
            {"minimum": 1, "maximum": 256, "step": 1},
        ),
        "eld_image_timeout": OptionInfo(
            30.0,
            "Image download timeout (seconds)",
            gr.Slider,
            {"minimum": 5.0, "maximum": 180.0, "step": 5.0},
        ),
        "eld_civitai_header": OptionHTML("<h3>Civitai fetch</h3>"),
        "eld_civitai_api_key": option("", "Civitai API key").info(
            "optional, but improves coverage; falls back to the CIVITAI_API_KEY environment variable"
        ),
        "eld_request_delay": OptionInfo(
            0.15,
            "Minimum delay between Civitai requests (seconds)",
            gr.Slider,
            {"minimum": 0.0, "maximum": 5.0, "step": 0.05},
        ),
        "eld_request_timeout": OptionInfo(
            30.0,
            "Civitai request timeout (seconds)",
            gr.Slider,
            {"minimum": 5.0, "maximum": 180.0, "step": 5.0},
        ),
        "eld_max_images": OptionInfo(
            0,
            "Gallery images inspected per version (0 = all)",
            gr.Slider,
            {"minimum": 0, "maximum": 500, "step": 10},
        ),
        "eld_base_model_filter": option("", "Only accept this base model").info(
            'e.g. "Krea 2" or "Flux.1 D"; leave empty to accept any exact hash match'
        ),
        "eld_require_explicit_label": option(
            False,
            "Require an explicit base-model label (rejects unlabeled exact hash matches)",
        ),
        "eld_retry_unresolved": option(
            False,
            "On a refresh, look up LoRAs Civitai had nothing for last time",
        ).info("off by default: a refresh is for picking up new LoRAs, not re-asking about known misses"),
    }

    for key, info in options_section(SECTION, options).items():
        shared.opts.add_option(key, info)
