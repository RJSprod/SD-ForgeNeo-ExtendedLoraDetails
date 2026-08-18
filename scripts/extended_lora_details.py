"""Extended LoRA Details — entry point.

Forge loads this file during ``load_scripts()``. All it does is register
callbacks; the real work lives in the ``extended_lora_details`` package at the
extension root, which is importable because Forge puts every extension root on
``sys.path`` before loading any script module.
"""

from __future__ import annotations

from modules import script_callbacks

from extended_lora_details import __version__, patch, settings_ui, tab_ui
from extended_lora_details.common import log, report


def on_ui_settings():
    try:
        settings_ui.register()
    except Exception:
        report("could not register settings")


def on_before_ui():
    # Runs after every script has loaded, so the built-in LoRA extension's
    # editor class is guaranteed to exist by now.
    try:
        patch.apply()
    except Exception:
        report("could not attach the details panel")


def on_ui_tabs():
    try:
        return tab_ui.create_tab()
    except Exception:
        report("could not build the extension tab")
        return []


def on_app_started(_demo, _app):
    try:
        from extended_lora_details.library import get_library

        get_library().ensure_loaded()
    except Exception:
        report("could not load the CSV library")


def on_script_unloaded():
    try:
        from extended_lora_details import hashing, jobs

        jobs.MANAGER.cancel_all()
        hashing.flush()
    except Exception:
        report("could not shut down cleanly")


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_ui(on_before_ui)
script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_app_started(on_app_started)
script_callbacks.on_script_unloaded(on_script_unloaded)

log(f"v{__version__} loaded")
