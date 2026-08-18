"""Attach the extended panel to Forge's LoRA metadata editor.

Forge exposes no hook for extending an extra-networks detail dialog, so the
extension wraps one method on the LoRA editor class. The wrapper is idempotent
(``Reload UI`` re-runs every script but leaves ``ui_edit_user_metadata`` in
``sys.modules``), and it fails soft: if the panel cannot be built, the stock
dialog is still created exactly as before.
"""

from __future__ import annotations

from .common import log, report

MARKER = "_extended_lora_details_patched"


def _editor_class():
    """``LoraUserMetadataEditor`` - importable because Forge puts every
    extension root on ``sys.path`` before loading any script."""
    try:
        from ui_edit_user_metadata import LoraUserMetadataEditor  # type: ignore
    except Exception:
        return None
    return LoraUserMetadataEditor


def apply() -> bool:
    editor_class = _editor_class()
    if editor_class is None:
        log("the built-in LoRA extension was not found; the details panel is disabled")
        return False

    if getattr(editor_class, MARKER, False):
        return True

    original_create_default_buttons = editor_class.create_default_buttons

    def create_default_buttons(self):
        from .common import opt

        if opt("eld_enabled", True):
            try:
                from . import details_ui

                details_ui.build_panel(self)
            except Exception:
                report("could not build the extended details panel")

        return original_create_default_buttons(self)

    editor_class.create_default_buttons = create_default_buttons
    setattr(editor_class, MARKER, True)
    log("extended details panel attached to the LoRA metadata editor")
    return True
