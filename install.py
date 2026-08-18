"""Dependency check, run by Forge when the extension is loaded."""

import launch

if not launch.is_installed("requests"):
    launch.run_pip("install requests", "requests for Extended LoRA Details")
