"""Write UTF-8 text to disk tool.

Each tool module is intentionally self-contained: schema, logic, and helper
comments live side by side so dropping the file into the ``assistant_tools``
directory is all it takes to activate it.
"""
from __future__ import annotations

import json
from pathlib import Path

TOOL_NAME = "write_file"
"""Public identifier for the file writing helper."""

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Write UTF-8 text to a file (relative path only). Choose overwrite or append.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}


SAFE_ROOT = Path.cwd()
"""Base directory that the tool is allowed to modify."""


def _resolve_path(path: str) -> Path:
    candidate = (SAFE_ROOT / path).resolve()
    if not str(candidate).startswith(str(SAFE_ROOT.resolve())):
        raise ValueError("unsafe_path")
    return candidate


def invoke(path: str, content: str, mode: str = "overwrite") -> str:
    """Write ``content`` to ``path`` using the requested ``mode``."""

    try:
        target = _resolve_path(path)
    except ValueError:
        return json.dumps({"error": "unsafe_path", "path": path})

    if mode not in ("overwrite", "append"):
        return json.dumps({"error": "bad_mode", "allowed": ["overwrite", "append"]})

    target.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "w" if mode == "overwrite" else "a"

    try:
        with target.open(file_mode, encoding="utf-8") as fh:
            fh.write(content)
        return json.dumps({"status": "ok", "path": str(target), "mode": mode, "bytes_written": len(content)})
    except Exception as exc:  # pragma: no cover - filesystem error handling
        return json.dumps({"error": str(exc), "path": str(target)})
