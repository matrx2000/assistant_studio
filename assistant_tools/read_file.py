"""Read a UTF-8 text file tool.

Like every tool module, this file keeps the runtime metadata next to the
implementation so dropping it into the :mod:`assistant_tools` package is enough
for the chat worker to pick it up automatically.
"""
from __future__ import annotations

import json
from pathlib import Path

TOOL_NAME = "read_file"
"""Identifier used when the LLM requests this tool."""

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Read a small UTF-8 text file (relative path, up to ~4KB).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "default": 4096, "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


SAFE_ROOT = Path.cwd()
"""Used to restrict requests to the current working tree."""


def _resolve_path(path: str) -> Path:
    candidate = (SAFE_ROOT / path).resolve()
    if not str(candidate).startswith(str(SAFE_ROOT.resolve())):
        raise ValueError("unsafe_path")
    return candidate


def invoke(path: str, max_bytes: int = 4096) -> str:
    """Return a preview of ``path`` limited to ``max_bytes`` characters."""

    try:
        target = _resolve_path(path)
    except ValueError:
        return json.dumps({"error": "unsafe_path", "path": path})

    if not target.exists():
        return json.dumps({"error": "not_found", "path": str(target)})

    try:
        preview = target.read_text(encoding="utf-8", errors="replace")[:max_bytes]
        return json.dumps({"path": str(target), "preview": preview})
    except Exception as exc:  # pragma: no cover - filesystem error handling
        return json.dumps({"error": str(exc), "path": str(target)})
