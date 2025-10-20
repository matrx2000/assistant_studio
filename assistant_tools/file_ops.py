from __future__ import annotations

import json
import os
from typing import Callable, Dict, List


def read_file(path: str, max_bytes: int = 4096) -> str:
    norm = os.path.normpath(path)
    if os.path.isabs(path) or ".." in norm.split(os.sep):
        return json.dumps({"error": "unsafe_path"})
    if not os.path.exists(norm):
        return json.dumps({"error": "not_found"})
    try:
        with open(norm, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(max_bytes)
        return json.dumps({"path": norm, "preview": data})
    except Exception as e:
        return json.dumps({"error": str(e)})


def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    try:
        norm = os.path.normpath(path)
        if os.path.isabs(path) or ".." in norm.split(os.sep):
            return json.dumps({"error": "unsafe_path", "path": path})
        parent = os.path.dirname(norm)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if mode not in ("overwrite", "append"):
            return json.dumps({"error": "bad_mode", "allowed": ["overwrite", "append"]})
        m = "w" if mode == "overwrite" else "a"
        with open(norm, m, encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"status": "ok", "path": norm, "mode": mode, "bytes_written": len(content)})
    except Exception as e:
        return json.dumps({"error": str(e), "path": path})


FUNCTIONS: Dict[str, Callable[..., str]] = {
    "read_file": read_file,
    "write_file": write_file,
}

TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
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
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
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
    },
]
