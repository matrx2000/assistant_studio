from __future__ import annotations

"""All assistant tool implementations live in this module.

To add a new tool, define a function that returns a JSON-serialisable string and
register it with :func:`register_tool` (or use the :func:`tool` decorator). The
module exports ``FUNCTIONS`` and ``TOOLS`` which the chat worker imports
without needing any other changes.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional
from urllib import request as _urlreq

from openai import OpenAI

from . import config

FUNCTIONS: Dict[str, Callable[..., str]] = {}
TOOLS: List[dict] = []


def register_tool(
    name: str,
    fn: Callable[..., str],
    *,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable[..., str]:
    """Register ``fn`` as an assistant tool."""

    if not callable(fn):
        raise TypeError("fn must be callable")

    FUNCTIONS[name] = fn
    TOOLS.append(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
    )
    return fn


def tool(*, name: Optional[str] = None, description: str, parameters: Optional[Dict[str, Any]] = None):
    """Decorator variant of :func:`register_tool`."""

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        register_tool(name or fn.__name__, fn, description=description, parameters=parameters)
        return fn

    return decorator


@tool(
    description="Add two numbers and return the sum.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    },
)
def add(a: float, b: float) -> str:
    return json.dumps({"sum": a + b})


@tool(
    description="Read a small UTF-8 text file (relative path, up to ~4KB).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_bytes": {"type": "integer", "default": 4096, "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)
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
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps({"error": str(e)})


@tool(
    description="Write UTF-8 text to a file (relative path only). Choose overwrite or append.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
)
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
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps({"error": str(e), "path": path})


@tool(
    description="List available local models from the Ollama server.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)
def list_models() -> str:
    try:
        client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.API_KEY)
        data = client.models.list()
        names: List[dict] = []
        for model in data.data:
            name = getattr(model, "id", None) or getattr(model, "root", None) or getattr(model, "owned_by", None)
            if name:
                names.append({"name": name})
        if names:
            return json.dumps({"source": "v1/models", "models": names})
    except Exception:  # pragma: no cover - fall back to Ollama HTTP API
        pass

    try:
        url = config.get_ollama_host_base().rstrip("/") + "/api/tags"
        req = _urlreq.Request(url, method="GET")
        with _urlreq.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", "replace")
        obj = json.loads(raw)
        models = [
            {
                "name": m.get("name"),
                "size": m.get("size"),
                "modified_at": m.get("modified_at"),
            }
            for m in obj.get("models", [])
        ]
        return json.dumps({"source": "api/tags", "models": models})
    except Exception as exc:  # pragma: no cover - network failure fallback
        return json.dumps({"error": str(exc)})


@tool(
    description="Switch the active chat model. Optionally ensure it is loaded and unload the previous one.",
    parameters={
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "ensure_loaded": {"type": "boolean", "default": True},
            "unload_previous": {"type": "boolean", "default": False},
        },
        "required": ["model"],
        "additionalProperties": False,
    },
)
def set_model(model: str, ensure_loaded: bool = True, unload_previous: bool = False) -> str:
    errors: List[str] = []
    old = config.get_current_model()
    config.set_current_model(model)

    if ensure_loaded:
        try:
            client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.API_KEY)
            _ = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=4,
            )
        except Exception as exc:
            config.set_current_model(old)
            return json.dumps(
                {
                    "status": "failed",
                    "error": f"ensure_failed: {exc}",
                    "old": old,
                    "new": model,
                }
            )

    if unload_previous and old and old != model:
        host = config.get_ollama_host_base().rstrip("/")
        for path, body in (
            ("/api/unload", {"model": old}),
            ("/api/stop", {"model": old}),
            ("/api/stop", {"name": old}),
        ):
            try:
                req = _urlreq.Request(
                    host + path,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                _ = _urlreq.urlopen(req, timeout=3).read()
                break
            except Exception as exc:  # pragma: no cover - best effort cleanup
                errors.append(f"{path}:{exc}")

    return json.dumps({"status": "ok", "old": old, "new": config.get_current_model(), "errors": errors})


__all__ = [
    "FUNCTIONS",
    "TOOLS",
    "register_tool",
    "tool",
    "add",
    "read_file",
    "write_file",
    "list_models",
    "set_model",
]
