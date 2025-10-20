"""Switch the active chat model.

This module holds the only bit of shared state required by the UI: the currently
selected model. Because the state and tool definition live in the same file, the
behaviour stays transparent even though the loader discovers it dynamically.
"""
from __future__ import annotations

import json
import os
from typing import List
from urllib import request as urlrequest

from openai import OpenAI

TOOL_NAME = "set_model"
"""Name used when the assistant requests a model change."""

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
API_KEY = os.environ.get("OLLAMA_API_KEY", "local")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "devstral:latest")

_CURRENT_MODEL = DEFAULT_MODEL
"""In-memory store of the most recently selected model."""

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Switch the active chat model. Optionally ensure it is loaded and unload the previous one.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "ensure_loaded": {"type": "boolean", "default": True},
                "unload_previous": {"type": "boolean", "default": False},
            },
            "required": ["model"],
            "additionalProperties": False,
        },
    },
}


def get_current_model() -> str:
    """Return the last requested model (or the default)."""

    return _CURRENT_MODEL or DEFAULT_MODEL


def set_current_model(model: str) -> None:
    """Update the in-memory record of the selected model."""

    global _CURRENT_MODEL
    _CURRENT_MODEL = model


def reset_current_model() -> None:
    """Reset state to the default model from the environment."""

    global _CURRENT_MODEL
    _CURRENT_MODEL = DEFAULT_MODEL


def _ollama_host_base() -> str:
    base = OLLAMA_BASE_URL.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _ensure_model_loaded(model: str) -> None:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=4,
    )


def _unload_model(model: str) -> List[str]:
    """Best-effort attempt to unload a model via the Ollama HTTP API."""

    host = _ollama_host_base()
    errors: List[str] = []
    for path, body in (
        ("/api/unload", {"model": model}),
        ("/api/stop", {"model": model}),
        ("/api/stop", {"name": model}),
    ):
        try:
            request = urlrequest.Request(
                host + path,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(request, timeout=3):
                pass
            break
        except Exception as exc:  # pragma: no cover - network best effort
            errors.append(f"{path}:{exc}")
    return errors


def invoke(model: str, ensure_loaded: bool = True, unload_previous: bool = False) -> str:
    """Switch to ``model`` and optionally load/unload around the change."""

    previous = get_current_model()
    set_current_model(model)
    errors: List[str] = []

    if ensure_loaded:
        try:
            _ensure_model_loaded(model)
        except Exception as exc:
            set_current_model(previous)
            return json.dumps({"status": "failed", "error": f"ensure_failed: {exc}", "old": previous, "new": model})

    if unload_previous and previous and previous != model:
        errors.extend(_unload_model(previous))

    return json.dumps({"status": "ok", "old": previous, "new": get_current_model(), "errors": errors})


EXPORTED_ATTRIBUTES = [
    "get_current_model",
    "set_current_model",
    "reset_current_model",
    "OLLAMA_BASE_URL",
    "API_KEY",
    "DEFAULT_MODEL",
]
