from __future__ import annotations

import os
from typing import Optional


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "devstral:latest")
API_KEY = os.environ.get("OLLAMA_API_KEY", "local")

_CURRENT_MODEL: Optional[str] = DEFAULT_MODEL


def get_current_model() -> str:
    return _CURRENT_MODEL or DEFAULT_MODEL


def set_current_model(model: str) -> None:
    global _CURRENT_MODEL
    _CURRENT_MODEL = model


def reset_current_model() -> None:
    global _CURRENT_MODEL
    _CURRENT_MODEL = DEFAULT_MODEL


def get_ollama_host_base() -> str:
    base = OLLAMA_BASE_URL.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


__all__ = [
    "API_KEY",
    "DEFAULT_MODEL",
    "OLLAMA_BASE_URL",
    "get_current_model",
    "get_ollama_host_base",
    "reset_current_model",
    "set_current_model",
]
