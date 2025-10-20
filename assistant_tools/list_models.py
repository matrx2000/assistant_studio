"""List models available on the Ollama server.

The module keeps its configuration self-contained by reading the environment
variables directly. The exported tool can be discovered automatically simply by
placing this file in the ``assistant_tools`` directory.
"""
from __future__ import annotations

import json
import os
from typing import List
from urllib import request as urlrequest

from openai import OpenAI

TOOL_NAME = "list_models"
"""Identifier used when the assistant wants to enumerate models."""

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
API_KEY = os.environ.get("OLLAMA_API_KEY", "local")

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "List available local models from the Ollama server.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def _ollama_host_base() -> str:
    base = OLLAMA_BASE_URL.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def invoke() -> str:
    """Return model metadata from either the OpenAI API or Ollama HTTP endpoint."""

    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)
        response = client.models.list()
        models: List[dict] = []
        for item in response.data:
            name = getattr(item, "id", None) or getattr(item, "root", None) or getattr(item, "owned_by", None)
            if name:
                models.append({"name": name})
        if models:
            return json.dumps({"source": "v1/models", "models": models})
    except Exception:
        pass  # Fall back to the lightweight HTTP endpoint below.

    try:
        request = urlrequest.Request(_ollama_host_base() + "/api/tags", method="GET")
        with urlrequest.urlopen(request, timeout=5) as response:
            payload = response.read().decode("utf-8", "replace")
        parsed = json.loads(payload)
        models = [
            {
                "name": model.get("name"),
                "size": model.get("size"),
                "modified_at": model.get("modified_at"),
            }
            for model in parsed.get("models", [])
        ]
        return json.dumps({"source": "api/tags", "models": models})
    except Exception as exc:  # pragma: no cover - network failures
        return json.dumps({"error": str(exc)})
