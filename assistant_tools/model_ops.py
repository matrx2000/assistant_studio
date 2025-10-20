from __future__ import annotations

import json
from typing import Callable, Dict, List
from urllib import request as _urlreq

from openai import OpenAI

from . import config


def list_models() -> str:
    try:
        client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.API_KEY)
        data = client.models.list()
        names = []
        for model in data.data:
            name = getattr(model, "id", None) or getattr(model, "root", None) or getattr(model, "owned_by", None)
            if name:
                names.append({"name": name})
        if names:
            return json.dumps({"source": "v1/models", "models": names})
    except Exception:
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
    except Exception as exc:
        return json.dumps({"error": str(exc)})


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
            except Exception as exc:
                errors.append(f"{path}:{exc}")

    return json.dumps({"status": "ok", "old": old, "new": config.get_current_model(), "errors": errors})


FUNCTIONS: Dict[str, Callable[..., str]] = {
    "list_models": list_models,
    "set_model": set_model,
}

TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_models",
            "description": "List available local models from the Ollama server.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_model",
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
    },
]
