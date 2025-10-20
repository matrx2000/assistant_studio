from __future__ import annotations

import json
from typing import Callable, Dict, List


def add(a: float, b: float) -> str:
    return json.dumps({"sum": a + b})


FUNCTIONS: Dict[str, Callable[..., str]] = {"add": add}

TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
]
