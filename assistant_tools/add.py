"""Addition helper tool.

This module demonstrates the minimal structure required for a tool:

* ``TOOL_NAME`` identifies the function for the assistant runtime.
* ``TOOL_SPEC`` mirrors the metadata passed to the OpenAI Chat Completions API.
* ``invoke`` performs the actual work and returns a JSON-serialisable string.

Copy this file when creating a new tool so everything the LLM needs—the
implementation and the schema—lives together in one place.
"""
from __future__ import annotations

import json

TOOL_NAME = "add"
"""Public name for the tool exposed to the assistant."""

TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
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


def invoke(a: float, b: float) -> str:
    """Return the sum of ``a`` and ``b`` as a JSON payload."""

    return json.dumps({"sum": a + b})
