"""Shared tool discovery helpers for the chat worker.

Each module inside ``assistant_tools`` that exposes ``FUNCTIONS`` (a mapping of
callable tool implementations) and ``TOOLS`` (the OpenAI tool specifications)
will be imported automatically.  This allows new tools to be added simply by
dropping a ``.py`` file in this package without modifying ``main.py``.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict, Iterable, List, Tuple


def _iter_tool_modules() -> Iterable[str]:
    """Yield discoverable module names within ``assistant_tools``.

    Private modules (prefixed with ``_``) and packages are intentionally
    ignored.  The resulting iterator is sorted to guarantee deterministic load
    order so that ``FUNCTIONS`` and ``TOOLS`` are stable across runs.
    """

    module_names = [
        info.name
        for info in pkgutil.iter_modules(__path__)  # type: ignore[name-defined]
        if not info.ispkg and not info.name.startswith("_")
    ]
    return sorted(module_names)


def load_tools() -> Tuple[Dict[str, Callable[..., str]], List[dict]]:
    """Import all tool modules and merge their exports.

    Returns a ``(FUNCTIONS, TOOLS)`` tuple ready to be consumed by the chat
    worker. Duplicate function names keep the last discovered implementation to
    mimic ``dict.update`` semantics while still allowing modules to override a
    tool when necessary.
    """

    functions: Dict[str, Callable[..., str]] = {}
    tools: List[dict] = []

    package = __name__
    for name in _iter_tool_modules():
        module = importlib.import_module(f"{package}.{name}")

        module_functions = getattr(module, "FUNCTIONS", {})
        module_tools = getattr(module, "TOOLS", [])

        if isinstance(module_functions, dict):
            functions.update(module_functions)
        if isinstance(module_tools, list):
            tools.extend(module_tools)

    return functions, tools


FUNCTIONS, TOOLS = load_tools()

__all__ = ["FUNCTIONS", "TOOLS", "load_tools"]
