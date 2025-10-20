"""Dynamic assistant tool loader.

This package automatically discovers standalone tool modules stored alongside
this file. Each module must provide three core objects:

``TOOL_NAME``
    Unique string that will be used to register the function with the model.

``TOOL_SPEC``
    OpenAI-compatible tool description dictionary. The module should populate
    the ``function`` block with the same ``name`` so the runtime can expose it
    to the chat completions API.

``invoke(**kwargs)``
    Callable that executes the tool. The function receives the decoded
    arguments from the LLM and must return a JSON-serialisable string.

Modules may also declare ``EXPORTED_ATTRIBUTES`` with extra helper names that
should be re-exported from :mod:`assistant_tools`. This makes it easy for the
UI layer to access shared state (for example, the currently selected model).

To create a new tool simply drop a ``.py`` file in this directory following the
pattern above—no additional registration code is necessary.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import pkgutil
from types import ModuleType
from typing import Callable, Dict, Iterable, List

FUNCTIONS: Dict[str, Callable[..., str]] = {}
"""Mapping of tool name to the callable used at runtime."""

TOOLS: List[dict] = []
"""List of tool specification dictionaries consumed by OpenAI chat completions."""

__all__ = ["FUNCTIONS", "TOOLS"]


def _iter_tool_modules() -> Iterable[str]:
    base_path = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(base_path)]):
        if module_info.ispkg or module_info.name.startswith("__"):
            continue
        yield module_info.name


def _load_module(name: str) -> ModuleType:
    module = import_module(f"{__name__}.{name}")
    tool_name = getattr(module, "TOOL_NAME", None)
    tool_spec = getattr(module, "TOOL_SPEC", None)
    tool_callable = getattr(module, "invoke", None)

    if not tool_name or not isinstance(tool_name, str):
        raise ValueError(f"assistant tool module '{name}' must define TOOL_NAME (str)")
    if not isinstance(tool_spec, dict):
        raise ValueError(f"assistant tool module '{name}' must define TOOL_SPEC (dict)")
    if not callable(tool_callable):
        raise ValueError(f"assistant tool module '{name}' must define an 'invoke' callable")

    FUNCTIONS[tool_name] = tool_callable  # type: ignore[assignment]
    TOOLS.append(tool_spec)

    for attr in getattr(module, "EXPORTED_ATTRIBUTES", []):
        globals()[attr] = getattr(module, attr)
        if attr not in __all__:
            __all__.append(attr)

    return module


for module_name in sorted(_iter_tool_modules()):
    _load_module(module_name)
