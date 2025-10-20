"""Assistant tool registry.

All tool implementations live in :mod:`assistant_tools.tools`. The chat worker
imports ``FUNCTIONS`` and ``TOOLS`` from this package so adding a new tool is as
simple as defining it in that module.
"""

from .tools import FUNCTIONS, TOOLS, register_tool, tool

__all__ = ["FUNCTIONS", "TOOLS", "register_tool", "tool"]
