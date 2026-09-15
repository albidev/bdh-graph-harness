"""The MCP server module must import against both MCP SDK majors.

`requirements.txt` pins `mcp>=1.0,<2.0`, where `FastMCP` lives at
`mcp.server.fastmcp`. Under mcp 2.x that path is gone — FastMCP was renamed to
`MCPServer` and moved to `mcp.server.mcpserver` — so the module used to fail with
a bare `ModuleNotFoundError: No module named 'mcp.server.fastmcp'` that says
nothing about which pin was violated. It bites whenever the interpreter resolves
mcp 2.x: a shared venv, a future upgrade, or tooling invoking the suite with the
wrong python.

These tests pin the resolution contract, not a specific SDK version, so they
hold on whichever major the environment happens to have.
"""
from __future__ import annotations

import importlib.metadata as md

import pytest


def _mcp_major() -> int:
    try:
        return int(md.version("mcp").split(".", 1)[0])
    except Exception:  # pragma: no cover - mcp not installed at all
        pytest.skip("mcp is not installed")


def test_mcp_server_module_imports_on_this_sdk_major():
    """The module imports, whatever major is installed."""
    from bdh_graph_harness import mcp_server

    assert mcp_server.mcp is not None


def test_fastmcp_resolves_to_the_path_this_sdk_provides():
    """The resolved symbol comes from the location this major actually ships."""
    from bdh_graph_harness import mcp_server

    module = type(mcp_server.mcp).__module__
    major = _mcp_major()
    if major >= 2:
        assert module.startswith("mcp.server.mcpserver"), module
    else:
        assert module.startswith("mcp.server.fastmcp"), module


def test_server_has_the_transports_run_mcp_server_uses():
    """Both transports the entry point calls must exist on the resolved class."""
    from bdh_graph_harness import mcp_server

    assert hasattr(mcp_server.mcp, "run_stdio_async")
    assert hasattr(mcp_server.mcp, "streamable_http_app")


def test_tools_are_registered_and_served():
    """The decorators took effect: the server actually exposes its tools."""
    import asyncio

    from bdh_graph_harness import mcp_server

    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {getattr(t, "name", None) or t.get("name") for t in tools}
    assert {"query", "stats", "hebbian", "graph", "refresh"} <= names
