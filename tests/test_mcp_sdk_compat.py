"""The MCP server module must import against both MCP SDK majors.

`requirements.txt` currently requires `mcp>=2.3.0,<3.0`. SDK v1 exposes
`FastMCP` at `mcp.server.fastmcp`. Under SDK v2 that path is gone — FastMCP was renamed to
`MCPServer` and moved it to `mcp.server.mcpserver`. A bare v1-only import used to
fail with `ModuleNotFoundError` whenever an environment resolved SDK v2; the
compatibility import in `mcp_server.py` now handles that relocation.

These tests validate the import, transports, and registered tools for whichever
SDK major is installed. CI currently exercises SDK v2; it does not run a separate
SDK v1 job.
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
