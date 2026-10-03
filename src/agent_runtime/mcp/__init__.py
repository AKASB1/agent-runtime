"""MCP over stdio: the SDK client with a PID-owning lifecycle, tools registered as ``<server>.<tool>``."""

from agent_runtime.mcp.client import McpServer, McpServerConfig

__all__ = ["McpServer", "McpServerConfig"]
