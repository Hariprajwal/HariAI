"""Unified Tool Registry and Execution Engine for HariAI.

Integrates local autonomous tools (Universal Document Converter, Web Search,
Memory Inspection) with external Model Context Protocol (MCP) servers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hari_ai.tools.converter import convert_document
from hari_ai.tools.mcp_client import MCPClient, MCPServerConfig, load_mcp_config


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable[..., Any]] = None
    server_name: Optional[str] = None  # None for built-in, server_name for MCP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "source": self.server_name or "built-in",
        }


class ToolRegistry:
    """Manages available autonomous tools and routes tool execution."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._mcp_clients: Dict[str, MCPClient] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        # 1. Document Converter
        self.register(
            ToolDefinition(
                name="convert_document",
                description=(
                    "Converts any document between PDF, Word (.docx), Markdown (.md), "
                    "Text (.txt), HTML (.html), JSON (.json), CSV (.csv), and RTF (.rtf). "
                    "If output_path is omitted, creates a file with the same name and target extension."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Path to the input document file to convert (e.g. 'hari.pdf').",
                        },
                        "target_format": {
                            "type": "string",
                            "description": "Target format extension (e.g. 'docx', 'pdf', 'txt', 'html', 'md', 'csv', 'json').",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional custom output destination path. If omitted, uses same name in same directory.",
                        },
                    },
                    "required": ["source"],
                },
                handler=self._handle_convert_document,
            )
        )

        # 2. Web Search
        self.register(
            ToolDefinition(
                name="web_search",
                description="Performs live search queries and research briefing summarization.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query or topic."}
                    },
                    "required": ["query"],
                },
                handler=self._handle_web_search,
            )
        )

        # 3. System Status
        self.register(
            ToolDefinition(
                name="system_status",
                description="Inspects active hardware acceleration, compute device, and runtime state.",
                parameters={"type": "object", "properties": {}},
                handler=self._handle_system_status,
            )
        )

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def connect_mcp_servers(self) -> Dict[str, int]:
        """Connects to configured MCP servers and registers discovered tools."""
        configs = load_mcp_config()
        discovered: Dict[str, int] = {}
        for name, conf in configs.items():
            if not conf.enabled:
                continue
            client = MCPClient(conf)
            if client.start():
                self._mcp_clients[name] = client
                tools = client.list_tools()
                for t in tools:
                    self.register(
                        ToolDefinition(
                            name=f"{name}__{t.name}",
                            description=f"[{name}] {t.description}",
                            parameters=t.parameters,
                            server_name=name,
                        )
                    )
                discovered[name] = len(tools)
        return discovered

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a tool by name, routing either to built-in handler or MCP server."""
        tool = self.get_tool(name)
        if not tool:
            return {"error": f"Tool '{name}' not found."}

        # MCP Tool execution
        if tool.server_name:
            client = self._mcp_clients.get(tool.server_name)
            if not client:
                return {"error": f"MCP server '{tool.server_name}' is not connected."}
            actual_tool_name = name.split("__", 1)[-1]
            return client.call_tool(actual_tool_name, arguments)

        # Built-in Tool execution
        if tool.handler:
            try:
                res = tool.handler(**arguments)
                return {"result": res}
            except Exception as exc:
                return {"error": f"Tool execution failed: {exc}"}

        return {"error": "No handler configured for this tool."}

    def _handle_convert_document(
        self,
        source: str,
        target_format: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        dest = convert_document(source, target_format, output_path)
        return f"Successfully converted '{source}' -> '{dest}' ({dest.stat().st_size} bytes)"

    def _handle_web_search(self, query: str) -> str:
        from hari_ai.memory.hierarchy import HierarchicalMemoryManager
        from hari_ai.search.engine import SearchEngine

        mem = HierarchicalMemoryManager()
        engine = SearchEngine(memory_manager=mem)
        return engine.search_and_summarize(query)

    def _handle_system_status(self) -> Dict[str, Any]:
        from hari_ai.runtime.hardware import detect_hardware

        hw = detect_hardware()
        return {
            "device_type": hw.device_type,
            "device_name": hw.device_name,
            "gpu_available": hw.gpu_available,
            "total_memory_gb": hw.total_memory_gb,
            "cpu_cores": hw.cpu_cores,
            "optimal_threads": hw.optimal_threads,
            "summary": hw.summary(),
        }


# Singleton registry instance
default_registry = ToolRegistry()
