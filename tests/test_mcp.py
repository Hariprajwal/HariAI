"""Unit tests for HariAI Model Context Protocol (MCP) and Tool Registry."""
import unittest

from hari_ai.tools.mcp_client import MCPServerConfig
from hari_ai.tools.registry import ToolDefinition, ToolRegistry


class TestMCPAndToolRegistry(unittest.TestCase):
    def test_mcp_server_config(self):
        conf = MCPServerConfig(
            name="sqlite",
            command="uvx",
            args=["mcp-server-sqlite", "--db", "test.db"],
            env={"DEBUG": "1"},
        )
        d = conf.to_dict()
        self.assertEqual(d["command"], "uvx")
        self.assertEqual(d["args"], ["mcp-server-sqlite", "--db", "test.db"])

        restored = MCPServerConfig.from_dict("sqlite", d)
        self.assertEqual(restored.name, "sqlite")
        self.assertEqual(restored.command, "uvx")
        self.assertEqual(restored.env, {"DEBUG": "1"})

    def test_tool_registry_builtins(self):
        reg = ToolRegistry()
        tools = reg.list_tools()
        names = [t.name for t in tools]
        self.assertIn("convert_document", names)
        self.assertIn("web_search", names)
        self.assertIn("system_status", names)

    def test_execute_system_status_tool(self):
        reg = ToolRegistry()
        res = reg.execute_tool("system_status", {})
        self.assertIn("result", res)
        self.assertIn("device_type", res["result"])

    def test_execute_custom_tool(self):
        reg = ToolRegistry()
        reg.register(
            ToolDefinition(
                name="multiply",
                description="Multiply two numbers",
                parameters={"type": "object"},
                handler=lambda a, b: a * b,
            )
        )
        res = reg.execute_tool("multiply", {"a": 6, "b": 7})
        self.assertEqual(res.get("result"), 42)


if __name__ == "__main__":
    unittest.main()
