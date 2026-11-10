"""MCP subprocess protocol tests use a tiny local fixture process."""
import sys
import unittest

from hari_ai.tools.mcp_client import MCPClient, MCPServerConfig


SERVER_SCRIPT = r'''
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "notifications/initialized":
        continue
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fixture", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "ping", "description": "Return pong.",
                              "inputSchema": {"type": "object", "properties": {}}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "pong"}]}
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
'''


class TestMCPClientStdio(unittest.TestCase):
    def client(self, script=SERVER_SCRIPT):
        return MCPClient(MCPServerConfig('fixture', sys.executable, ['-u', '-c', script]))

    def test_protocol_handshake_discovery_and_call(self):
        client = self.client()
        try:
            self.assertTrue(client.start(timeout=2))
            tools = client.list_tools(timeout=2)
            self.assertEqual([tool.name for tool in tools], ['ping'])
            self.assertEqual(client.call_tool('ping', {}, timeout=2)['content'][0]['text'], 'pong')
        finally:
            client.stop()

    def test_no_response_obeys_timeout(self):
        client = self.client('import sys\nfor line in sys.stdin: pass\n')
        try:
            self.assertFalse(client.start(timeout=0.15))
            self.assertIsNone(client.proc)
        finally:
            client.stop()


if __name__ == '__main__':
    unittest.main()
