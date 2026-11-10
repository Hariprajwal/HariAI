"""Model Context Protocol (MCP) Client for HariAI.

Implements JSON-RPC 2.0 stdio client protocol for discovering and invoking
tools from local and external MCP servers (compatible with Claude, Cursor,
and Antigravity MCP specifications).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "cwd": self.cwd,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> MCPServerConfig:
        return cls(
            name=name,
            command=data.get("command", ""),
            args=data.get("args", []),
            env=data.get("env", {}),
            cwd=data.get("cwd"),
            enabled=data.get("enabled", True),
        )


@dataclass
class MCPTool:
    name: str
    description: str
    parameters: Dict[str, Any]
    server_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "server": self.server_name,
        }


def get_mcp_config_path() -> Path:
    base = Path(os.environ.get("HARIAI_HOME", Path.home() / ".hari_ai"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "mcp_servers.json"


def load_mcp_config() -> Dict[str, MCPServerConfig]:
    path = get_mcp_config_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        servers_data = raw.get("mcpServers", raw)
        return {
            name: MCPServerConfig.from_dict(name, conf)
            for name, conf in servers_data.items()
            if isinstance(conf, dict) and "command" in conf
        }
    except Exception:
        return {}


def save_mcp_config(configs: Dict[str, MCPServerConfig]) -> None:
    path = get_mcp_config_path()
    data = {"mcpServers": {name: conf.to_dict() for name, conf in configs.items()}}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class MCPClient:
    """Manages an active connection to an MCP server over stdio."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.proc: Optional[subprocess.Popen] = None
        self._req_id = 0
        self._lock = threading.Lock()
        self._tools: List[MCPTool] = []
        self._server_info: Dict[str, Any] = {}

    def start(self, timeout: float = 10.0) -> bool:
        """Starts the MCP server process and completes the protocol handshake."""
        if not self.config.command:
            return False

        env = os.environ.copy()
        env.update(self.config.env)

        cmd = [self.config.command] + self.config.args
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.config.cwd or None,
                env=env,
            )
        except Exception:
            return False

        # Protocol handshake: initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "HariAI", "version": "0.2.0"},
            },
        }
        res = self._send_request(init_req, timeout=timeout)
        if not res or "result" not in res:
            self.stop()
            return False

        self._server_info = res["result"].get("serverInfo", {})

        # Notification: initialized
        self._send_notification({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return True

    def stop(self) -> None:
        """Gracefully terminates the MCP server process."""
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def list_tools(self, timeout: float = 5.0) -> List[MCPTool]:
        """Discovers tools available on the MCP server."""
        if not self.proc:
            return []
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }
        res = self._send_request(req, timeout=timeout)
        if not res or "result" not in res:
            return []

        raw_tools = res["result"].get("tools", [])
        tools: List[MCPTool] = []
        for t in raw_tools:
            name = t.get("name", "")
            if name:
                tools.append(
                    MCPTool(
                        name=name,
                        description=t.get("description", ""),
                        parameters=t.get("inputSchema", {}),
                        server_name=self.config.name,
                    )
                )
        self._tools = tools
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Calls a tool on the MCP server and returns the result."""
        if not self.proc:
            return {"isError": True, "content": [{"type": "text", "text": "MCP server is not running."}]}

        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        res = self._send_request(req, timeout=timeout)
        if not res:
            return {"isError": True, "content": [{"type": "text", "text": "MCP call timed out or failed."}]}
        if "error" in res:
            return {"isError": True, "content": [{"type": "text", "text": str(res["error"])}]}
        return res.get("result", {})

    def _next_id(self) -> int:
        with self._lock:
            self._req_id += 1
            return self._req_id

    def _send_notification(self, payload: Dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin:
            return
        try:
            line = json.dumps(payload) + "\n"
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except Exception:
            pass

    def _send_request(self, payload: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            return None
        with self._lock:
            try:
                line = json.dumps(payload) + "\n"
                self.proc.stdin.write(line)
                self.proc.stdin.flush()

                # Read response line with timeout
                start = time.monotonic()
                while time.monotonic() - start < timeout:
                    if self.proc.poll() is not None:
                        return None
                    resp_line = self.proc.stdout.readline()
                    if not resp_line:
                        time.sleep(0.02)
                        continue
                    resp_line = resp_line.strip()
                    if not resp_line:
                        continue
                    try:
                        data = json.loads(resp_line)
                        if data.get("id") == payload.get("id"):
                            return data
                    except json.JSONDecodeError:
                        continue
                return None
            except Exception:
                return None

