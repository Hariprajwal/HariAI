"""Tool planner tests use a fake local model and no configured MCP servers."""
import json
import unittest
from types import SimpleNamespace

from hari_ai.tools.agent import LocalToolAgent, validate_schema
from hari_ai.tools.registry import ToolDefinition, ToolRegistry


class FakeLocalModel:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.last_finish_reason = 'stop'
        self.calls = []

    def chat_stream(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        yield next(self.answers)


class TestToolAgent(unittest.TestCase):
    def test_local_planner_executes_only_available_schema_valid_tool(self):
        local = FakeLocalModel([
            json.dumps({'tool': 'multiply', 'arguments': {'left': 6, 'right': 7}, 'final': ''}),
            json.dumps({'tool': None, 'arguments': {}, 'final': 'The tool returned 42.'}),
        ])
        engine = SimpleNamespace(settings=SimpleNamespace(backend='local'), local_model=local)
        registry = ToolRegistry()
        registry.connect_mcp_servers = lambda: {}
        registry.register(ToolDefinition(
            name='multiply', description='Multiply two numbers.',
            parameters={'type': 'object', 'required': ['left', 'right'],
                        'additionalProperties': False,
                        'properties': {'left': {'type': 'integer'}, 'right': {'type': 'integer'}}},
            handler=lambda left, right: left * right,
        ))

        result = LocalToolAgent(engine, registry).run('Multiply 6 by 7')
        self.assertEqual(result, 'The tool returned 42.')
        self.assertEqual(len(local.calls), 2)
        self.assertIn('"result": 42', local.calls[1][0][1]['content'])

    def test_bad_arguments_stop_before_tool_execution(self):
        local = FakeLocalModel([json.dumps({'tool': 'multiply', 'arguments': {'left': 'six'}, 'final': ''})])
        engine = SimpleNamespace(settings=SimpleNamespace(backend='local'), local_model=local)
        registry = ToolRegistry()
        registry.connect_mcp_servers = lambda: {}
        registry.register(ToolDefinition(
            name='multiply', description='Multiply.',
            parameters={'type': 'object', 'required': ['left'],
                        'properties': {'left': {'type': 'integer'}}},
            handler=lambda left: self.fail('Invalid schema arguments were executed.'),
        ))
        with self.assertRaisesRegex(ValueError, 'wrong JSON type'):
            LocalToolAgent(engine, registry).run('Multiply')

    def test_nonlocal_planner_never_sends_tool_context_to_remote_backend(self):
        engine = SimpleNamespace(settings=SimpleNamespace(backend='ollama'), local_model=None)
        registry = ToolRegistry()
        registry.connect_mcp_servers = lambda: {}
        agent = LocalToolAgent(engine, registry)
        with self.assertRaisesRegex(RuntimeError, 'local HariAI GGUF'):
            agent.run('Look up the configured information')

    def test_schema_validation_rejects_nonfinite_and_unvalidated_patterns(self):
        with self.assertRaisesRegex(ValueError, 'wrong JSON type'):
            validate_schema(float('nan'), {'type': 'number'})
        with self.assertRaisesRegex(ValueError, 'cannot validate safely'):
            validate_schema('value', {'type': 'string', 'pattern': '.*'})


if __name__ == '__main__':
    unittest.main()
