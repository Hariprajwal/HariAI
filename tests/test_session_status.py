"""Live status is observed without model loads or service requests."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hari_ai.dream.controller import DreamEngineController, SleepTrigger
from hari_ai.memory.hierarchy import EpisodicEntry, HierarchicalMemoryManager
from hari_ai.runtime.local_model import LocalModel
from hari_ai.runtime.session_status import SessionStatus
from hari_ai.runtime.settings import ChatSettings


class TestSessionStatus(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'custom.gguf'
        self.path.write_bytes(b'model-placeholder')
        self.memory = HierarchicalMemoryManager()
        self.dream = DreamEngineController(self.memory)
        self.chat = SimpleNamespace(settings=ChatSettings(),
            local_model=LocalModel(str(self.path)), memory=self.memory, history=[], engine=None)

    def provider(self):
        return SessionStatus(lambda: self.chat, self.dream)

    def test_available_file_does_not_load_model(self):
        with patch.object(self.chat.local_model, '_load', side_effect=AssertionError('must not load')):
            status = self.provider().snapshot()
        self.assertEqual(status.model_name, 'custom.gguf')
        self.assertEqual(status.model_path, str(self.path))
        self.assertEqual(status.model_bytes, 17)
        self.assertTrue(status.model_available)
        self.assertFalse(status.model_loaded)
        self.assertEqual(status.device, 'CPU')
        self.assertEqual(status.context_tokens, 2048)
        self.assertIn('first reply', status.readiness)

    def test_missing_file_is_unknown_size_and_not_ready(self):
        self.chat.local_model.path = self.path.with_name('missing.gguf')
        status = self.provider().snapshot()
        self.assertIsNone(status.model_bytes)
        self.assertFalse(status.model_available)
        self.assertEqual(status.readiness, 'Model file missing')

    def test_settings_replacement_and_loaded_metadata_are_live(self):
        provider = self.provider()
        self.chat = SimpleNamespace(settings=ChatSettings(), memory=self.memory, history=[],
            local_model=SimpleNamespace(path=self.path, threads=2, context_size=512,
                _model=SimpleNamespace(metadata={'general.name': 'A custom model'})))
        status = provider.snapshot()
        self.assertTrue(status.model_loaded)
        self.assertEqual(status.model_name, 'A custom model')
        self.assertEqual(status.context_tokens, 512)
        self.assertLessEqual(status.threads, 2)

    def test_memory_dream_turns_and_uptime_update(self):
        with patch('hari_ai.runtime.session_status.time.monotonic', side_effect=[10, 12, 75]), \
                patch('hari_ai.runtime.session_status.psutil', None):
            provider = self.provider()
            first = provider.snapshot()
            self.chat.history.extend([SimpleNamespace(role='user'), SimpleNamespace(role='assistant')])
            self.memory.working.append_thought('one')
            self.memory.episodic.record(EpisodicEntry('1', 0.0, 'hello', 'hello'))
            self.dream.run_sleep_cycle(SleepTrigger.MANUAL)
            second = provider.snapshot()
        self.assertEqual(first.uptime_seconds, 2)
        self.assertEqual(second.uptime_seconds, 65)
        self.assertEqual(first.turns, 0)
        self.assertEqual(second.turns, 1)
        self.assertEqual(second.working_items, 1)
        self.assertEqual(second.episodic_entries, 1)
        self.assertEqual(second.dream_cycles, 1)
        self.assertEqual(second.last_inspected, 1)
        self.assertEqual(second.dream_state, 'wake')

    def test_ollama_remains_unverified_without_any_network_requests(self):
        self.chat.settings = ChatSettings(backend='ollama', ollama_model='my-model')
        self.chat.ollama = Mock()
        status = self.provider().snapshot()
        self.chat.ollama.assert_not_called()
        self.assertEqual(self.chat.ollama.mock_calls, [])
        self.assertEqual(status.model_name, 'my-model')
        self.assertFalse(status.model_loaded)
        self.assertFalse(status.model_available)
        self.assertIsNone(status.model_bytes)
        self.assertIn('unverified', status.readiness)

    def test_nano_reports_actual_device_and_checkpoint_state(self):
        self.chat.settings = ChatSettings(backend='nano', nano_checkpoint=str(self.path))
        self.chat.engine = SimpleNamespace(checkpoint_loaded=True, device='cuda:0',
            model=SimpleNamespace(config=SimpleNamespace(max_seq_len=1024)))
        status = self.provider().snapshot()
        self.assertEqual(status.device, 'cuda:0')
        self.assertTrue(status.model_loaded)
        self.assertEqual(status.model_bytes, 17)
        self.assertEqual(status.context_tokens, 1024)
        self.assertIn('Research', status.readiness)

    def test_cpu_failure_does_not_hide_other_measurements(self):
        metrics = Mock()
        metrics.cpu_percent.side_effect = [0.0, OSError('not available')]
        metrics.Process.return_value.memory_info.return_value.rss = 25_000_000
        metrics.virtual_memory.return_value.available = 2_500_000_000
        with patch('hari_ai.runtime.session_status.psutil', metrics), \
                patch('hari_ai.runtime.session_status.time.monotonic', side_effect=[0, 1]):
            status = self.provider().snapshot()
        self.assertIsNone(status.cpu_percent)
        self.assertEqual(status.process_mb, 25)
        self.assertEqual(status.available_ram_gb, 2.5)

    def test_nonfinite_metrics_are_unavailable(self):
        metrics = Mock()
        metrics.cpu_percent.return_value = float('nan')
        metrics.Process.return_value.memory_info.return_value.rss = float('inf')
        metrics.virtual_memory.return_value.available = -1
        with patch('hari_ai.runtime.session_status.psutil', metrics):
            status = self.provider().snapshot()
        self.assertIsNone(status.cpu_percent)
        self.assertIsNone(status.process_mb)
        self.assertIsNone(status.available_ram_gb)

    def test_cpu_sampling_recovers_after_initial_failure(self):
        metrics = Mock()
        metrics.cpu_percent.side_effect = [OSError('unavailable'), 0.0, 42.0]
        metrics.Process.return_value.memory_info.return_value.rss = 25_000_000
        metrics.virtual_memory.return_value.available = 2_500_000_000
        with patch('hari_ai.runtime.session_status.psutil', metrics), \
                patch('hari_ai.runtime.session_status.time.monotonic', side_effect=[0, 1, 2]):
            provider = self.provider()
            self.assertIsNone(provider.snapshot().cpu_percent)
            self.assertEqual(provider.snapshot().cpu_percent, 42.0)


if __name__ == '__main__':
    unittest.main()
