"""Explicit local memory is deduplicated and deletable."""
import tempfile
import unittest
from pathlib import Path
from hari_ai.memory.store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_opt_in_persistence_and_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'memory.sqlite3'
            store = MemoryStore(path)
            self.assertEqual(store.list(), [])
            self.assertFalse(path.exists())
            entry_id = store.add('Keep this preference')
            self.assertEqual(store.add('Keep this preference'), entry_id)
            self.assertEqual(MemoryStore(path).list()[0]['content'], 'Keep this preference')
            self.assertTrue(store.delete(entry_id))
            self.assertFalse(store.delete(entry_id))
            self.assertEqual(store.list(), [])

    def test_rejects_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MemoryStore(Path(tmp) / 'memory.db').add('  ')


if __name__ == '__main__':
    unittest.main()
