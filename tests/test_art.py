"""Unit tests for Mahakaal terminal graphic rendering and dashboard integration."""
import unittest
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hari_ai.cli.art import (
    find_mahakal_path,
    get_mahakal_text,
    get_mahakal_panel,
    get_mahakal_ascii,
    get_mahakal_braille,
    get_mahakal_trinetra,
    get_mahakal_trishul,
    get_mahakal_mantra,
)
from hari_ai.cli.dashboard import CreatorCard


class TestMahakalArt(unittest.TestCase):
    def test_find_mahakal_path(self):
        path = find_mahakal_path()
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())

    def test_get_mahakal_text_cluster_size(self):
        art = get_mahakal_text(width=14, height_lines=5, bg_hex="#0c0912")
        self.assertIsInstance(art, Text)
        self.assertGreater(len(art.plain), 0)

    def test_get_mahakal_ascii(self):
        art = get_mahakal_ascii(width=14, height=5)
        self.assertIsInstance(art, Text)
        self.assertGreater(len(art.plain), 0)

    def test_get_mahakal_braille(self):
        art = get_mahakal_braille(width=16, height_lines=6)
        self.assertIsInstance(art, Text)
        self.assertGreater(len(art.plain), 0)

    def test_get_mahakal_trinetra(self):
        art = get_mahakal_trinetra()
        self.assertIsInstance(art, Text)
        self.assertIn("OM NAMAH SHIVAYA", art.plain)

    def test_get_mahakal_trishul(self):
        art = get_mahakal_trishul()
        self.assertIsInstance(art, Text)
        self.assertIn("ॐ", art.plain)

    def test_get_mahakal_mantra(self):
        art = get_mahakal_mantra()
        self.assertIsInstance(art, Text)
        self.assertIn("त्र्यम्बकं", art.plain)

    def test_get_mahakal_text_creator_size(self):
        art = get_mahakal_text(width=16, height_lines=8, bg_hex="#0c0912")
        self.assertIsInstance(art, Text)
        self.assertGreater(len(art.plain), 0)

    def test_get_mahakal_panel(self):
        panel = get_mahakal_panel(width=60, height_lines=12)
        self.assertIsInstance(panel, Panel)
        self.assertIn("MAHAKAAL", str(panel.title))

    def test_creator_card_renders_with_art(self):
        card = CreatorCard()
        rendered = card.render()
        self.assertIsInstance(rendered, Table)


if __name__ == "__main__":
    unittest.main()

