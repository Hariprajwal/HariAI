import base64
import unittest

from hari_ai.agent.ocr import LocalScreenOCR, OCRUnavailable, planner_context


PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC'
)


class FakeReader:
    def __init__(self):
        self.calls = []

    def readtext(self, image, *, detail, paragraph):
        self.calls.append((image.shape, detail, paragraph))
        return [
            ([[0, 0], [1, 0], [1, 1], [0, 1]], '  HariAI  ', 0.996),
            ([[0, 0], [0, 0]], 'invalid', 0.9),
        ]


class TestLocalScreenOCR(unittest.TestCase):
    def test_recognized_text_is_bounded_normalized_and_marked_as_untrusted_context(self):
        reader = FakeReader()
        ocr = LocalScreenOCR(reader_factory=lambda *_, **__: reader)
        lines = ocr.read(PNG)
        self.assertEqual(reader.calls, [((1, 1, 3), 1, False)])
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, 'HariAI')
        self.assertEqual(lines[0].box, (0, 0, 1000, 1000))
        self.assertAlmostEqual(lines[0].confidence, .996)
        self.assertEqual(planner_context(lines), '[0,0,1000,1000 score=1.00] HariAI')

    def test_missing_runtime_is_a_clear_non_downloading_failure(self):
        def unavailable(*_, **__):
            raise ImportError('not installed')

        with self.assertRaisesRegex(OCRUnavailable, 'not installed'):
            LocalScreenOCR(reader_factory=unavailable).read(PNG)

    def test_rejects_non_png_before_trying_to_load_optional_ocr(self):
        with self.assertRaises(ValueError):
            LocalScreenOCR(reader_factory=lambda *_, **__: self.fail('reader must not load')).read(b'not a png')


if __name__ == '__main__':
    unittest.main()
