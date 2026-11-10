"""Unit tests for HariAI Universal Document Converter."""
import tempfile
import unittest
from pathlib import Path

from hari_ai.tools.converter import DocumentContent, DocumentConverter, convert_document


class TestDocumentConverter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_same_name_rule(self):
        src = self.dir / "hari.pdf"
        src.write_text("Hello from Hari", encoding="utf-8")

        # Convert to docx without output_path
        out = convert_document(src, "docx")
        self.assertEqual(out.name, "hari.docx")
        self.assertTrue(out.is_file())
        self.assertEqual(out.parent, src.parent)

        # Convert to txt
        out_txt = convert_document(out, "txt")
        self.assertEqual(out_txt.name, "hari.txt")
        self.assertTrue(out_txt.is_file())

    def test_markdown_to_docx_roundtrip(self):
        md = self.dir / "report.md"
        md.write_text("# Title\n\nParagraph 1.\n\nParagraph 2.\n", encoding="utf-8")

        docx = convert_document(md, "docx")
        self.assertTrue(docx.is_file())
        self.assertGreater(docx.stat().st_size, 0)

        # Read back
        conv = DocumentConverter()
        content = conv.read_docx(docx)
        self.assertIn("Paragraph 1.", content.paragraphs)

    def test_docx_to_pdf_conversion(self):
        docx = self.dir / "doc.docx"
        conv = DocumentConverter()
        doc_content = DocumentContent(
            title="System Architecture",
            paragraphs=["Paragraph explaining the system.", "Another paragraph."],
        )
        conv.write_docx(doc_content, docx)
        self.assertTrue(docx.is_file())

        pdf = convert_document(docx, "pdf")
        self.assertEqual(pdf.name, "doc.pdf")
        self.assertTrue(pdf.is_file())
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF-1.4"))

    def test_csv_and_json_conversions(self):
        csv_file = self.dir / "data.csv"
        csv_file.write_text("Name,Role,Status\nHari,Creator,Active\nBot,Swarm,Running\n", encoding="utf-8")

        # CSV to HTML
        html_file = convert_document(csv_file, "html")
        self.assertEqual(html_file.name, "data.html")
        self.assertIn("<table>", html_file.read_text(encoding="utf-8"))

        # CSV to JSON
        json_file = convert_document(csv_file, "json")
        self.assertEqual(json_file.name, "data.json")
        self.assertIn("Hari", json_file.read_text(encoding="utf-8"))

    def test_custom_output_destination(self):
        src = self.dir / "input.txt"
        src.write_text("Content to convert", encoding="utf-8")
        custom_out = self.dir / "subfolder" / "custom_named.md"

        res = convert_document(src, output_path=custom_out)
        self.assertEqual(res, custom_out)
        self.assertTrue(custom_out.is_file())


if __name__ == "__main__":
    unittest.main()
