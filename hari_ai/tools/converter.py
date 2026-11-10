"""Universal Document Converter for HariAI.

Converts documents between PDF, Word (.docx), Markdown (.md), Text (.txt),
HTML (.html), JSON (.json), CSV (.csv), and RTF (.rtf).

Guarantees 100% offline operation using built-in pure-Python OpenXML, PDF,
and text engines with zero heavy dependencies (< 50 KB added to package).
"""
from __future__ import annotations

import csv
import html
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import zipfile
import zlib


class DocumentContent:
    """Intermediate structured representation of a document."""

    def __init__(
        self,
        title: str = "",
        paragraphs: Optional[List[str]] = None,
        headings: Optional[List[Tuple[int, str]]] = None,
        tables: Optional[List[List[List[str]]]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        self.title = title
        self.paragraphs: List[str] = paragraphs or []
        self.headings: List[Tuple[int, str]] = headings or []
        self.tables: List[List[List[str]]] = tables or []
        self.metadata: Dict[str, str] = metadata or {}

    def get_full_text(self) -> str:
        parts: List[str] = []
        if self.title:
            parts.append(self.title)
        for p in self.paragraphs:
            clean = p.strip()
            if clean:
                parts.append(clean)
        for table in self.tables:
            for row in table:
                parts.append(" | ".join(row))
        return "\n\n".join(parts)


class _HTMLToDocumentParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.paragraphs: List[str] = []
        self.headings: List[Tuple[int, str]] = []
        self.tables: List[List[List[str]]] = []
        self._current_tag = ""
        self._text_buf: List[str] = []
        self._in_title = False
        self._current_table: Optional[List[List[str]]] = None
        self._current_row: Optional[List[str]] = None

    def handle_starttag(self, tag: str, attrs: Any):
        self._current_tag = tag.lower()
        if self._current_tag == "title":
            self._in_title = True
        elif self._current_tag == "table":
            self._current_table = []
        elif self._current_tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif self._current_tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self._flush_text()

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t == "title":
            self._in_title = False
            self.title = "".join(self._text_buf).strip()
            self._text_buf.clear()
        elif t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            lvl = int(t[1])
            heading_text = "".join(self._text_buf).strip()
            if heading_text:
                self.headings.append((lvl, heading_text))
                self.paragraphs.append(heading_text)
            self._text_buf.clear()
        elif t in ("p", "div", "li"):
            self._flush_text()
        elif t in ("td", "th") and self._current_row is not None:
            cell = "".join(self._text_buf).strip()
            self._current_row.append(cell)
            self._text_buf.clear()
        elif t == "tr" and self._current_table is not None and self._current_row is not None:
            if self._current_row:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif t == "table" and self._current_table is not None:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = None

    def handle_data(self, data: str):
        if data.strip():
            self._text_buf.append(data)
        elif self._text_buf:
            self._text_buf.append(" ")

    def _flush_text(self):
        txt = "".join(self._text_buf).strip()
        if txt:
            self.paragraphs.append(txt)
        self._text_buf.clear()


class DocumentConverter:
    """Universal document converter with built-in pure-Python handlers."""

    SUPPORTED_FORMATS = {"pdf", "docx", "txt", "md", "html", "htm", "json", "csv", "tsv", "rtf"}

    @classmethod
    def detect_format(cls, path_or_ext: str | Path) -> str:
        s = str(path_or_ext).lower().strip()
        if "." in s:
            s = s.rsplit(".", 1)[-1]
        s = s.lstrip(".")
        if s in ("htm", "html"):
            return "html"
        if s in ("markdown", "md"):
            return "md"
        if s in ("text", "txt"):
            return "txt"
        if s in cls.SUPPORTED_FORMATS:
            return s
        return "txt"

    # -------------------------------------------------------------------------
    # Parsers
    # -------------------------------------------------------------------------

    def read(self, path: Path) -> DocumentContent:
        fmt = self.detect_format(path)
        if fmt == "docx":
            return self.read_docx(path)
        elif fmt == "pdf":
            return self.read_pdf(path)
        elif fmt == "html":
            return self.read_html(path)
        elif fmt == "md":
            return self.read_md(path)
        elif fmt == "csv" or fmt == "tsv":
            return self.read_csv(path, delimiter="\t" if fmt == "tsv" else ",")
        elif fmt == "json":
            return self.read_json(path)
        elif fmt == "rtf":
            return self.read_rtf(path)
        else:
            return self.read_txt(path)

    def read_txt(self, path: Path) -> DocumentContent:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines()]
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        title = lines[0] if lines else path.stem
        return DocumentContent(title=title, paragraphs=paragraphs or lines)

    def read_md(self, path: Path) -> DocumentContent:
        text = path.read_text(encoding="utf-8", errors="replace")
        paragraphs: List[str] = []
        headings: List[Tuple[int, str]] = []
        tables: List[List[List[str]]] = []

        current_table: List[List[str]] = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#"):
                m = re.match(r"^(#+)\s*(.*)", s)
                if m:
                    lvl = len(m.group(1))
                    h_text = m.group(2).strip()
                    headings.append((lvl, h_text))
                    paragraphs.append(h_text)
                    continue
            if s.startswith("|") and s.endswith("|"):
                cells = [c.strip() for c in s[1:-1].split("|")]
                if all(re.match(r"^:?-+:?$", c) for c in cells):
                    continue  # Table divider
                current_table.append(cells)
                continue
            else:
                if current_table:
                    tables.append(current_table)
                    current_table = []
                if s:
                    paragraphs.append(s)
        if current_table:
            tables.append(current_table)

        title = headings[0][1] if headings else path.stem
        return DocumentContent(title=title, paragraphs=paragraphs, headings=headings, tables=tables)

    def read_html(self, path: Path) -> DocumentContent:
        raw = path.read_text(encoding="utf-8", errors="replace")
        parser = _HTMLToDocumentParser()
        parser.feed(raw)
        title = parser.title or path.stem
        return DocumentContent(
            title=title,
            paragraphs=parser.paragraphs,
            headings=parser.headings,
            tables=parser.tables,
        )

    def read_csv(self, path: Path, delimiter: str = ",") -> DocumentContent:
        raw = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
        rows = [row for row in reader if row]
        paragraphs = [", ".join(row) for row in rows]
        return DocumentContent(title=path.stem, paragraphs=paragraphs, tables=[rows] if rows else [])

    def read_json(self, path: Path) -> DocumentContent:
        raw = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
        paragraphs: List[str] = []
        tables: List[List[List[str]]] = []

        if isinstance(data, list) and data and isinstance(data[0], dict):
            headers = list(data[0].keys())
            rows = [headers]
            for item in data:
                rows.append([str(item.get(h, "")) for h in headers])
            tables.append(rows)
            paragraphs = [" | ".join(r) for r in rows]
        elif isinstance(data, dict):
            for k, v in data.items():
                paragraphs.append(f"{k}: {v}")
        else:
            paragraphs.append(str(data))

        return DocumentContent(title=path.stem, paragraphs=paragraphs, tables=tables)

    def read_docx(self, path: Path) -> DocumentContent:
        """Extracts text, headings, and tables from Word .docx without external dependencies."""
        from xml.etree import ElementTree as ET

        paragraphs: List[str] = []
        headings: List[Tuple[int, str]] = []
        tables: List[List[List[str]]] = []

        with zipfile.ZipFile(path, "r") as z:
            xml_data = z.read("word/document.xml")
            tree = ET.fromstring(xml_data)

            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

            for elem in tree.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag == "p":
                    texts = [t.text for t in elem.iter(f"{{{ns['w']}}}t") if t.text]
                    p_text = "".join(texts).strip()
                    if p_text:
                        p_style = elem.find(".//w:pStyle", ns)
                        if p_style is not None:
                            val = p_style.attrib.get(f"{{{ns['w']}}}val", "").lower()
                            if "heading" in val or "title" in val:
                                lvl = 1
                                m = re.search(r"\d+", val)
                                if m:
                                    lvl = int(m.group(0))
                                headings.append((lvl, p_text))
                        paragraphs.append(p_text)
                elif tag == "tbl":
                    tbl_rows: List[List[str]] = []
                    for row in elem.iter(f"{{{ns['w']}}}tr"):
                        row_cells: List[str] = []
                        for cell in row.iter(f"{{{ns['w']}}}tc"):
                            cell_texts = [t.text for t in cell.iter(f"{{{ns['w']}}}t") if t.text]
                            row_cells.append("".join(cell_texts).strip())
                        if row_cells:
                            tbl_rows.append(row_cells)
                    if tbl_rows:
                        tables.append(tbl_rows)

        title = headings[0][1] if headings else path.stem
        return DocumentContent(title=title, paragraphs=paragraphs, headings=headings, tables=tables)

    def read_pdf(self, path: Path) -> DocumentContent:
        """Extracts text from PDF files using pure-Python stream decoder or pypdf if available."""
        # 1. Try pypdf if installed
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages_text = [p.extract_text() or "" for p in reader.pages]
            full_text = "\n\n".join(pages_text)
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            return DocumentContent(title=path.stem, paragraphs=paragraphs or [full_text])
        except Exception:
            pass

        # 2. Native pure-Python PDF stream decoder
        data = path.read_bytes()
        extracted: List[str] = []
        for match in re.finditer(rb"stream[\r\n]+(.*?)[\r\n]+endstream", data, re.DOTALL):
            stream_bytes = match.group(1)
            try:
                stream_bytes = zlib.decompress(stream_bytes)
            except Exception:
                pass

            for s in re.findall(rb"\(((?:[^()\\]|\\.)*)\)", stream_bytes):
                clean = s.replace(b"\\\\", b"\\").replace(b"\\(", b"(").replace(b"\\)", b")")
                txt = clean.decode("latin1", errors="replace").strip()
                if txt and len(txt) > 1:
                    extracted.append(txt)

        paragraphs = [p for p in extracted if p]
        return DocumentContent(title=path.stem, paragraphs=paragraphs or [path.stem])

    def read_rtf(self, path: Path) -> DocumentContent:
        raw = path.read_text(encoding="latin1", errors="replace")
        # Remove RTF control words
        text = re.sub(r"\\[a-zA-Z0-9]+(-?\d+)?\s?", " ", raw)
        text = re.sub(r"[{}]", " ", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return DocumentContent(title=path.stem, paragraphs=lines)

    # -------------------------------------------------------------------------
    # Writers / Exporters
    # -------------------------------------------------------------------------

    def write(self, content: DocumentContent, target_path: Path) -> Path:
        fmt = self.detect_format(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "docx":
            return self.write_docx(content, target_path)
        elif fmt == "pdf":
            return self.write_pdf(content, target_path)
        elif fmt == "html":
            return self.write_html(content, target_path)
        elif fmt == "md":
            return self.write_md(content, target_path)
        elif fmt == "csv" or fmt == "tsv":
            return self.write_csv(content, target_path, delimiter="\t" if fmt == "tsv" else ",")
        elif fmt == "json":
            return self.write_json(content, target_path)
        else:
            return self.write_txt(content, target_path)

    def write_txt(self, content: DocumentContent, target_path: Path) -> Path:
        target_path.write_text(content.get_full_text(), encoding="utf-8")
        return target_path

    def write_md(self, content: DocumentContent, target_path: Path) -> Path:
        out: List[str] = []
        if content.title:
            out.append(f"# {content.title}\n")
        for p in content.paragraphs:
            out.append(p + "\n")
        for table in content.tables:
            if not table:
                continue
            headers = table[0]
            out.append("| " + " | ".join(headers) + " |")
            out.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in table[1:]:
                padded = row + [""] * (len(headers) - len(row))
                out.append("| " + " | ".join(padded[:len(headers)]) + " |")
            out.append("")
        target_path.write_text("\n".join(out), encoding="utf-8")
        return target_path

    def write_html(self, content: DocumentContent, target_path: Path) -> Path:
        out: List[str] = [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"UTF-8\">",
            f"<title>{html.escape(content.title or 'Document')}</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; "
            "max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #222; }",
            "h1 { color: #5a2e9d; border-bottom: 2px solid #e1d3f5; padding-bottom: 8px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }",
            "th { background: #f4ecfb; color: #4a2185; }",
            "</style>",
            "</head>",
            "<body>",
        ]
        if content.title:
            out.append(f"<h1>{html.escape(content.title)}</h1>")
        for p in content.paragraphs:
            out.append(f"<p>{html.escape(p)}</p>")
        for table in content.tables:
            if not table:
                continue
            out.append("<table>")
            out.append("<thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in table[0]) + "</tr></thead>")
            out.append("<tbody>")
            for row in table[1:]:
                out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
        out.append("</body></html>")
        target_path.write_text("\n".join(out), encoding="utf-8")
        return target_path

    def write_csv(self, content: DocumentContent, target_path: Path, delimiter: str = ",") -> Path:
        with target_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter)
            if content.tables:
                for row in content.tables[0]:
                    writer.writerow(row)
            else:
                for p in content.paragraphs:
                    writer.writerow([p])
        return target_path

    def write_json(self, content: DocumentContent, target_path: Path) -> Path:
        data = {
            "title": content.title,
            "paragraphs": content.paragraphs,
            "headings": [{"level": lvl, "text": txt} for lvl, txt in content.headings],
            "tables": content.tables,
            "metadata": content.metadata,
        }
        target_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return target_path

    def write_docx(self, content: DocumentContent, target_path: Path) -> Path:
        """Pure-Python OpenXML Word .docx generator."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            # [Content_Types].xml
            z.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                '</Types>',
            )
            # _rels/.rels
            z.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                '</Relationships>',
            )
            # word/document.xml
            p_xml: List[str] = []
            if content.title:
                esc_title = html.escape(content.title)
                p_xml.append(
                    f'<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="36"/></w:rPr><w:t>{esc_title}</w:t></w:r></w:p>'
                )

            for p in content.paragraphs:
                esc_p = html.escape(p)
                p_xml.append(f'<w:p><w:r><w:t>{esc_p}</w:t></w:r></w:p>')

            for table in content.tables:
                if not table:
                    continue
                tbl_xml = ["<w:tbl><w:tblPr><w:tblBorders><w:top w:val=\"single\" w:sz=\"4\"/><w:bottom w:val=\"single\" w:sz=\"4\"/><w:left w:val=\"none\"/><w:right w:val=\"none\"/><w:insideH w:val=\"single\" w:sz=\"4\"/><w:insideV w:val=\"none\"/></w:tblBorders></w:tblPr>"]
                for row_idx, row in enumerate(table):
                    tbl_xml.append("<w:tr>")
                    for cell in row:
                        esc_c = html.escape(cell)
                        b_tag = "<w:b/>" if row_idx == 0 else ""
                        tbl_xml.append(
                            f'<w:tc><w:p><w:r><w:rPr>{b_tag}</w:rPr><w:t>{esc_c}</w:t></w:r></w:p></w:tc>'
                        )
                    tbl_xml.append("</w:tr>")
                tbl_xml.append("</w:tbl>")
                p_xml.append("".join(tbl_xml))

            body_content = "".join(p_xml)
            doc_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f'<w:body>{body_content}</w:body>'
                '</w:document>'
            )
            z.writestr("word/document.xml", doc_xml)

        target_path.write_bytes(buf.getvalue())
        return target_path

    def write_pdf(self, content: DocumentContent, target_path: Path) -> Path:
        """Pure-Python PDF 1.4 document generator with text wrapping & pagination."""
        def esc_pdf(s: str) -> str:
            return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        # Prepare formatted lines (wrap at ~80 characters)
        raw_lines: List[str] = []
        if content.title:
            raw_lines.append(f"=== {content.title} ===")
            raw_lines.append("")

        for p in content.paragraphs:
            # Simple text wrap
            words = p.split()
            current_line: List[str] = []
            for w in words:
                if sum(len(x) + 1 for x in current_line) + len(w) > 78:
                    raw_lines.append(" ".join(current_line))
                    current_line = [w]
                else:
                    current_line.append(w)
            if current_line:
                raw_lines.append(" ".join(current_line))
            raw_lines.append("")

        for table in content.tables:
            for row in table:
                raw_lines.append(" | ".join(row))
            raw_lines.append("")

        lines_per_page = 46
        pages: List[List[str]] = []
        for i in range(0, max(1, len(raw_lines)), lines_per_page):
            chunk = raw_lines[i : i + lines_per_page]
            if chunk:
                pages.append(chunk)

        if not pages:
            pages = [[content.title or "Document"]]

        # Build PDF objects
        objects: List[bytes] = []
        total_pages = len(pages)
        # Obj 1: Catalog
        objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        # Obj 2: Pages container
        kids = " ".join(f"{3 + i*2} 0 R" for i in range(total_pages))
        objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {total_pages} >>\nendobj\n".encode("latin1"))

        # Font object: Obj ID is 3 + total_pages*2
        font_obj_id = 3 + total_pages * 2

        for page_idx, page_lines in enumerate(pages):
            page_obj_id = 3 + page_idx * 2
            content_obj_id = page_obj_id + 1

            # Page Object
            objects.append(
                f"{page_obj_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_obj_id} 0 R /Resources << /Font << /F1 {font_obj_id} 0 R >> >> >>\nendobj\n".encode("latin1")
            )

            # Content stream
            stream_cmds = ["BT", "/F1 10 Tf", "14 TL", "50 740 Td"]
            for line in page_lines:
                clean = esc_pdf(line)
                stream_cmds.append(f"({clean}) '")
            stream_cmds.append("ET")
            stream_bytes = "\n".join(stream_cmds).encode("latin1", errors="replace")

            objects.append(
                f"{content_obj_id} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin1")
                + stream_bytes
                + b"\nendstream\nendobj\n"
            )

        # Font Object
        objects.append(
            f"{font_obj_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("latin1")
        )

        out: List[bytes] = [b"%PDF-1.4\n"]
        offsets: List[int] = [0]
        curr = len(out[0])
        for obj in objects:
            offsets.append(curr)
            out.append(obj)
            curr += len(obj)

        xref_offset = curr
        out.append(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin1"))
        for off in offsets[1:]:
            out.append(f"{off:010d} 00000 n \n".encode("latin1"))
        out.append(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1")
        )

        target_path.write_bytes(b"".join(out))
        return target_path

    # -------------------------------------------------------------------------
    # Core Conversion Interface
    # -------------------------------------------------------------------------

    def convert(
        self,
        source: str | Path,
        target_format: Optional[str] = None,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Converts any document to any document.

        If output_path is not specified, uses the exact same file name in the
        same directory with the target extension (e.g., hari.pdf -> hari.docx).
        """
        src = Path(source).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found: {src}")

        if output_path is not None:
            dest = Path(output_path).expanduser().resolve()
        elif target_format is not None:
            clean_fmt = target_format.lower().lstrip(".")
            # EXACT SAME NAME with target extension!
            dest = src.with_suffix("." + clean_fmt)
        else:
            raise ValueError("Either target_format (e.g. 'docx', 'pdf', 'txt') or output_path must be provided.")

        # Read source content
        content = self.read(src)

        # Write to destination format
        return self.write(content, dest)


def convert_document(
    source: str | Path,
    target_format: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Helper entry point for document conversion."""
    return DocumentConverter().convert(source, target_format, output_path)
