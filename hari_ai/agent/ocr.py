"""Optional, local-only OCR for a user-selected desktop region.

This module deliberately has no OCR dependency in HariAI's base package.  It
uses an already-installed EasyOCR runtime with downloads disabled, so opening
HariAI never fetches a recognition model or sends pixels to a remote service.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable

import numpy as np
from PIL import Image

MAX_OCR_IMAGE_BYTES = 8_388_608
MAX_OCR_PIXELS = 16_000_000
MAX_OCR_LINES = 80
MAX_OCR_TEXT_CHARS = 8_000


class OCRUnavailable(RuntimeError):
    """The optional offline OCR runtime or its local weights are unavailable."""


@dataclass(frozen=True)
class OCRLine:
    """One recognized line, expressed in the screenshot's 0..1000 coordinate space."""

    text: str
    confidence: float
    box: tuple[int, int, int, int]


def _normalised_box(points: Any, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(points, (list, tuple)) or not points:
        return None
    coordinates: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            coordinates.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            return None
    left = max(0, min(1000, round(min(x for x, _ in coordinates) / image_width * 1000)))
    right = max(0, min(1000, round(max(x for x, _ in coordinates) / image_width * 1000)))
    top = max(0, min(1000, round(min(y for _, y in coordinates) / image_height * 1000)))
    bottom = max(0, min(1000, round(max(y for _, y in coordinates) / image_height * 1000)))
    return (left, top, right, bottom) if left < right and top < bottom else None


class LocalScreenOCR:
    """Lazily run pre-provisioned English EasyOCR entirely on this computer."""

    def __init__(self, reader_factory: Callable[..., Any] | None = None):
        self._reader_factory = reader_factory
        self._reader: Any | None = None

    def _load_reader(self) -> Any:
        if self._reader is not None:
            return self._reader
        try:
            if self._reader_factory is not None:
                reader = self._reader_factory(['en'], gpu=False, download_enabled=False, verbose=False)
            else:
                import easyocr
                reader = easyocr.Reader(['en'], gpu=False, download_enabled=False, verbose=False)
        except ImportError as exc:
            raise OCRUnavailable(
                'Optional local OCR is not installed. Install EasyOCR separately, then provision its English weights explicitly.'
            ) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise OCRUnavailable(
                'Local OCR is not ready. HariAI did not download any OCR weights; provision English EasyOCR weights and try again.'
            ) from exc
        self._reader = reader
        return reader

    def read(self, png: bytes) -> list[OCRLine]:
        """Recognize bounded text and return line confidence plus screenshot-relative boxes."""
        if not isinstance(png, bytes) or not png.startswith(b'\x89PNG\r\n\x1a\n') or len(png) > MAX_OCR_IMAGE_BYTES:
            raise ValueError('OCR requires a PNG screenshot no larger than 8 MB.')
        try:
            with Image.open(BytesIO(png)) as image:
                image.load()
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > MAX_OCR_PIXELS:
                    raise ValueError('OCR screenshot dimensions are outside the supported range.')
                image_array = np.asarray(image.convert('RGB'))
                results = self._load_reader().readtext(image_array, detail=1, paragraph=False)
        except OCRUnavailable:
            raise
        except (OSError, ValueError) as exc:
            raise ValueError('OCR could not read the selected screenshot.') from exc
        if not isinstance(results, list):
            raise OCRUnavailable('Local OCR returned an unsupported result. No text was sent to the planner.')

        lines: list[OCRLine] = []
        used = 0
        for item in results[:MAX_OCR_LINES]:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            text = item[1]
            if not isinstance(text, str):
                continue
            text = ' '.join(text.split())
            if not text:
                continue
            try:
                confidence = max(0.0, min(1.0, float(item[2])))
            except (TypeError, ValueError):
                continue
            box = _normalised_box(item[0], width, height)
            if box is None or used + len(text) > MAX_OCR_TEXT_CHARS:
                continue
            lines.append(OCRLine(text, confidence, box))
            used += len(text) + 1
        return lines


def planner_context(lines: list[OCRLine]) -> str:
    """Make a small, readable extract; it remains untrusted screen content."""
    return '\n'.join(
        f'[{line.box[0]},{line.box[1]},{line.box[2]},{line.box[3]} score={line.confidence:.2f}] {line.text}'
        for line in lines
    )[:MAX_OCR_TEXT_CHARS]
