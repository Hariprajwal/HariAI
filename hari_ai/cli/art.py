"""Terminal artwork renderer for HariAI.

Converts image assets into 24-bit TrueColor Unicode half-block characters (▀)
for rendering directly inside Rich and Textual panels.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.style import Style
from rich.text import Text

_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
_MAHAKAL_PATHS = [
    _ASSET_DIR / "mahakal.png",
    Path("mahakal.png"),
    Path(__file__).resolve().parents[2] / "mahakal.png",
]


def find_mahakal_path() -> Optional[Path]:
    """Locate mahakal.png in package assets or project directory."""
    for p in _MAHAKAL_PATHS:
        try:
            if p.is_file():
                return p
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=32)
def get_mahakal_text(width: int = 10, height_lines: int = 4, bg_hex: str = "#0c0912") -> Text:
    """Render Mahakaal image as a Rich Text object using 24-bit half-blocks.

    Args:
        width: Number of character columns.
        height_lines: Number of terminal character lines (each line represents 2 vertical pixels).
        bg_hex: Background color behind transparent / composite pixels.
    """
    pixel_height = height_lines * 2
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return _fallback_text(width, height_lines)

    path = find_mahakal_path()
    if not path:
        return _fallback_text(width, height_lines)

    try:
        img = Image.open(path).convert("RGBA")
        img_resized = ImageOps.contain(img, (width, pixel_height))

        # Parse background hex to RGB tuple
        bg_hex_clean = bg_hex.lstrip("#")
        bg_rgb = tuple(int(bg_hex_clean[i:i + 2], 16) for i in (0, 2, 4))
        canvas = Image.new("RGBA", (width, pixel_height), (*bg_rgb, 255))

        x = max(0, (width - img_resized.width) // 2)
        y = max(0, (pixel_height - img_resized.height) // 2)
        canvas.alpha_composite(img_resized, (x, y))
        rgb = canvas.convert("RGB")

        out = Text()
        for r in range(0, pixel_height, 2):
            for c in range(width):
                top = rgb.getpixel((c, r))
                bot = rgb.getpixel((c, min(r + 1, pixel_height - 1)))
                fg = "#{:02x}{:02x}{:02x}".format(*top)
                bg = "#{:02x}{:02x}{:02x}".format(*bot)
                out.append("▀", style=Style(color=fg, bgcolor=bg))
            if r < pixel_height - 2:
                out.append("\n")
        return out
    except Exception:
        return _fallback_text(width, height_lines)


def _fallback_text(width: int, height_lines: int) -> Text:
    out = Text()
    for i in range(height_lines):
        out.append(" " * width)
        if i < height_lines - 1:
            out.append("\n")
    return out


def get_mahakal_panel(width: int = 64, height_lines: int = 14, border_style: str = "#b77bea") -> Panel:
    """Creates a full-width Rich Panel enclosing Mahakaal graphic."""
    text = get_mahakal_text(width=width, height_lines=height_lines, bg_hex="#0c0911")
    return Panel(
        text,
        title="[bold #c17aff]✦ MAHAKAAL · DIVINE INSPIRATION ✦[/bold #c17aff]",
        border_style=border_style,
        padding=0,
        style="on #0c0911",
    )
