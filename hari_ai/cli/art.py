"""Terminal artwork renderer for HariAI.

Converts image assets into high-definition 24-bit TrueColor Unicode half-block
characters (▀) and shaded purple ASCII art (.:-=+*#%@) for rendering directly
inside Rich and Textual panels.
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

ASCII_CHARS = " .:-=+*#%@"
PURPLE_PALETTE = [
    "#201232",
    "#341852",
    "#4d1f7c",
    "#6829a8",
    "#8535d4",
    "#a34ff4",
    "#be73ff",
    "#d59eff",
    "#eac4ff",
    "#ffffff",
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
def get_mahakal_text(
    width: int = 14,
    height_lines: int = 5,
    bg_hex: str = "#0c0912",
    mode: str = "blocks",
) -> Text:
    """Render Mahakaal image as a Rich Text object with crisp high contrast.

    Args:
        width: Number of character columns.
        height_lines: Number of terminal character lines (for blocks, each line is 2 vertical pixels).
        bg_hex: Background color behind transparent / composite pixels.
        mode: 'blocks' for 24-bit half-blocks or 'ascii' for shaded character art.
    """
    if mode == "ascii":
        return get_mahakal_ascii(width=width, height=height_lines)

    pixel_height = height_lines * 2
    try:
        from PIL import Image, ImageOps, ImageEnhance
    except ImportError:
        return _fallback_text(width, height_lines)

    path = find_mahakal_path()
    if not path:
        return _fallback_text(width, height_lines)

    try:
        img = Image.open(path).convert("RGB")
        # For compact sidebar or small cards, focus on the iconic upper silhouette (moon, face, trishul)
        if height_lines <= 8:
            w_img, h_img = img.size
            img = img.crop((int(w_img * 0.05), int(h_img * 0.02), int(w_img * 0.95), int(h_img * 0.65)))

        # Enhance contrast, sharpness, and brightness so fine contours (Trishul, moon) pop vibrantly
        enh = ImageEnhance.Contrast(img).enhance(2.6)
        sharp = ImageEnhance.Sharpness(enh).enhance(2.6)
        boosted = ImageEnhance.Brightness(sharp).enhance(1.3)

        img_resized = ImageOps.contain(boosted, (width, pixel_height), method=Image.Resampling.LANCZOS)

        # Parse background hex to RGB tuple
        bg_hex_clean = bg_hex.lstrip("#")
        bg_rgb = tuple(int(bg_hex_clean[i:i + 2], 16) for i in (0, 2, 4))
        canvas = Image.new("RGB", (width, pixel_height), bg_rgb)

        x = max(0, (width - img_resized.width) // 2)
        y = max(0, (pixel_height - img_resized.height) // 2)
        canvas.paste(img_resized, (x, y))

        out = Text()
        for r in range(0, pixel_height, 2):
            for c in range(width):
                top = canvas.getpixel((c, r))
                bot = canvas.getpixel((c, min(r + 1, pixel_height - 1)))
                fg = "#{:02x}{:02x}{:02x}".format(*top)
                bg = "#{:02x}{:02x}{:02x}".format(*bot)
                out.append("▀", style=Style(color=fg, bgcolor=bg))
            if r < pixel_height - 2:
                out.append("\n")
        return out
    except Exception:
        return _fallback_text(width, height_lines)


@functools.lru_cache(maxsize=32)
def get_mahakal_ascii(width: int = 14, height: int = 5) -> Text:
    """Renders Mahakaal using shaded purple ASCII character art."""
    try:
        from PIL import Image, ImageOps, ImageEnhance
    except ImportError:
        return _fallback_text(width, height)

    path = find_mahakal_path()
    if not path:
        return _fallback_text(width, height)

    try:
        img = Image.open(path).convert("L")
        if height <= 8:
            w_img, h_img = img.size
            img = img.crop((int(w_img * 0.05), int(h_img * 0.02), int(w_img * 0.95), int(h_img * 0.65)))

        enh = ImageEnhance.Contrast(img).enhance(2.6)
        sharp = ImageEnhance.Sharpness(enh).enhance(2.6)
        bright = ImageEnhance.Brightness(sharp).enhance(1.3)

        img_resized = ImageOps.contain(bright, (width, height), method=Image.Resampling.LANCZOS)
        canvas = Image.new("L", (width, height), 0)
        x = max(0, (width - img_resized.width) // 2)
        y = max(0, (height - img_resized.height) // 2)
        canvas.paste(img_resized, (x, y))

        out = Text()
        for r in range(height):
            for c in range(width):
                b = canvas.getpixel((c, r))
                char_idx = min(len(ASCII_CHARS) - 1, b * len(ASCII_CHARS) // 256)
                color_idx = min(len(PURPLE_PALETTE) - 1, b * len(PURPLE_PALETTE) // 256)
                out.append(ASCII_CHARS[char_idx], style=Style(color=PURPLE_PALETTE[color_idx]))
            if r < height - 1:
                out.append("\n")
        return out
    except Exception:
        return _fallback_text(width, height)


def get_mahakal_braille(width: int = 16, height_lines: int = 6) -> Text:
    """Renders Mahakaal using high-density 2x4 sub-pixel Unicode Braille art.

    Braille provides four times the pixel resolution of half-block cells while
    keeping the artwork inside a compact terminal panel. Each character is
    coloured from the source image, so the moon, trishul, and silhouette retain
    their purple highlights without enlarging into blurry blocks.
    """
    pixel_w = width * 2
    pixel_h = height_lines * 4
    try:
        from PIL import Image, ImageOps, ImageEnhance
        path = find_mahakal_path()
        if not path:
            return _fallback_text(width, height_lines)

        img = Image.open(path).convert("RGB")
        w_img, h_img = img.size
        # Keep the complete portrait while removing only the empty outer edge.
        img = img.crop((int(w_img * 0.03), 0, int(w_img * 0.97), int(h_img * 0.98)))
        enhanced = ImageEnhance.Contrast(img).enhance(3.0)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.4)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.25)
        img_resized = ImageOps.contain(enhanced, (pixel_w, pixel_h), method=Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (pixel_w, pixel_h), (0, 0, 0))
        x = max(0, (pixel_w - img_resized.width) // 2)
        y = max(0, (pixel_h - img_resized.height) // 2)
        canvas.paste(img_resized, (x, y))

        out = Text()
        # Braille 2x4 dot mapping
        dot_offsets = [
            (0, 0, 1), (0, 1, 2), (0, 2, 4),
            (1, 0, 8), (1, 1, 16), (1, 2, 32),
            (0, 3, 64), (1, 3, 128)
        ]
        threshold = 85
        for r in range(0, pixel_h, 4):
            for c in range(0, pixel_w, 2):
                code = 0
                active_colours = []
                for dx, dy, bit in dot_offsets:
                    px = c + dx
                    py = min(pixel_h - 1, r + dy)
                    red, green, blue = canvas.getpixel((px, py))
                    brightness = red * 0.2126 + green * 0.7152 + blue * 0.0722
                    if brightness > threshold:
                        code |= bit
                        active_colours.append((red, green, blue))
                braille_char = chr(0x2800 + code) if code else " "
                if active_colours:
                    count = len(active_colours)
                    red = min(255, int(sum(colour[0] for colour in active_colours) / count * 1.15))
                    green = min(255, int(sum(colour[1] for colour in active_colours) / count * 1.10))
                    blue = min(255, int(sum(colour[2] for colour in active_colours) / count * 1.15))
                    out.append(braille_char, style=Style(color=f"#{red:02x}{green:02x}{blue:02x}"))
                else:
                    out.append(braille_char)
            if r < pixel_h - 4:
                out.append("\n")
        return out
    except Exception:
        return _fallback_text(width, height_lines)


def get_mahakal_trinetra() -> Text:
    """Renders the sacred Tripundra (3 ash lines) with glowing Trinetra and crescent moon."""
    t = Text()
    t.append("       🌙  ✦ MAHAKAAL ✦\n", style=Style(color="#eac4ff", bold=True))
    t.append("  ════════════════════════════\n", style=Style(color="#a34ff4", bold=True))
    t.append("  ══════════ ( 👁 ) ══════════\n", style=Style(color="#ffffff", bold=True))
    t.append("  ════════════════════════════\n", style=Style(color="#a34ff4", bold=True))
    t.append("       OM NAMAH SHIVAYA", style=Style(color="#d59eff"))
    return t


def get_mahakal_trishul() -> Text:
    """Renders the sacred Mahatrisula & Damru (Cosmic Trident & Drum) in geometric Unicode."""
    t = Text()
    t.append("          ▲\n", style=Style(color="#eac4ff", bold=True))
    t.append("        \\ █ /\n", style=Style(color="#c17aff", bold=True))
    t.append("      ───(ॐ)───\n", style=Style(color="#ffffff", bold=True))
    t.append("       /  █  \\\n", style=Style(color="#c17aff", bold=True))
    t.append("         [⋈]   \n", style=Style(color="#eac4ff", bold=True))
    t.append("          █    \n", style=Style(color="#a34ff4"))
    t.append("          █    ", style=Style(color="#a34ff4"))
    return t


def get_mahakal_mantra() -> Text:
    """Renders the sacred Mahamrityunjaya Mantra in Sanskrit typography."""
    t = Text()
    t.append("ॐ त्र्यम्बकं यजामहे सुगन्धिं पुष्टिवर्धनम् ।\n", style=Style(color="#ffffff", bold=True))
    t.append("उर्वारुकमिव बन्धनान्मृत्योर्मुक्षीय मामृतात् ॥\n\n", style=Style(color="#eac4ff", bold=True))
    t.append("✦ MAHAKAAL · THE ETERNAL CONQUEROR OF TIME ✦", style=Style(color="#c17aff", bold=True))
    return t


def _fallback_text(width: int, height_lines: int) -> Text:
    out = Text()
    for i in range(height_lines):
        out.append(" " * width)
        if i < height_lines - 1:
            out.append("\n")
    return out


def get_mahakal_panel(
    width: int = 64,
    height_lines: int = 16,
    border_style: str = "#b77bea",
    mode: str = "blocks",
) -> Panel:
    """Creates a full-width Rich Panel enclosing Mahakaal artwork in the requested representation."""
    if mode == "braille":
        text = get_mahakal_braille(width=width, height_lines=height_lines)
    elif mode == "trinetra":
        text = get_mahakal_trinetra()
    elif mode == "trishul":
        text = get_mahakal_trishul()
    elif mode == "mantra":
        text = get_mahakal_mantra()
    else:
        text = get_mahakal_text(width=width, height_lines=height_lines, bg_hex="#0c0911")

    return Panel(
        text,
        title="[bold #c17aff]✦ MAHAKAAL · DIVINE INSPIRATION ✦[/bold #c17aff]",
        border_style=border_style,
        padding=0,
        style="on #0c0911",
    )
