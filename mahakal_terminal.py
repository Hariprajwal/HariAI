"""Standalone terminal viewer for Mahakal image inside a purple-bordered panel.

Uses Pillow + Rich 24-bit TrueColor terminal graphics (Unicode half-blocks).
"""

from pathlib import Path
from PIL import Image, ImageOps
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

console = Console(color_system="truecolor")

# Locate mahakal.png in current dir or assets
IMAGE_PATH = Path("mahakal.png")
if not IMAGE_PATH.is_file():
    IMAGE_PATH = Path(__file__).resolve().parent / "hari_ai" / "assets" / "mahakal.png"

WIDTH = 72
HEIGHT = 24

# Load Mahakal image
img = Image.open(IMAGE_PATH).convert("RGBA")

# Resize while preserving proportions
img = ImageOps.contain(img, (WIDTH, HEIGHT))

# Dark background
canvas = Image.new("RGBA", (WIDTH, HEIGHT), "#0c0911")
x = (WIDTH - img.width) // 2
y = (HEIGHT - img.height) // 2
canvas.alpha_composite(img, (x, y))
pixels = canvas.convert("RGB")

output = Text()

# Convert image pixels into terminal graphics
for row in range(0, HEIGHT, 2):
    for col in range(WIDTH):
        top = pixels.getpixel((col, row))
        bottom = pixels.getpixel((col, min(row + 1, HEIGHT - 1)))
        fg = "#{:02x}{:02x}{:02x}".format(*top)
        bg = "#{:02x}{:02x}{:02x}".format(*bottom)
        output.append("▀", style=Style(color=fg, bgcolor=bg))
    output.append("\n")

# Display inside the purple panel
panel = Panel(
    output,
    title="[bold #c17aff]✦ MAHAKAAL · DIVINE INSPIRATION ✦[/bold #c17aff]",
    width=78,
    border_style="#b77bea",
    padding=0,
    style="on #0c0911",
)

if __name__ == "__main__":
    console.print(panel)
