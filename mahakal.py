import sys
from pathlib import Path
from PIL import Image

# Step 1: Characters from darkest to brightest
# Dark areas get light characters, bright areas get dense characters
ASCII_CHARS = " .:-=+*#%@"

# Step 2: Purple shades from darkest to brightest (ANSI 256-color codes)
PURPLE_SHADES = [53, 54, 55, 91, 92, 93, 129, 135, 141, 147, 183, 189, 231]

# Step 3: Terminal width we want the art to fill
TERMINAL_WIDTH = 60


def main():
    # Step 4: Get the image path from the command line (default: mahakal.png)
    image_path = sys.argv[1] if len(sys.argv) > 1 else "mahakal.png"
    p = Path(image_path)
    if not p.is_file():
        p = Path(__file__).resolve().parent / "hari_ai" / "assets" / "mahakal.png"
    print(f"Loading image: {p}")

    # Step 5: Open the image and convert it to grayscale (brightness only)
    img = Image.open(p).convert("L")
    print(f"Original size: {img.width} x {img.height}")

    # Step 6: Work out the height. Terminal characters are about twice as tall
    # as they are wide, so we halve the rows to keep the proportions correct
    new_height = int((img.height / img.width) * TERMINAL_WIDTH * 0.5)
    img = img.resize((TERMINAL_WIDTH, new_height), Image.Resampling.LANCZOS)
    print(f"Resized to: {TERMINAL_WIDTH} x {new_height} characters")

    # Step 7: Print the header in bright purple
    print("\n\033[1;38;5;141m" + "MAHAKAL".center(TERMINAL_WIDTH) + "\033[0m\n")

    # Step 8: Read all pixel brightness values
    pixels = list(img.getdata())
    print("Converting pixels to purple characters...")

    # Step 9: Walk through the pixels row by row and build each line
    for row in range(new_height):
        line = ""
        for col in range(TERMINAL_WIDTH):
            brightness = pixels[row * TERMINAL_WIDTH + col]  # 0 (dark) to 255 (bright)

            # Pick the character and the purple shade for this brightness
            char_index = brightness * (len(ASCII_CHARS) - 1) // 255
            shade_index = brightness * (len(PURPLE_SHADES) - 1) // 255

            line += f"\033[38;5;{PURPLE_SHADES[shade_index]}m{ASCII_CHARS[char_index]}"

        # Reset the color at the end of each line
        print(line + "\033[0m")

    # Step 10: Finish
    print("\n\033[38;5;141m   Om Namah Shivaya   \033[0m")
    print("Done!")


if __name__ == "__main__":
    main()

