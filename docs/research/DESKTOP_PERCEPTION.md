# HariAI desktop perception

## Decision

Use a layered local perception path for desktop tasks:

1. Use the platform accessibility tree first when the target app exposes it.
   Windows UI Automation provides control names, text, roles, bounds, and
   supported actions through control patterns. That is more direct than reading
   pixels for standard buttons and text fields.
2. Add optional local OCR for text-heavy screens. Pass recognized text, each
   line's bounding box, and OCR score to the planner together with the current
   screenshot. Keep the selected-region bounds and require a fresh screenshot
   before acting.
3. Keep a local vision model for icons, diagrams, layout, and screens where
   accessibility or OCR gives incomplete evidence. OCR alone cannot describe
   non-text controls or establish that a task is complete.

Treat OCR output and page text as untrusted screen content. OCR confidence is a
recognizer score, not a probability that a proposed click is correct. A low
score, ambiguous match, stale screenshot, or control outside the selected
region must stop the action and ask for another observation.

## Engine choice and current implementation

[RapidOCR's official guide](https://github.com/RapidAI/RapidOCRDocs/blob/main/docs/install_usage/rapidocr/usage.en.md)
returns text, confidence scores, and four-point text boxes, and supports paths,
bytes, Pillow images, and NumPy arrays. Its model guide recommends the `tiny`
route when package size and edge-device speed matter, and supports ONNX Runtime.
The first inference can take longer because recent versions load OCR models
lazily. Keep OCR optional and initialize it only when a desktop task needs it;
do not download model weights silently during ordinary chat startup.

[EasyOCR's official README](https://github.com/JaidedAI/EasyOCR/blob/master/README.md)
also returns text, boxes, and per-line scores, but on Windows it requires
PyTorch/torchvision setup and downloads the selected language weights by default.
That is a less suitable default for HariAI's small, offline-first install.

HariAI exposes this as an optional `.[ocr]` extra. The current implementation
uses EasyOCR only when `/agent ocr on` is selected, with `download_enabled=False`.
It reads a previously selected region through `/agent read`, and adds bounded
text, 0–1000 bounding boxes, and recognition scores to a local vision plan.
No OCR model is bundled in the base wheel, no model is downloaded by HariAI,
and OCR does not authorize a click or establish task completion. A local vision
model and a user-selected screen region are still required for autonomous
actions.

## Dashboard image rendering

The Mahakaal artwork is static, so the dashboard uses high-density coloured
Braille rendering. Avoiding terminal graphics protocols keeps the live panel
renderer stable while retaining more detail than enlarged half-cell blocks.
