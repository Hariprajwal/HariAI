"""Capture the real dashboard as SVG, optionally after real model replies.

Example: python scripts/preview_dashboard.py --prompt hello --prompt "who made you?"
No conversation, model state, or telemetry is fabricated for the preview.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import importlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def capture(args) -> None:
    # Preview the colour theme even when a CI shell exports NO_COLOR.
    os.environ.pop('NO_COLOR', None)
    from hari_ai.cli.dashboard import HariDashboard
    cli = importlib.import_module('hari_ai.cli.main')
    app = HariDashboard(lambda: cli._CHAT_ENGINE, cli._DREAM_CONTROLLER, cli.dashboard_command)
    async with app.run_test(size=(args.width, args.height)) as pilot:
        await pilot.pause()
        for prompt in args.prompt:
            await app.submit_prompt(prompt)

            async def finished():
                while app.busy:
                    await pilot.pause(0.05)

            await asyncio.wait_for(finished(), timeout=120)
            await pilot.pause()
            if app.last_error:
                raise RuntimeError(app.last_error)
        app.refresh_panels()
        await pilot.pause()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(app.export_screenshot(), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=166)
    parser.add_argument('--height', type=int, default=50)
    parser.add_argument('--prompt', action='append', default=[])
    parser.add_argument('--output', type=Path, default=Path('checkpoints/ui-preview/dashboard.svg'))
    args = parser.parse_args()
    if args.width < 40 or args.height < 15:
        parser.error('Preview needs at least 40 columns and 15 rows.')
    # Headless previews have no terminal driver to capture native load logs.
    # Python stream capture is safe here and never replaces OS console handles.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        asyncio.run(capture(args))
    print(f'Captured {args.width}x{args.height} terminal: {args.output.resolve()}')


if __name__ == '__main__':
    main()
