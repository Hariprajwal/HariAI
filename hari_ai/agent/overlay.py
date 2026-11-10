"""Floating Take Over Pill Window & Screen Glow Overlay for HariAI.

Renders an on-screen floating control pill:
  '● HariAI has control   [ Pause ]  [ Take over ]'
accompanied by a glowing purple indicator border along the screen edge.

Clicking [ Take over ] or pressing <Esc> instantly surrenders all mouse and keyboard
controls back to the user.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Optional


class TakeoverOverlay:
    """Floating pill window with Take Over & Pause buttons and purple edge glow."""

    def __init__(
        self,
        stop_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
        on_takeover: Optional[Callable[[], None]] = None,
    ):
        self.stop_event = stop_event or threading.Event()
        self.pause_event = pause_event or threading.Event()
        self.on_takeover = on_takeover
        self._root = None
        self._thread: Optional[threading.Thread] = None
        self._is_paused = False
        self._is_active = False

    def start(self) -> None:
        """Starts the floating overlay in a dedicated thread."""
        try:
            self._thread = threading.Thread(target=self._run_tk, daemon=True)
            self._thread.start()
        except Exception:
            self._is_active = False

    def _run_tk(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            self._root = root
            root.title("HariAI Autonomous Control")
            root.overrideredirect(True)
            root.wm_attributes("-topmost", True)

            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()

            # Dimensions for pill containing title + two buttons: [ Pause ] [ Take over ]
            pill_w = 360
            pill_h = 46
            x = max(0, (sw - pill_w) // 2)
            y = 10
            root.geometry(f"{pill_w}x{pill_h}+{x}+{y}")
            root.configure(bg="#0c0912")

            # Purple glow border frame
            frame = tk.Frame(
                root,
                bg="#0c0912",
                highlightbackground="#b77bea",
                highlightthickness=2,
            )
            frame.pack(fill="both", expand=True, padx=1, pady=1)

            # Glowing indicator dot
            dot = tk.Label(frame, text="●", fg="#c17aff", bg="#0c0912", font=("Helvetica", 12, "bold"))
            dot.pack(side="left", padx=(10, 2))

            # Status label
            lbl_text = tk.StringVar(value="HariAI has control")
            lbl = tk.Label(frame, textvariable=lbl_text, fg="#f0e6ff", bg="#0c0912", font=("Segoe UI", 10, "bold"))
            lbl.pack(side="left", padx=(2, 8))

            # Action: Take over
            def _trigger_takeover(event=None):
                self.stop_event.set()
                if self.on_takeover:
                    try:
                        self.on_takeover()
                    except Exception:
                        pass
                self.close()

            # Action: Toggle Pause / Resume
            def _toggle_pause():
                self._is_paused = not self._is_paused
                if self._is_paused:
                    self.pause_event.set()
                    btn_pause.configure(text="Resume", bg="#3d1f6d", fg="#e1d3f5")
                    lbl_text.set("HariAI paused")
                    dot.configure(fg="#f3c762")
                else:
                    self.pause_event.clear()
                    btn_pause.configure(text="Pause", bg="#201232", fg="#c17aff")
                    lbl_text.set("HariAI has control")
                    dot.configure(fg="#c17aff")

            # Button 1: Take over (Primary prominent white pill)
            btn_takeover = tk.Button(
                frame,
                text="Take over",
                bg="#ffffff",
                fg="#0c0912",
                activebackground="#eac4ff",
                activeforeground="#000000",
                font=("Segoe UI", 9, "bold"),
                relief="flat",
                padx=10,
                pady=3,
                cursor="hand2",
                command=_trigger_takeover,
            )
            btn_takeover.pack(side="right", padx=(4, 10), pady=6)

            # Button 2: Pause / Resume (Secondary purple pill)
            btn_pause = tk.Button(
                frame,
                text="Pause",
                bg="#201232",
                fg="#c17aff",
                activebackground="#3d1f6d",
                activeforeground="#ffffff",
                font=("Segoe UI", 9, "bold"),
                relief="flat",
                padx=8,
                pady=3,
                cursor="hand2",
                command=_toggle_pause,
            )
            btn_pause.pack(side="right", padx=(4, 4), pady=6)

            # Emergency keybinds
            root.bind("<Escape>", _trigger_takeover)

            self._is_active = True
            root.mainloop()
        except Exception:
            self._is_active = False

    def close(self) -> None:
        """Closes the overlay window safely."""
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
            self._root = None
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active


def launch_standalone_overlay() -> None:
    """Entry point when run directly: python -m hari_ai.agent.overlay."""
    stop_event = threading.Event()
    overlay = TakeoverOverlay(stop_event)
    print("Launching HariAI Takeover Overlay [ Pause ] [ Take over ]...")
    print("Press <Esc> or click [ Take over ] to dismiss.")
    overlay.start()
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        overlay.close()
        print("Takeover overlay closed.")


if __name__ == "__main__":
    launch_standalone_overlay()
