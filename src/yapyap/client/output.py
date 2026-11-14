"""Where transcripts go: typed as keys, pasted, or printed."""
from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading

from . import hyprland
from .config import OutputConfig
from .session import wayland_env

log = logging.getLogger("yapyap")

# Windows that paste with Ctrl+Shift+V rather than Ctrl+V.
TERMINAL_CLASSES = {
    "kitty", "foot", "alacritty", "wezterm", "org.wezfurlong.wezterm", "com.mitchellh.ghostty",
    "konsole", "org.kde.konsole", "gnome-terminal-server", "xterm", "code", "cursor",
}


def copy_to_clipboard(text: str) -> None:
    if shutil.which("wl-copy"):
        subprocess.run(["wl-copy", "--", text], check=False, env=wayland_env())
    elif shutil.which("xclip"):
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=False)


class Output:
    """Receives one transcript at a time: begin(), then piece() for each
    streamed piece of text, then finish() with the whole transcript."""

    streams = False

    def __init__(self, config: OutputConfig):
        self.config = config

    def begin(self) -> None:
        pass

    def piece(self, text: str) -> None:
        pass

    def finish(self, text: str) -> None:
        if self.config.clipboard and text:
            copy_to_clipboard(text)

    def close(self) -> None:
        pass


class TypeOutput(Output):
    """Types each piece as it streams in, on a worker thread.

    Pieces that arrive while earlier ones are being typed go out together,
    and after a quarter second with nothing to type the keyboard returns to
    its first layout group.
    """

    streams = True
    IDLE_RESTORE_S = 0.25

    def __init__(self, config: OutputConfig):
        super().__init__(config)
        from .typer import Typer

        self.typer = Typer(config)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._typed_any = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def begin(self) -> None:
        self._typed_any = False

    def piece(self, text: str) -> None:
        if text:
            self._typed_any = True
            self._queue.put(text)

    def finish(self, text: str) -> None:
        if not self._typed_any and text:
            self._queue.put(text)
        super().finish(text)

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=5)
        self.typer.close()

    def _run(self) -> None:
        while True:
            try:
                first = self._queue.get(timeout=self.IDLE_RESTORE_S)
            except queue.Empty:
                self.typer.restore_layout()
                continue
            if first is None:
                return
            batch = [first]
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._queue.put(None)
                    break
                batch.append(item)
            try:
                self.typer.wait_for_modifiers_released()
                self.typer.type_text("".join(batch))
            except OSError:
                log.exception("typing failed")


class PasteOutput(Output):
    """Copies each transcript and pastes it into the focused window (Hyprland)."""

    def finish(self, text: str) -> None:
        if not text:
            return
        copy_to_clipboard(text)
        window = hyprland.active_window()
        if not window:
            log.info("no Hyprland window to paste into; the transcript is on the clipboard")
            return
        window_class, address = window
        shortcut = "CTRL+SHIFT,V" if window_class in TERMINAL_CLASSES else "CTRL,V"
        hyprland.send_shortcut(shortcut, address)


class StdoutOutput(Output):
    """Prints one line per transcript, for piping into other tools."""

    def finish(self, text: str) -> None:
        if text:
            print(text.replace("\n", " "), flush=True)
        super().finish(text)


def create_output(config: OutputConfig) -> Output:
    if config.mode == "type":
        return TypeOutput(config)
    if config.mode == "paste":
        return PasteOutput(config)
    return StdoutOutput(config)
