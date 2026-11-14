"""Microphone capture through PipeWire (pw-record) or PulseAudio (parec)."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading

from ..audio import SAMPLE_RATE


class Recorder:
    """Record 16 kHz mono 16-bit PCM from a source while start() ... stop() runs.

    `target` names a PipeWire/PulseAudio source; empty uses the default
    source, re-read at every recording so a newly plugged microphone is
    picked up.
    """

    def __init__(self, target: str = ""):
        self.target = target
        self.command = self._command()
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._chunks: list[bytes] = []

    def _command(self) -> list[str]:
        if shutil.which("pw-record"):
            command = ["pw-record", "--raw", "--format=s16", f"--rate={SAMPLE_RATE}", "--channels=1", "--latency=20ms"]
            if self.target:
                command += ["--target", self.target]
            return command + ["-"]
        if shutil.which("parec"):
            command = ["parec", "--raw", "--format=s16le", f"--rate={SAMPLE_RATE}", "--channels=1", "--latency-msec=20"]
            if self.target:
                command += ["--device", self.target]
            return command
        raise RuntimeError("no audio capture tool found: install pipewire (pw-record) or pulseaudio-utils (parec)")

    @property
    def recording(self) -> bool:
        return self._process is not None

    def start(self) -> None:
        if self._process:
            return
        self._chunks = []
        self._process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._reader = threading.Thread(target=self._read, args=(self._process,), daemon=True)
        self._reader.start()

    def _read(self, process: subprocess.Popen) -> None:
        fd = process.stdout.fileno()
        while chunk := os.read(fd, 8192):
            self._chunks.append(chunk)

    def stop(self) -> bytes:
        """Stop recording and return the captured PCM."""
        process, self._process = self._process, None
        if not process:
            return b""
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if self._reader:
            self._reader.join(timeout=2)
        pcm = b"".join(self._chunks)
        return pcm[: len(pcm) - len(pcm) % 2]
