"""The dictation loop: hold the chord to record, release to transcribe and output."""
from __future__ import annotations

import logging
import queue
import signal
import sys
import threading
import time

from ..audio import SAMPLE_RATE, pcm16_to_wav
from . import api
from .capture import Recorder
from .config import Config
from .ducking import PulseAudioDucker
from .hotkey import ChordListener
from .output import create_output
from .status import Status

log = logging.getLogger("yapyap")

# Shorter recordings are an accidental tap of the chord, not speech.
MIN_SECONDS = 0.3


class Dictation:
    def __init__(self, config: Config):
        self.config = config
        self.recorder = Recorder(config.capture.target)
        self.ducker = PulseAudioDucker(
            config.ducking.enabled,
            config.ducking.volume,
            config.ducking.fade_out_ms,
            config.ducking.fade_in_ms,
        )
        self.status = Status(config.status.waybar_signal)
        self.output = create_output(config.output)
        self.jobs: queue.Queue[bytes | None] = queue.Queue()
        self.worker = threading.Thread(target=self._transcribe_jobs, daemon=True)

    def start_recording(self) -> None:
        self.ducker.duck()
        try:
            self.recorder.start()
        except OSError:
            log.exception("could not start recording")
            self.ducker.restore()
            return
        self.status.update(active=True, recording=True)
        log.info("recording")

    def stop_recording(self) -> None:
        pcm = self.recorder.stop()
        self.ducker.restore()
        self.status.update(active=True, recording=False)
        seconds = len(pcm) / 2 / SAMPLE_RATE
        if seconds < MIN_SECONDS:
            log.info("ignored a %.2f s recording", seconds)
            return
        self.jobs.put(pcm)

    def _transcribe_jobs(self) -> None:
        while (pcm := self.jobs.get()) is not None:
            self._transcribe(pcm)

    def _transcribe(self, pcm: bytes) -> None:
        server = self.config.server
        seconds = len(pcm) / 2 / SAMPLE_RATE
        start = time.perf_counter()
        first: list[float] = []

        def on_piece(text: str) -> None:
            if not first:
                first.append(time.perf_counter() - start)
                text = text.lstrip()  # the transcript is stripped; match it
            self.output.piece(text)

        self.output.begin()
        try:
            text = api.transcribe(
                server.url,
                pcm16_to_wav(pcm),
                token=server.token,
                language=server.language,
                prompt=server.prompt,
                on_delta=on_piece if self.output.streams else None,
                timeout=server.timeout,
            )
        except api.TranscriptionError as exc:
            log.error("%s", exc)
            return
        self.output.finish(text)
        log.info(
            "%.1f s of speech: first words after %s, transcript after %.0f ms, %d characters",
            seconds,
            f"{1000 * first[0]:.0f} ms" if first else "-",
            1000 * (time.perf_counter() - start),
            len(text),
        )

    def run(self) -> None:
        self.worker.start()
        self.status.update(active=True, recording=False)
        listener = ChordListener(self.config.hotkey.chord, self.start_recording, self.stop_recording)
        log.info("hold %s to dictate; transcribing with %s", "+".join(self.config.hotkey.chord), self.config.server.url)
        try:
            listener.run()
        finally:
            if self.recorder.recording:
                self.recorder.stop()
            self.ducker.restore(wait=True)
            self.jobs.put(None)
            self.worker.join(timeout=30)
            self.output.close()
            self.status.update(active=False, recording=False)


def run(config: Config) -> None:
    # Exit through the `finally` blocks on SIGTERM (systemd stop) too, so
    # audio volume and keyboard layout are always restored.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        Dictation(config).run()
    except KeyboardInterrupt:
        pass
