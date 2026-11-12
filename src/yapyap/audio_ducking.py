#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PlaybackStream:
    index: int
    volumes: tuple[int, ...]
    application_name: str


def _playback_streams(payload: str) -> list[PlaybackStream]:
    try:
        sink_inputs = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return []

    streams = []
    for sink_input in sink_inputs:
        if sink_input.get("corked") or sink_input.get("mute"):
            continue

        volumes = tuple(
            int(channel["value"])
            for channel in sink_input.get("volume", {}).values()
            if "value" in channel
        )
        if not volumes:
            continue

        properties = sink_input.get("properties", {})
        streams.append(
            PlaybackStream(
                index=int(sink_input["index"]),
                volumes=volumes,
                application_name=(
                    properties.get("application.name")
                    or properties.get("media.name")
                    or f"stream {sink_input['index']}"
                ),
            )
        )
    return streams


class PulseAudioDucker:
    """Smoothly lower active application streams while push-to-talk is held."""

    def __init__(
        self,
        enabled: bool,
        volume_ratio: float = 0.50,
        fade_out_ms: int = 120,
        fade_in_ms: int = 180,
        *,
        run: Callable = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.enabled = enabled and shutil.which("pactl") is not None
        self.volume_ratio = min(max(volume_ratio, 0.0), 1.0)
        self.fade_out_ms = max(fade_out_ms, 0)
        self.fade_in_ms = max(fade_in_ms, 0)
        self._run = run
        self._sleep = sleep
        self._lock = threading.Lock()
        self._persistence_lock = threading.Lock()
        self._persistence_guard_depth = 0
        self._wpctl_available = shutil.which("wpctl") is not None
        self._generation = 0
        self._ducked = False
        self._original_streams: dict[int, PlaybackStream] = {}
        self._worker: threading.Thread | None = None

    def duck(self) -> None:
        self._start_transition(ducked=True)

    def restore(self, wait: bool = False) -> None:
        worker = self._start_transition(ducked=False)
        if wait and worker:
            worker.join(timeout=max(self.fade_in_ms / 1000 + 1, 1))

    def _start_transition(self, ducked: bool) -> threading.Thread | None:
        if not self.enabled:
            return None

        with self._lock:
            if self._ducked == ducked:
                return self._worker
            self._ducked = ducked
            self._generation += 1
            generation = self._generation

        worker = threading.Thread(
            target=self._transition,
            args=(generation, ducked),
            name="yapyap-audio-duck",
            daemon=True,
        )
        with self._lock:
            self._worker = worker
        worker.start()
        return worker

    def _transition(self, generation: int, ducked: bool) -> None:
        persistence_paused = self._pause_stream_persistence()
        try:
            if ducked:
                current_streams = self._list_streams()
                with self._lock:
                    if generation != self._generation or not self._ducked:
                        return
                    for stream in current_streams:
                        self._original_streams.setdefault(stream.index, stream)
                    target_streams = {
                        stream.index: tuple(
                            round(volume * self.volume_ratio)
                            for volume in self._original_streams[stream.index].volumes
                        )
                        for stream in current_streams
                    }
                duration_ms = self.fade_out_ms
            else:
                with self._lock:
                    if generation != self._generation or self._ducked:
                        return
                    target_streams = {
                        index: stream.volumes
                        for index, stream in self._original_streams.items()
                    }
                current_by_index = {
                    stream.index: stream for stream in self._list_streams()
                }
                current_streams = [
                    current_by_index[index]
                    for index in target_streams
                    if index in current_by_index
                ]
                duration_ms = self.fade_in_ms

            if not current_streams:
                if not ducked:
                    self._finish_restore(generation)
                return

            starts = {stream.index: stream.volumes for stream in current_streams}
            steps = max(1, round(duration_ms / 15))
            step_delay = duration_ms / steps / 1000
            changed_streams = 0

            for step in range(1, steps + 1):
                with self._lock:
                    if generation != self._generation or self._ducked != ducked:
                        return

                progress = step / steps
                for stream_index, target_volumes in target_streams.items():
                    start_volumes = starts.get(stream_index)
                    if not start_volumes or len(start_volumes) != len(target_volumes):
                        continue
                    next_volumes = tuple(
                        round(start + (target - start) * progress)
                        for start, target in zip(start_volumes, target_volumes)
                    )
                    if self._set_stream_volume(stream_index, next_volumes):
                        changed_streams += step == 1

                if step < steps and step_delay:
                    self._sleep(step_delay)

            if ducked:
                names = ", ".join(
                    stream.application_name
                    for stream in current_streams
                    if stream.index in target_streams
                )
                print(
                    f"🔉 Ducked {changed_streams} playback stream(s) to "
                    f"{self.volume_ratio:.0%}: {names}",
                    file=sys.stderr,
                )
            else:
                self._finish_restore(generation)
                print(
                    f"🔊 Restored {changed_streams} playback stream(s)",
                    file=sys.stderr,
                )
        finally:
            if persistence_paused:
                self._resume_stream_persistence()

    def _pause_stream_persistence(self) -> bool:
        """Keep temporary ducking volumes out of WirePlumber's saved state."""
        if not self._wpctl_available:
            return False

        with self._persistence_lock:
            if self._persistence_guard_depth:
                self._persistence_guard_depth += 1
                return True

            try:
                result = self._run(
                    ["wpctl", "settings", "node.stream.restore-props"],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=1,
                )
                if "Value: true" not in result.stdout:
                    return False
                self._run(
                    [
                        "wpctl",
                        "settings",
                        "node.stream.restore-props",
                        "false",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1,
                )
            except (
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ):
                return False

            self._persistence_guard_depth = 1
            return True

    def _resume_stream_persistence(self) -> None:
        with self._persistence_lock:
            if not self._persistence_guard_depth:
                return
            self._persistence_guard_depth -= 1
            if self._persistence_guard_depth:
                return

            try:
                self._run(
                    [
                        "wpctl",
                        "settings",
                        "node.stream.restore-props",
                        "true",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1,
                )
            except (
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ):
                print(
                    "⚠️ Could not re-enable saved application volumes",
                    file=sys.stderr,
                )

    def _finish_restore(self, generation: int) -> None:
        with self._lock:
            if generation == self._generation and not self._ducked:
                self._original_streams.clear()

    def _list_streams(self) -> list[PlaybackStream]:
        try:
            result = self._run(
                ["pactl", "-f", "json", "list", "sink-inputs"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=1,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return []
        return _playback_streams(result.stdout)

    def _set_stream_volume(self, stream_index: int, volumes: tuple[int, ...]) -> bool:
        try:
            self._run(
                [
                    "pactl",
                    "set-sink-input-volume",
                    str(stream_index),
                    *(str(volume) for volume in volumes),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
        return True
