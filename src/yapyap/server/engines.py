"""Speech recognition engines behind the Yap Yap server.

Each engine turns 16 kHz mono float32 audio into text and can report text
while it decodes, which the server forwards as a stream.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger("yapyap.server")

OnDelta = Callable[[str], None]


class Engine(Protocol):
    model: str

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        prompt: str | None = None,
        on_delta: OnDelta | None = None,
    ) -> str: ...


class OpenVINOEngine:
    """Whisper through OpenVINO GenAI, for Intel GPUs, NPUs and CPUs.

    `model` is a local directory with an OpenVINO export, or a Hugging Face
    repository such as OpenVINO/whisper-large-v3-turbo-fp16-ov, downloaded
    into `cache_dir` on first start.
    """

    def __init__(self, model: str, device: str = "GPU", cache_dir: str | None = None):
        import openvino_genai

        self._genai = openvino_genai
        self.model = model
        path = resolve_model_dir(model, cache_dir)
        log.info("loading %s on %s", path, device)
        self._pipe = openvino_genai.WhisperPipeline(str(path), device)
        self._lock = threading.Lock()

    def transcribe(self, audio, *, language=None, prompt=None, on_delta=None):
        config = self._pipe.get_generation_config()
        config.task = "transcribe"
        if language:
            config.language = f"<|{language}|>"
        if prompt:
            config.initial_prompt = prompt

        def streamer(chunk: str):
            if on_delta and chunk:
                on_delta(chunk)
            return self._genai.StreamingStatus.RUNNING

        with self._lock:
            result = self._pipe.generate(audio.tolist(), config, streamer=streamer)
        return result.texts[0].strip()


class WhisperCppEngine:
    """Whisper through whisper.cpp (pywhispercpp), for CPUs and NVIDIA GPUs.

    `model` is a whisper.cpp model name such as large-v3-turbo-q8_0, which
    pywhispercpp downloads on first use, or a path to a ggml model file.
    Text streams one segment at a time.
    """

    def __init__(self, model: str, threads: int | None = None):
        from pywhispercpp.model import Model

        self.model = model
        log.info("loading whisper.cpp model %s", model)
        self._model = Model(model, n_threads=threads or os.cpu_count() or 4, print_progress=False, print_realtime=False)
        self._lock = threading.Lock()

    def transcribe(self, audio, *, language=None, prompt=None, on_delta=None):
        def on_segment(segment):
            # pywhispercpp passes one segment (or, in older versions, a list).
            if on_delta:
                for item in segment if isinstance(segment, list) else [segment]:
                    on_delta(item.text)

        params = {"language": language or "auto"}
        if prompt:
            params["initial_prompt"] = prompt
        with self._lock:
            segments = self._model.transcribe(audio, new_segment_callback=on_segment, **params)
        return " ".join(segment.text.strip() for segment in segments).strip()


def resolve_model_dir(model: str, cache_dir: str | None) -> Path:
    """Return a local model directory, downloading a Hugging Face repo if needed."""
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, cache_dir=cache_dir))


def create_engine(name: str, model: str, *, device: str, cache_dir: str | None, threads: int | None) -> Engine:
    if name == "openvino":
        return OpenVINOEngine(model, device=device, cache_dir=cache_dir)
    if name == "whispercpp":
        return WhisperCppEngine(model, threads=threads)
    raise ValueError(f"unknown engine {name!r}; choose openvino or whispercpp")
