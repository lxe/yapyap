"""Yap Yap transcription server: an OpenAI-compatible speech-to-text API.

POST /v1/audio/transcriptions takes a multipart upload with `file` (WAV) and
optional `language`, `prompt`, `stream` and `response_format` fields, as in
OpenAI's API. With stream=true the answer is Server-Sent Events: one
`transcript.text.delta` per piece of text as the engine decodes it, then a
`transcript.text.done` with the whole transcript.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import io
import json
import logging
import os
import subprocess
import time
import wave

import numpy as np
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

from ..audio import SAMPLE_RATE, trailing_pad_seconds
from .engines import Engine, create_engine

log = logging.getLogger("yapyap.server")


class BadAudio(ValueError):
    pass


def decode_audio(data: bytes) -> np.ndarray:
    """Decode an upload to 16 kHz mono float32.

    16 kHz mono 16-bit WAV, which the Yap Yap client sends, is read
    directly; anything else goes through ffmpeg when it is installed.
    """
    try:
        with wave.open(io.BytesIO(data)) as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (SAMPLE_RATE, 1, 2):
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
                return pcm.astype(np.float32) / 32768.0
    except (wave.Error, EOFError):
        pass
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
             "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
            input=data, capture_output=True, timeout=120, check=True,
        )
    except FileNotFoundError:
        raise BadAudio("send 16 kHz mono 16-bit WAV, or install ffmpeg on the server for other formats")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise BadAudio("could not decode the audio file")
    return np.frombuffer(result.stdout, dtype=np.float32)


def with_trailing_silence(audio: np.ndarray) -> np.ndarray:
    pad = trailing_pad_seconds(len(audio) / SAMPLE_RATE)
    return np.concatenate([audio, np.zeros(int(pad * SAMPLE_RATE), dtype=np.float32)])


def create_app(engine: Engine, token: str | None = None) -> Starlette:
    def authorized(request: Request) -> bool:
        if not token:
            return True
        header = request.headers.get("authorization", "")
        return hmac.compare_digest(header.encode(), f"Bearer {token}".encode())

    async def health(request: Request):
        return JSONResponse({"status": "ok", "model": engine.model})

    async def models(request: Request):
        if not authorized(request):
            return JSONResponse({"error": "missing or wrong API token"}, status_code=401)
        return JSONResponse({"object": "list", "data": [{"id": engine.model, "object": "model"}]})

    async def transcriptions(request: Request):
        if not authorized(request):
            return JSONResponse({"error": "missing or wrong API token"}, status_code=401)
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            return JSONResponse({"error": "multipart field 'file' is required"}, status_code=400)
        try:
            audio = decode_audio(await upload.read())
        except BadAudio as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        audio_seconds = len(audio) / SAMPLE_RATE
        options = {
            "language": (form.get("language") or "").strip() or None,
            "prompt": (form.get("prompt") or "").strip() or None,
        }
        stream = str(form.get("stream", "")).lower() in {"true", "1"}
        response_format = form.get("response_format") or "json"
        audio = with_trailing_silence(audio)
        start = time.perf_counter()

        if not stream:
            text = await run_in_threadpool(engine.transcribe, audio, **options)
            log.info("transcribed %.1f s of audio in %.0f ms", audio_seconds, 1000 * (time.perf_counter() - start))
            if response_format == "text":
                return PlainTextResponse(text)
            return JSONResponse({"text": text})

        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        first: list[float] = []

        def on_delta(delta: str):
            if not first:
                first.append(time.perf_counter() - start)
            loop.call_soon_threadsafe(events.put_nowait, ("delta", delta))

        def run():
            try:
                text = engine.transcribe(audio, on_delta=on_delta, **options)
                loop.call_soon_threadsafe(events.put_nowait, ("done", text))
            except Exception as exc:  # reported to the client as an SSE error event
                log.exception("transcription failed")
                loop.call_soon_threadsafe(events.put_nowait, ("error", str(exc)))

        async def sse():
            task = asyncio.ensure_future(run_in_threadpool(run))
            try:
                while True:
                    kind, value = await events.get()
                    if kind == "delta":
                        event = {"type": "transcript.text.delta", "delta": value}
                    elif kind == "done":
                        event = {"type": "transcript.text.done", "text": value}
                    else:
                        event = {"type": "error", "error": {"message": value}}
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if kind != "delta":
                        break
            finally:
                await task
                log.info(
                    "streamed %.1f s of audio: first text %.0f ms, done %.0f ms",
                    audio_seconds, 1000 * first[0] if first else -1, 1000 * (time.perf_counter() - start),
                )

        return StreamingResponse(sse(), media_type="text/event-stream")

    return Starlette(routes=[
        Route("/health", health),
        Route("/v1/models", models),
        Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
    ])


def warm_up(engine: Engine) -> None:
    """Run one short transcription so the first real request is not slow."""
    start = time.perf_counter()
    engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
    log.info("warm-up done in %.0f ms", 1000 * (time.perf_counter() - start))


def main(argv: list[str] | None = None) -> None:
    env = os.environ.get
    parser = argparse.ArgumentParser(description="Yap Yap transcription server (OpenAI-compatible API).")
    parser.add_argument("--engine", default=env("YAPYAP_ENGINE", "openvino"), choices=["openvino", "whispercpp"])
    parser.add_argument("--model", default=env("YAPYAP_MODEL"),
                        help="model directory, Hugging Face repo (openvino) or whisper.cpp model name")
    parser.add_argument("--device", default=env("YAPYAP_DEVICE", "GPU"), help="OpenVINO device: GPU, NPU or CPU")
    parser.add_argument("--threads", type=int, default=int(env("YAPYAP_THREADS", "0")) or None,
                        help="whisper.cpp CPU threads (default: all cores)")
    parser.add_argument("--cache-dir", default=env("YAPYAP_CACHE_DIR"), help="where downloaded models are kept")
    parser.add_argument("--host", default=env("YAPYAP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env("YAPYAP_PORT", "8122")))
    parser.add_argument("--token-file", default=env("YAPYAP_TOKEN_FILE"),
                        help="file holding the API token clients must send (or set YAPYAP_TOKEN)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = env("YAPYAP_TOKEN")
    if args.token_file:
        with open(args.token_file, encoding="utf-8") as handle:
            token = handle.read().strip()
    if not token and args.host not in {"127.0.0.1", "localhost", "::1"}:
        log.warning("listening on %s without an API token: anyone who can reach it can use it", args.host)

    default_models = {"openvino": "OpenVINO/whisper-large-v3-turbo-fp16-ov", "whispercpp": "large-v3-turbo-q8_0"}
    engine = create_engine(
        args.engine, args.model or default_models[args.engine],
        device=args.device, cache_dir=args.cache_dir, threads=args.threads,
    )
    warm_up(engine)

    import uvicorn

    uvicorn.run(create_app(engine, token), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
