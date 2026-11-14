import json
import socket
import threading
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from yapyap.audio import SAMPLE_RATE, pcm16_to_wav
from yapyap.client import api
from yapyap.server.app import create_app


class FakeEngine:
    model = "fake-whisper"

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, *, language=None, prompt=None, on_delta=None):
        self.calls.append({"seconds": len(audio) / SAMPLE_RATE, "language": language, "prompt": prompt})
        pieces = [" Привет,", " world", "."]
        for piece in pieces:
            if on_delta:
                on_delta(piece)
        return "".join(pieces).strip()


def wav(seconds=1.0):
    return pcm16_to_wav(b"\0\0" * int(seconds * SAMPLE_RATE))


@pytest.fixture
def engine():
    return FakeEngine()


def test_json_transcription_pads_audio(engine):
    client = TestClient(create_app(engine))
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav(1.0), "audio/wav")},
        data={"model": "whisper", "language": "ru", "prompt": "Привет, world"},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "Привет, world."}
    assert engine.calls == [{"seconds": 3.0, "language": "ru", "prompt": "Привет, world"}]


def test_text_response_format(engine):
    client = TestClient(create_app(engine))
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav(), "audio/wav")},
        data={"response_format": "text"},
    )
    assert response.text == "Привет, world."


def test_streaming_sends_deltas_then_done(engine):
    client = TestClient(create_app(engine))
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav(), "audio/wav")},
        data={"stream": "true"},
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]
    assert [e["type"] for e in events] == ["transcript.text.delta"] * 3 + ["transcript.text.done"]
    assert events[-1]["text"] == "Привет, world."


def test_token_is_required_when_set(engine):
    client = TestClient(create_app(engine, token="secret"))
    files = {"file": ("a.wav", wav(), "audio/wav")}
    assert client.post("/v1/audio/transcriptions", files=files).status_code == 401
    ok = client.post("/v1/audio/transcriptions", files=files, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200


def test_missing_file_and_bad_audio(engine, monkeypatch):
    client = TestClient(create_app(engine))
    assert client.post("/v1/audio/transcriptions", data={"model": "x"}).status_code == 400
    monkeypatch.setenv("PATH", "")  # no ffmpeg: only 16 kHz mono WAV is accepted
    bad = client.post("/v1/audio/transcriptions", files={"file": ("a.ogg", b"not audio", "audio/ogg")})
    assert bad.status_code == 400


@pytest.fixture
def live_server(engine):
    """The real app on a real port, for the stdlib HTTP client."""
    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(engine, token="t0k"), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_client_streams_from_server(live_server):
    pieces = []
    text = api.transcribe(live_server, wav(), token="t0k", on_delta=pieces.append)
    assert pieces == [" Привет,", " world", "."]
    assert text == "Привет, world."


def test_client_without_streaming(live_server):
    assert api.transcribe(live_server, wav(), token="t0k") == "Привет, world."


def test_client_reports_errors(live_server):
    with pytest.raises(api.TranscriptionError, match="401"):
        api.transcribe(live_server, wav(), token="wrong")
    with pytest.raises(api.TranscriptionError, match="cannot reach"):
        api.transcribe("http://127.0.0.1:9", wav())
    assert api.list_models(live_server, token="t0k") == ["fake-whisper"]
