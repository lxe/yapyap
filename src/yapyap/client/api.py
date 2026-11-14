"""Client for an OpenAI-compatible transcription endpoint."""
from __future__ import annotations

import http.client
import json
import uuid
from typing import Callable
from urllib.parse import urlsplit


class TranscriptionError(RuntimeError):
    pass


def _multipart(fields: dict[str, str], filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode()
    )
    parts += [data, f"\r\n--{boundary}--\r\n".encode()]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe(
    url: str,
    wav: bytes,
    *,
    token: str = "",
    language: str = "",
    prompt: str = "",
    on_delta: Callable[[str], None] | None = None,
    timeout: float = 60.0,
) -> str:
    """Send a WAV recording to `url` and return the transcript.

    With on_delta, the request asks for a stream and on_delta receives each
    piece of text as the server decodes it. A server that does not stream
    answers with plain JSON, and on_delta is simply never called.
    """
    parts = urlsplit(url)
    connection_class = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    fields = {"model": "whisper"}
    if language:
        fields["language"] = language
    if prompt:
        fields["prompt"] = prompt
    if on_delta:
        fields["stream"] = "true"
    body, content_type = _multipart(fields, "speech.wav", wav)
    headers = {"Content-Type": content_type}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    path = (parts.path.rstrip("/") or "") + "/v1/audio/transcriptions"

    connection = connection_class(parts.hostname, parts.port, timeout=timeout)
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read().decode("utf-8", "replace")[:300]
            raise TranscriptionError(f"server answered {response.status}: {detail}")
        if response.getheader("Content-Type", "").startswith("text/event-stream"):
            return _read_stream(response, on_delta)
        return json.loads(response.read()).get("text", "").strip()
    except (OSError, http.client.HTTPException) as exc:
        raise TranscriptionError(f"cannot reach {url}: {exc}") from exc
    finally:
        connection.close()


def list_models(url: str, *, token: str = "", timeout: float = 10.0) -> list[str]:
    """Return the server's model names; raises TranscriptionError when unreachable or refused."""
    parts = urlsplit(url)
    connection_class = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    connection = connection_class(parts.hostname, parts.port, timeout=timeout)
    try:
        connection.request("GET", (parts.path.rstrip("/") or "") + "/v1/models", headers=headers)
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise TranscriptionError(f"server answered {response.status}: {body.decode('utf-8', 'replace')[:300]}")
        return [model.get("id", "") for model in json.loads(body).get("data", [])]
    except (OSError, http.client.HTTPException) as exc:
        raise TranscriptionError(f"cannot reach {url}: {exc}") from exc
    finally:
        connection.close()


def _read_stream(response, on_delta) -> str:
    pieces: list[str] = []
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        kind = event.get("type")
        if kind == "transcript.text.delta":
            pieces.append(event.get("delta", ""))
            if on_delta:
                on_delta(event.get("delta", ""))
        elif kind == "transcript.text.done":
            return event.get("text", "").strip()
        elif kind == "error":
            raise TranscriptionError(event.get("error", {}).get("message", "transcription failed"))
    return "".join(pieces).strip()
