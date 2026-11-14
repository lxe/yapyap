import json
import os
import socket
import tempfile
import time


def default_socket_path():
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return os.path.join(runtime_dir, f"yapyap-{os.getuid()}.sock")
    return os.path.join(tempfile.gettempdir(), f"yapyap-{os.getuid()}.sock")


SOCKET_PATH = os.environ.get("YAPYAP_SOCKET", default_socket_path())


def parse_hostport(value):
    if ":" not in value:
        raise ValueError(f"Expected HOST:PORT, got {value!r}")

    host, port_text = value.rsplit(":", 1)
    if not host:
        raise ValueError(f"Host is required in {value!r}")

    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"Invalid port in {value!r}") from exc

    if not (1 <= port <= 65535):
        raise ValueError(f"Port out of range in {value!r}")

    return host, port


def tcp_bind_address():
    host = os.environ.get("YAPYAP_TCP_HOST", "").strip()
    if not host:
        return None

    port_text = os.environ.get("YAPYAP_TCP_PORT", "").strip()
    if not port_text:
        raise ValueError("YAPYAP_TCP_PORT is required when YAPYAP_TCP_HOST is set")

    return parse_hostport(f"{host}:{port_text}")


def request_transcription(payload, *, host=None, socket_path=SOCKET_PATH, timeout=30, on_delta=None):
    """Send one request and return (response, timings).

    With on_delta, the request asks for streaming. A server that streams
    answers with {"delta": ...} lines, each passed to on_delta, before its
    final {"text": ...} line; a server that does not stream only sends the
    final line, so callers need no special case.
    """
    if on_delta is not None:
        payload = {**payload, "stream": True}
    if host:
        host_name, port = parse_hostport(host)
        family = socket.AF_INET
        address = (host_name, port)
    else:
        family = socket.AF_UNIX
        address = socket_path

    with socket.socket(family, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        connect_start = time.perf_counter()
        client.connect(address)
        connect_time = time.perf_counter() - connect_start
        request_start = time.perf_counter()
        client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        reader = client.makefile("r", encoding="utf-8")
        while True:
            response = reader.readline()
            if not response:
                break
            message = json.loads(response)
            if "delta" in message and "text" not in message and "error" not in message:
                if on_delta is not None:
                    on_delta(message["delta"])
                continue
            break
        roundtrip_time = time.perf_counter() - request_start

    if not response:
        raise ConnectionError("No response received from transcription server")

    return message, {
        "client_connect_s": connect_time,
        "client_request_roundtrip_s": roundtrip_time,
    }
