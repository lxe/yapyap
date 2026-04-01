import os
import tempfile


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
