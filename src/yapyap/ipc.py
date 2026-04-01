import os
import tempfile


def default_socket_path():
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return os.path.join(runtime_dir, f"yapyap-{os.getuid()}.sock")
    return os.path.join(tempfile.gettempdir(), f"yapyap-{os.getuid()}.sock")


SOCKET_PATH = os.environ.get("YAPYAP_SOCKET", default_socket_path())
