"""Desktop session environment for helper tools (wl-copy, wtype)."""
from __future__ import annotations

import glob
import os


def wayland_env() -> dict[str, str]:
    """The current environment, plus WAYLAND_DISPLAY when a systemd user
    service started without it."""
    env = dict(os.environ)
    if not env.get("WAYLAND_DISPLAY"):
        runtime = env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
        sockets = [path for path in sorted(glob.glob(os.path.join(runtime, "wayland-[0-9]*"))) if not path.endswith(".lock")]
        if sockets:
            env["WAYLAND_DISPLAY"] = os.path.basename(sockets[0])
    return env
