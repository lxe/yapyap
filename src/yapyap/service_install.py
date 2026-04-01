#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SERVICE_NAME = "yapyap"
SERVICE_FILENAME = f"{SERVICE_NAME}.service"

ENV_TEMPLATE = """# Optional overrides for the yapyap user service.
# Uncomment and edit any of these values, then restart the service:
#   systemctl --user restart yapyap
#
# CHORD=KEY_LEFTCTRL,KEY_LEFTALT
# MODEL=large-v3-turbo-q8_0
# YAPYAP_WAYBAR_SIGNAL=11
# YAPYAP_SERVER_ONLY=1
# YAPYAP_TCP_HOST=0.0.0.0
# YAPYAP_TCP_PORT=8122
# YAPYAP_REMOTE_HOST=192.168.0.102:8122
# YAPYAP_AUTO_PASTE=1
# YAPYAP_PASTE_DELAY=0.05
"""

SERVICE_TEMPLATE = """# Managed by `yapyap-install-service`. Re-run that command after upgrades.
[Unit]
Description=YapYap dictation pipeline
After=default.target

[Service]
Type=simple
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-__ENV_FILE__
ExecStart=__RUNNER_PATH__
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""

RUNNER_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

STATE_FILE="${YAPYAP_STATE:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/yapyap-state-$(id -u).json}"
WAYBAR_SIGNAL="${YAPYAP_WAYBAR_SIGNAL:-11}"
AUTO_PASTE="${YAPYAP_AUTO_PASTE:-1}"
PASTE_DELAY="${YAPYAP_PASTE_DELAY:-0.05}"

YAPYAP_BIN=__YAPYAP_BIN__
PYTHON_BIN=__PYTHON_BIN__
HYPRCTL_BIN=__HYPRCTL_BIN__
WLCOPY_BIN=__WLCOPY_BIN__

signal_waybar() {
  if [ -n "$WAYBAR_SIGNAL" ]; then
    pkill "-RTMIN+${WAYBAR_SIGNAL}" waybar >/dev/null 2>&1 || true
  fi
}

cleanup() {
  "$PYTHON_BIN" - "$STATE_FILE" <<'PY'
import json
import os
import sys
import time

path = sys.argv[1]
payload = {
    "active": False,
    "recording": False,
    "pid": os.getpid(),
    "updated_at": time.time(),
}

os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
PY
  signal_waybar
}

trap cleanup EXIT

export YAPYAP_STATE="$STATE_FILE"
export YAPYAP_WAYBAR_SIGNAL="$WAYBAR_SIGNAL"

active_window_class() {
  local payload=""

  if [ -z "$HYPRCTL_BIN" ]; then
    printf '\\n'
    return 0
  fi

  if ! payload="$("$HYPRCTL_BIN" activewindow -j 2>/dev/null)"; then
    printf '\\n'
    return 0
  fi

  "$PYTHON_BIN" -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    print("")
    raise SystemExit(0)

value = payload.get("class") or payload.get("initialClass") or ""
print(str(value).strip().lower())
' <<<"$payload" || printf '\\n'
}

paste_shortcut_for_class() {
  local class_name="${1:-}"
  case "$class_name" in
    kitty|alacritty|foot|wezterm|org.wezfurlong.wezterm|cursor|code)
      printf '%s\\n' "CTRL+SHIFT,V,"
      ;;
    *)
      printf '%s\\n' "CTRL,V,"
      ;;
  esac
}

"$YAPYAP_BIN" | while IFS= read -r line; do
  local_window_class=""
  local_paste_shortcut=""

  [ -n "$line" ] || continue

  if [ -n "$WLCOPY_BIN" ]; then
    if ! "$WLCOPY_BIN" -- "$line"; then
      printf 'yapyap: wl-copy failed for transcript\\n' >&2
      continue
    fi
  else
    printf 'yapyap: wl-copy not found, skipping clipboard update\\n' >&2
    continue
  fi

  if [ "$AUTO_PASTE" != "1" ] || [ -z "$HYPRCTL_BIN" ]; then
    continue
  fi

  if ! local_window_class="$(active_window_class)"; then
    local_window_class=""
  fi
  if ! local_paste_shortcut="$(paste_shortcut_for_class "$local_window_class")"; then
    local_paste_shortcut="CTRL,V,"
  fi

  sleep "$PASTE_DELAY"
  printf 'yapyap: pasting transcript to class=%s via shortcut=%s\\n' \
    "${local_window_class:-unknown}" \
    "$local_paste_shortcut" >&2

  if ! "$HYPRCTL_BIN" dispatch sendshortcut "$local_paste_shortcut"; then
    printf 'yapyap: hyprctl sendshortcut failed for class=%s via shortcut=%s\\n' \
      "${local_window_class:-unknown}" \
      "$local_paste_shortcut" >&2
  fi
done
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install or update the yapyap systemd user service.",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Write files and enable the unit, but do not start or restart it.",
    )
    return parser.parse_args()


def resolve_command(name: str, *, allow_missing: bool = False) -> str:
    path = shutil.which(name)
    if path:
        return path

    if name == "yapyap":
        sibling = Path(sys.argv[0]).resolve().with_name("yapyap")
        if sibling.exists():
            return str(sibling)

    if allow_missing:
        return ""

    raise RuntimeError(f"Could not find required command: {name}")


def atomic_write(path: Path, content: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(content)
        tmp_path = Path(handle.name)
    os.chmod(tmp_path, mode)
    os.replace(tmp_path, path)


def render_service(env_path: Path, runner_path: Path) -> str:
    return (
        SERVICE_TEMPLATE.replace("__ENV_FILE__", str(env_path))
        .replace("__RUNNER_PATH__", str(runner_path))
    )


def render_runner() -> str:
    replacements = {
        "__YAPYAP_BIN__": shlex.quote(resolve_command("yapyap")),
        "__PYTHON_BIN__": shlex.quote(sys.executable),
        "__HYPRCTL_BIN__": shlex.quote(resolve_command("hyprctl", allow_missing=True)),
        "__WLCOPY_BIN__": shlex.quote(resolve_command("wl-copy", allow_missing=True)),
    }
    script = RUNNER_TEMPLATE
    for needle, replacement in replacements.items():
        script = script.replace(needle, replacement)
    return script


def run_systemctl(*args: str) -> None:
    systemctl = resolve_command("systemctl")
    subprocess.run([systemctl, "--user", *args], check=True)


def install_env_file(path: Path) -> bool:
    if path.exists():
        return False
    atomic_write(path, ENV_TEMPLATE, mode=0o644)
    return True


def main() -> None:
    args = parse_args()
    try:
        home = Path.home()
        service_path = home / ".config/systemd/user" / SERVICE_FILENAME
        runner_path = home / ".local/bin/yapyap-runner"
        env_path = home / ".config/yapyap/env"

        atomic_write(runner_path, render_runner(), mode=0o755)
        atomic_write(service_path, render_service(env_path, runner_path), mode=0o644)
        env_created = install_env_file(env_path)

        print(f"Wrote {runner_path}")
        print(f"Wrote {service_path}")
        if env_created:
            print(f"Created {env_path}")
        else:
            print(f"Kept existing {env_path}")

        run_systemctl("daemon-reload")
        run_systemctl("enable", SERVICE_NAME)
        if args.no_start:
            print(f"Enabled {SERVICE_NAME}; start it with: systemctl --user start {SERVICE_NAME}")
        else:
            run_systemctl("restart", SERVICE_NAME)
            print(f"Enabled and restarted {SERVICE_NAME}")
    except subprocess.CalledProcessError as exc:
        print(
            "Installed service files, but `systemctl --user` failed. "
            f"Run these commands manually:\n"
            f"  systemctl --user daemon-reload\n"
            f"  systemctl --user enable {SERVICE_NAME}\n"
            f"  systemctl --user {'start' if args.no_start else 'restart'} {SERVICE_NAME}",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode) from exc
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
