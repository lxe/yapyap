"""The `yapyap` command: dictation client, file transcription and service setup."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .client import api
from .client.config import DEFAULT_CONFIG, config_path, load_config

SERVICE = """\
[Unit]
Description=Yap Yap push-to-talk dictation
After=pipewire.service pipewire-pulse.service

[Service]
ExecStart={exe} run
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""


def cmd_run(args) -> None:
    from .client.daemon import run

    run(load_config(args.config))


def cmd_transcribe(args) -> None:
    config = load_config(args.config).server
    url = args.server or config.url
    for path in args.files:
        text = api.transcribe(
            url,
            Path(path).read_bytes(),
            token=config.token,
            language=args.language or config.language,
            prompt=config.prompt,
            on_delta=(lambda piece: print(piece, end="", flush=True)) if args.stream else None,
            timeout=max(config.timeout, 300),
        )
        print() if args.stream else print(text)


def cmd_bench(args) -> None:
    import statistics
    import time
    import wave

    config = load_config(args.config).server
    url = args.server or config.url
    token = args.token if args.token is not None else config.token
    print(f"| Speech | First words | Full transcript |\n|---|---|---|")
    for path in args.files:
        data = Path(path).read_bytes()
        with wave.open(path) as wav:
            seconds = wav.getnframes() / wav.getframerate()
        runs = []
        for attempt in range(args.runs + 1):  # the first run warms up and is not counted
            start = time.perf_counter()
            first: list[float] = []
            api.transcribe(url, data, token=token, on_delta=lambda _: first or first.append(time.perf_counter() - start))
            total = time.perf_counter() - start
            if attempt:
                runs.append((first[0] if first else total, total))
        first_ms = 1000 * statistics.median(r[0] for r in runs)
        total_s = statistics.median(r[1] for r in runs)
        print(f"| {seconds:.0f} s | {first_ms:.0f} ms | {total_s:.2f} s |")


def cmd_check(args) -> None:
    config = load_config(args.config).server
    models = api.list_models(config.url, token=config.token)
    print(f"{config.url} is up; model: {', '.join(models) or 'unknown'}")


def cmd_install(args) -> None:
    path = config_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG, encoding="utf-8")
        path.chmod(0o600)  # it can hold an API token
        print(f"wrote {path}")
    exe = shutil.which("yapyap") or os.path.abspath(sys.argv[0])
    unit = Path(os.environ.get("XDG_CONFIG_HOME") or "~/.config").expanduser() / "systemd/user/yapyap.service"
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(SERVICE.format(exe=exe), encoding="utf-8")
    print(f"wrote {unit}")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "yapyap.service"], check=True)
    if not args.no_start:
        subprocess.run(["systemctl", "--user", "restart", "yapyap.service"], check=True)
        print("started yapyap.service; logs: journalctl --user -u yapyap -f")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="yapyap", description="Push-to-talk dictation for Linux.")
    parser.add_argument("--config", type=Path, help=f"config file (default: {config_path()})")
    parser.add_argument("-v", "--verbose", action="store_true", help="log debug details")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("run", help="run the dictation client").set_defaults(func=cmd_run)

    transcribe = commands.add_parser("transcribe", help="transcribe audio files with the server")
    transcribe.add_argument("files", nargs="+")
    transcribe.add_argument("--server", help="server URL (default: from the config file)")
    transcribe.add_argument("--language", help="language code, e.g. en (default: detect)")
    transcribe.add_argument("--stream", action="store_true", help="print text as it is decoded")
    transcribe.set_defaults(func=cmd_transcribe)

    commands.add_parser("check", help="check that the server is reachable").set_defaults(func=cmd_check)

    bench = commands.add_parser("bench", help="time the server on WAV files (median of several streamed runs)")
    bench.add_argument("files", nargs="+")
    bench.add_argument("--server", help="server URL (default: from the config file)")
    bench.add_argument("--token", help="API token (default: from the config file)")
    bench.add_argument("--runs", type=int, default=5)
    bench.set_defaults(func=cmd_bench)

    install = commands.add_parser("install", help="write the config file and systemd user service")
    install.add_argument("--no-start", action="store_true", help="enable the service without starting it")
    install.set_defaults(func=cmd_install)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s" if os.environ.get("INVOCATION_ID") else "%(asctime)s %(levelname)s %(message)s",
    )
    try:
        args.func(args)
    except (api.TranscriptionError, ValueError, OSError) as exc:
        sys.exit(f"yapyap: {exc}")


if __name__ == "__main__":
    main()
