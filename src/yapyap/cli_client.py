#!/usr/bin/env python3
import argparse
import base64
import json
import os
import socket
import sys
import time

from .ipc import SOCKET_PATH, parse_hostport


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send a transcription request to a running yapyap instance.",
    )
    parser.add_argument(
        "--host",
        help="Connect over TCP instead of the local Unix socket using HOST:PORT.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print client and server timings to stderr.",
    )
    parser.add_argument("audio_file")
    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = os.path.abspath(args.audio_file)
    payload = {"trace": args.trace}

    if args.host:
        host, port = parse_hostport(args.host)
        with open(audio_path, "rb") as handle:
            payload["audio_b64"] = base64.b64encode(handle.read()).decode("ascii")
        payload["filename"] = os.path.basename(audio_path)
        family = socket.AF_INET
        address = (host, port)
        missing_error = None
        refused_message = f"yapyap is not accepting TCP requests on {host}:{port}."
    else:
        payload["path"] = audio_path
        family = socket.AF_UNIX
        address = SOCKET_PATH
        missing_error = f"yapyap is not running. Start `yapyap` first so {SOCKET_PATH} exists."
        refused_message = f"yapyap is not accepting CLI requests on {SOCKET_PATH}."

    client_start = time.perf_counter()
    try:
        with socket.socket(family, socket.SOCK_STREAM) as client:
            connect_start = time.perf_counter()
            client.connect(address)
            connect_time = time.perf_counter() - connect_start
            request_start = time.perf_counter()
            client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            response = client.makefile("r", encoding="utf-8").readline()
            roundtrip_time = time.perf_counter() - request_start
    except FileNotFoundError:
        print(missing_error or f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    except ConnectionRefusedError:
        print(refused_message, file=sys.stderr)
        sys.exit(1)
    except TimeoutError:
        print(f"Timed out connecting to {args.host}.", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(response)
    if "error" in payload:
        print(payload["error"], file=sys.stderr)
        sys.exit(1)

    if args.trace:
        print(f"client_connect_s={connect_time:.6f}", file=sys.stderr)
        print(f"client_request_roundtrip_s={roundtrip_time:.6f}", file=sys.stderr)
        print(f"client_total_s={time.perf_counter() - client_start:.6f}", file=sys.stderr)
        for key, value in payload.get("timings", {}).items():
            if isinstance(value, float):
                print(f"{key}={value:.6f}", file=sys.stderr)
            else:
                print(f"{key}={value}", file=sys.stderr)

    if payload["text"]:
        print(payload["text"], flush=True)


if __name__ == "__main__":
    main()
