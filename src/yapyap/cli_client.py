#!/usr/bin/env python3
import json
import os
import socket
import sys
import time

from .ipc import SOCKET_PATH


def main():
    trace = False
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--trace":
        trace = True
        args = args[1:]

    if len(args) != 1:
        print("Usage: yapyap-cli [--trace] <audio_file>", file=sys.stderr)
        sys.exit(1)

    client_start = time.perf_counter()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            connect_start = time.perf_counter()
            client.connect(SOCKET_PATH)
            connect_time = time.perf_counter() - connect_start
            request_start = time.perf_counter()
            client.sendall((json.dumps({"path": os.path.abspath(args[0]), "trace": trace}) + "\n").encode("utf-8"))
            response = client.makefile("r", encoding="utf-8").readline()
            roundtrip_time = time.perf_counter() - request_start
    except FileNotFoundError:
        print(f"yapyap is not running. Start `yapyap` first so {SOCKET_PATH} exists.", file=sys.stderr)
        sys.exit(1)
    except ConnectionRefusedError:
        print(f"yapyap is not accepting CLI requests on {SOCKET_PATH}.", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(response)
    if "error" in payload:
        print(payload["error"], file=sys.stderr)
        sys.exit(1)

    if trace:
        print(f"client_connect_s={connect_time:.6f}", file=sys.stderr)
        print(f"client_request_roundtrip_s={roundtrip_time:.6f}", file=sys.stderr)
        print(f"client_total_s={time.perf_counter() - client_start:.6f}", file=sys.stderr)
        for key, value in payload.get("timings", {}).items():
            print(f"{key}={value:.6f}", file=sys.stderr)

    if payload["text"]:
        print(payload["text"], flush=True)


if __name__ == "__main__":
    main()
