#!/usr/bin/env bash
# Launch the Parental Control System (macOS / Linux)
cd "$(dirname "$0")"
exec python3 main.py "$@"
