#!/usr/bin/env bash
# Usage: ./separate.sh song.wav
# Demucs 6-stem model → isolates the guitar track (wet reference).
set -e
cd "$(dirname "$0")"
IN="$1"
[ -f "$IN" ] || { echo "usage: $0 <song.wav>"; exit 1; }
.venv/bin/demucs -n htdemucs_6s --two-stems=guitar "$IN"
echo "Guitar stem at: separated/htdemucs_6s/$(basename "${IN%.*}")/guitar.wav"
