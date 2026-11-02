#!/usr/bin/env bash
# Full blind pipeline: song -> stem -> estimated dry -> aligned pair -> .nam model
# Usage: ./pipeline.sh <song.wav> [deeffect_mode] [epochs]
#   deeffect_mode: separator (default) | bypass | remfx
#   for bypass also set DRY=<path to your dry wav>
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

SONG="$1"
MODE="${2:-separator}"
EPOCHS="${3:-40}"
[ -f "$SONG" ] || { echo "usage: $0 <song.wav> [separator|bypass|remfx] [epochs]"; exit 1; }
NAME=$(basename "${SONG%.*}")
WORK="runs/$NAME"

mkdir -p "$WORK"

echo "==> 1/4 stem separation"
./separate.sh "$SONG"
WET="separated/htdemucs_6s/$NAME/guitar.wav"

echo "==> 2/4 de-effect -> estimated dry"
if [ "$MODE" = bypass ]; then
  [ -f "$DRY" ] || { echo "bypass needs DRY=<dry.wav>"; exit 1; }
  $PY deeffect.py bypass "$WET" "$WORK/est_dry.wav" "$DRY"
else
  $PY deeffect.py "$MODE" "$WET" "$WORK/est_dry.wav" ${DEEFECT_MODEL:+$DEEFECT_MODEL}
fi

echo "==> 3/4 align pair"
$PY align.py "$WORK/est_dry.wav" "$WET" "$WORK/pair"

echo "==> 4/4 train ($EPOCHS epochs)"
$PY train.py "$WORK/pair" "$WORK/model" --epochs "$EPOCHS"

echo "Done. Model artifacts in $WORK/model"
echo "Load the .nam file into a NAM plugin/player and feed it your DI."
