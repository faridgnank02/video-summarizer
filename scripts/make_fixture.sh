#!/usr/bin/env bash
# Generates tests/fixtures/spoken_30s.wav using macOS `say` + ffmpeg.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tests/fixtures
TEXT="This is a test recording for the video intelligence pipeline. \
The pipeline transcribes audio with whisper, segments it into chapters, \
and synthesizes a structured report with key quotes and action items. \
This sentence exists purely so the smoke test has known words to find: \
pipeline, whisper, chapters, report."
say -o /tmp/vi_fixture.aiff "$TEXT"
ffmpeg -y -i /tmp/vi_fixture.aiff -ac 1 -ar 16000 tests/fixtures/spoken_30s.wav
echo "wrote tests/fixtures/spoken_30s.wav"
