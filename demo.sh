#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
chmod +x bin/aim-linux-x86_64
./bin/aim-linux-x86_64 analyze --weights checkpoints/sample_262p.pt --out results/sample
printf '\nAIM program: results/sample/learned_program.aim\n'
