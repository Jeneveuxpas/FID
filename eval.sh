#!/usr/bin/env bash
# Use the active training environment. Overrides can be appended after the checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
    echo 'Usage: bash eval.sh /path/to/checkpoints/1000000.pt [--output-dir results/step1M] [other run.py options]' >&2
    exit 2
fi
CKPT="$1"
shift
if [[ -z "${PYTHON:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PYTHON=python; else PYTHON=python3; fi
fi
exec "$PYTHON" "$SCRIPT_DIR/run.py" run \
    --ckpt "$CKPT" --model SiT-XL/2 --gpus 0,1,2,3 \
    --batch-size 64 --num-samples 50000 --mode sde --num-steps 250 \
    --cfg-scale 1.7 --guidance-low 0.0 --guidance-high 0.7 \
    --vae mse --seed 0 --output-dir results/step1M-cfg1.7 "$@"
