#!/usr/bin/env bash
set -euo pipefail

for model in rnn lstm gru lnn; do
  python -m rul.train --model "$model" --epochs 30
  python -m rul.evaluate --checkpoint "outputs/checkpoints/${model}_best.pt"
done
