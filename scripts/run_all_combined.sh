#!/usr/bin/env bash
set -euo pipefail

python scripts/build_combined_dataset.py

for model in rnn lstm gru lnn; do
  python -m rul.train \
    --model "$model" \
    --data-dir data/raw/FDALL \
    --output-dir outputs_combined \
    --epochs 30

  python -m rul.evaluate \
    --checkpoint "outputs_combined/checkpoints/${model}_best.pt" \
    --data-dir data/raw/FDALL \
    --output-dir outputs_combined
done

python -m rul.compare \
  --metrics-dir outputs_combined/metrics \
  --output outputs_combined/metrics/model_comparison.csv

python scripts/compare_before_after.py
