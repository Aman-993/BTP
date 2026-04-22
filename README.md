# RUL Prediction on NASA C-MAPSS with RNNs and LNNs

This repo implements a beginner-friendly comparison inspired by the paper:

**Accuracy, Memory Efficiency and Generalization: A Comparative Study on Liquid Neural Networks and Recurrent Neural Networks**

The paper is a review/comparison of RNN, LSTM, GRU, and Liquid Neural Network ideas. This repo applies those ideas to jet engine Remaining Useful Life prediction on the NASA C-MAPSS dataset. You can train on **FD001 only** or on a combined **FD001-FD004** dataset.

## 1. Repo Structure

```text
rul-lnn-rnn-cmapss/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/
│       └── FD001/
│           ├── train_FD001.txt
│           ├── test_FD001.txt
│           └── RUL_FD001.txt
│       └── FDALL/
│           ├── train_FDALL.txt
│           ├── test_FDALL.txt
│           └── RUL_FDALL.txt
├── outputs/
│   ├── checkpoints/
│   └── metrics/
├── rul/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare.py
│   └── utils.py
└── scripts/
    ├── build_combined_dataset.py
    ├── run_all.sh
    └── run_all_combined.sh
```

## 2. What Each File Does

`rul/data.py`

Loads the C-MAPSS text files, adds RUL labels, scales sensor values, removes constant columns, and converts each engine history into fixed-length sequences.

`rul/models.py`

Contains all model architectures:

- `rnn`: vanilla recurrent neural network
- `lstm`: long short-term memory network
- `gru`: gated recurrent unit
- `lnn`: a compact Liquid Time-Constant style network

`rul/train.py`

Trains one selected model and saves the best checkpoint by validation RMSE.

`rul/evaluate.py`

Loads a saved checkpoint and evaluates it on the FD001 test engines.

`rul/compare.py`

Combines all test metric JSON files into one comparison CSV/table.

`rul/utils.py`

Small helper functions for seeds, metrics, device selection, parameter counting, and JSON saving.

`scripts/run_all.sh`

Runs training and evaluation for all four models.

`scripts/build_combined_dataset.py`

Combines FD001, FD002, FD003, and FD004 into one larger dataset while safely renumbering engine IDs.

`scripts/run_all_combined.sh`

Builds the combined dataset, trains all four models on it, evaluates them, and creates a comparison table.

## 3. Install Dependencies

From this repo folder:

```bash
cd /Users/shreya/Documents/Codex/2026-04-21-files-mentioned-by-the-user-rul/rul-lnn-rnn-cmapss
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you have an Apple Silicon Mac, PyTorch should use `mps` automatically when available. Otherwise it uses CPU or CUDA.

## 4. Dataset

Your FD001 files are already copied here:

```text
data/raw/FD001/train_FD001.txt
data/raw/FD001/test_FD001.txt
data/raw/FD001/RUL_FD001.txt
```

FD001 columns are:

```text
unit, cycle, setting_1, setting_2, setting_3, sensor_1 ... sensor_21
```

The training file does not contain RUL directly, so the code creates it:

```text
RUL = last_cycle_of_engine - current_cycle
```

The test file stops before failure, so the code uses `RUL_FD001.txt` to compute the final test labels.

## 5. Train One Model

Start with LSTM because it is usually a strong baseline:

```bash
python -m rul.train --model lstm --epochs 30
```

Train the other models:

```bash
python -m rul.train --model rnn --epochs 30
python -m rul.train --model gru --epochs 30
python -m rul.train --model lnn --epochs 30
```

For a fast smoke test, use fewer epochs:

```bash
python -m rul.train --model lnn --epochs 2
```

## 6. Evaluate a Model

```bash
python -m rul.evaluate --checkpoint outputs/checkpoints/lstm_best.pt
```

This creates:

```text
outputs/metrics/lstm_test_metrics.json
outputs/metrics/lstm_test_predictions.csv
```

For K-step rollout, first train a next-step feature prediction checkpoint:

```bash
python -m rul.train --model lstm --task next_step --epochs 30
```

Then evaluate rollout with that next-step checkpoint:

```bash
python -m rul.evaluate \
  --checkpoint outputs/checkpoints/lstm_next_step_best.pt \
  --data-dir data/raw/FD001 \
  --rollout \
  --rollout-k 1 2 3 5
```

Do not use `outputs/checkpoints/lstm_best.pt` for rollout. That checkpoint predicts scalar RUL, while rollout needs the next full feature vector.

## 7. Run All Models

```bash
bash scripts/run_all.sh
```

Then make one comparison table:

```bash
python -m rul.compare
```

Output:

```text
outputs/metrics/model_comparison.csv
```

## 8. Combine FD001-FD004 and Train Again

FD001 alone is useful for a clean first experiment. To test whether using all four datasets improves the models, combine FD001, FD002, FD003, and FD004 into one larger dataset:

```bash
python scripts/build_combined_dataset.py
```

This creates:

```text
data/raw/FDALL/train_FDALL.txt
data/raw/FDALL/test_FDALL.txt
data/raw/FDALL/RUL_FDALL.txt
```

Important: the script renumbers engine IDs. This is necessary because FD001, FD002, FD003, and FD004 all start their unit numbers from `1`, so directly concatenating the files would incorrectly merge different engines.

The combined dataset currently has:

```text
train rows: 160359
train units: 709
test rows: 104897
test units: 707
RUL rows: 707
```

Train one model on the combined dataset:

```bash
python -m rul.train \
  --model lstm \
  --data-dir data/raw/FDALL \
  --output-dir outputs_combined \
  --epochs 30
```

Evaluate it:

```bash
python -m rul.evaluate \
  --checkpoint outputs_combined/checkpoints/lstm_best.pt \
  --data-dir data/raw/FDALL \
  --output-dir outputs_combined
```

Run all four models on the combined dataset:

```bash
bash scripts/run_all_combined.sh
```

Combined-dataset results will be saved separately:

```text
outputs_combined/metrics/model_comparison.csv
```

This keeps your FD001-only results in `outputs/` and your FD001-FD004 results in `outputs_combined/`, so you can compare before vs after.

## 9. Important Training Options

Change sequence length:

```bash
python -m rul.train --model gru --sequence-length 50
```

Change hidden size:

```bash
python -m rul.train --model lnn --hidden-size 128
```

Use more liquid solver steps:

```bash
python -m rul.train --model lnn --solver-steps 3
```

Use CPU explicitly:

```bash
python -m rul.train --model lstm --device cpu
```

## 10. How the LNN in This Repo Works

The LNN model is implemented as a simple Liquid Time-Constant style recurrent cell:

```text
h(t+1) = h(t) + dt * (-h(t) + tanh(Wx + Uh)) / tau(x, h)
```

The important part is `tau(x, h)`. It is learned from the input and hidden state, so the model can adapt its time constant while processing the sequence. This is the main liquid-network idea used here.

## 11. Suggested Experiment Table

After running all models, compare:

```text
model | test_rmse | test_mae | test_score | parameter_count
```

Lower `test_rmse`, `test_mae`, and `test_score` are better. Lower `parameter_count` means the model is more memory efficient.

For the before/after dataset-size comparison, use two tables:

```text
FD001 only:       outputs/metrics/model_comparison.csv
FD001-FD004 all: outputs_combined/metrics/model_comparison.csv
```

Or create one before/after table:

```bash
python scripts/compare_before_after.py
```

Output:

```text
outputs_combined/metrics/before_after_comparison.csv
```

## 12. Beginner Debug Checklist

If `ModuleNotFoundError: No module named 'torch'` appears, activate the environment and reinstall:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If a data file is missing, make sure these three files exist:

```text
data/raw/FD001/train_FD001.txt
data/raw/FD001/test_FD001.txt
data/raw/FD001/RUL_FD001.txt
```

If training is slow, reduce epochs first:

```bash
python -m rul.train --model lstm --epochs 5
```
