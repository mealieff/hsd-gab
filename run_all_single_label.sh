#!/bin/bash

TEST_EMB="test_embeddings.npy"
TEST_LABELS="test_labels.npy"

# Check test files exist
if [[ ! -f "$TEST_EMB" || ! -f "$TEST_LABELS" ]]; then
  echo "ERROR: Required test files not found: $TEST_EMB or $TEST_LABELS"
  exit 1
fi

# Directories to process
DIRS=("sing_label_data" "sing_label_data2_1" "sing_label_data3_1")

for DIR in "${DIRS[@]}"; do
  echo "== Processing directory: $DIR =="

  if [[ ! -d "$DIR" ]]; then
    echo "WARNING: Directory $DIR not found, skipping."
    continue
  fi

  # Check for *_single_label.npy files
  shopt -s nullglob
  FILES=("$DIR"/*_single_label.npy)
  shopt -u nullglob

  if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No *_single_label.npy files found in $DIR"
    continue
  fi

  # --- Step 1: Run with split_dev to select threshold ---
  echo "Running main.py (dev search) on $DIR"
  DEV_LOG=$(mktemp)
  python3 main.py \
    --data_dir "$DIR" \
    --setting single \
    --split_dev \
    --confidence > "$DEV_LOG" 2>&1

  # Extract best threshold
  BEST_THRESH=$(grep "\[THRESHOLD\] Best" "$DEV_LOG" | awk '{print $3}')
  if [[ -z "$BEST_THRESH" ]]; then
    echo "ERROR: Could not extract best threshold for $DIR"
    cat "$DEV_LOG"
    rm "$DEV_LOG"
    continue
  fi
  echo "[INFO] Best threshold for $DIR = $BEST_THRESH"
  rm "$DEV_LOG"

  # --- Step 2: Retrain on full train set (no split_dev) with best threshold ---
  echo "Running main.py (final single-label) on $DIR with threshold=$BEST_THRESH"
  python3 main.py \
    --data_dir "$DIR" \
    --setting single \
    --confidence \
    --threshold "$BEST_THRESH"
done
