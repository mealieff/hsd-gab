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
  
  # Loop over embedding files only
  for EMB_FILE in "$DIR"/*_embeddings.npy; do
    # Skip if no matching files
    [[ -f "$EMB_FILE" ]] || { echo "No embedding files found in $DIR"; continue; }

    # Find corresponding label file
    LABEL_FILE="${EMB_FILE/_embeddings.npy/_labels.npy}"
    if [[ ! -f "$LABEL_FILE" ]]; then
      echo "WARNING: Label file not found for $EMB_FILE, skipping."
      continue
    fi

    echo "Running single-label training on $EMB_FILE"
    python3 sing_train.py \
      --train "$EMB_FILE" \
      --train_labels "$LABEL_FILE" \
      --test_embeddings "$TEST_EMB" \
      --test_labels "$TEST_LABELS" \
      --split_dev
  done
done

