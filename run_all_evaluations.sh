#!/bin/bash

TEST_EMB="test_embeddings.npy"
TEST_LABELS="test_labels.npy"

# Check test files exist
if [[ ! -f "$TEST_EMB" || ! -f "$TEST_LABELS" ]]; then
  echo "ERROR: Required test files not found: $TEST_EMB or $TEST_LABELS"
  exit 1
fi

for DIR in sing_label_data sing_label_data2_1 sing_label_data3_1; do
  echo "== Directory: $DIR =="
  for FILE in "$DIR"/*.npy; do
    if [[ ! -f "$FILE" ]]; then
      echo "ERROR: Training file not found: $FILE"
      exit 1
    fi
    python3 sing_train.py \
      --train "$FILE" \
      --test_embeddings "$TEST_EMB" \
      --test_labels "$TEST_LABELS" \
      --split_dev
    done
  done
done

