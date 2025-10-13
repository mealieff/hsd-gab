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

# Mode flags (set exactly one to true)
HYPERPARAM_SEARCH=false
CONF_DEV=true

for DIR in "${DIRS[@]}"; do
  echo "== Processing directory: $DIR =="

  if [[ ! -d "$DIR" ]]; then
    echo "WARNING: Directory $DIR not found, skipping."
    continue
  fi

  # Find all *_single_label.npy files
  shopt -s nullglob
  FILES=("$DIR"/*_single_label.npy)
  shopt -u nullglob

  if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No *_single_label.npy files found in $DIR"
    continue
  fi

  # Process each file individually
  for FILE in "${FILES[@]}"; do
    FILE_NAME=$(basename "$FILE")
    PREFIX="${FILE_NAME%_single_label.npy}"
    echo "== Processing file: $FILE_NAME =="

    # --- Step 1: Dev set / confidence mode ---
    if [[ "$CONF_DEV" = true ]]; then
      echo "Running main.py (confidence dev set) on $FILE_NAME"
      DEV_LOG=$(mktemp)
      python3 main.py \
        --data_dir "$DIR" \
        --setting single \
        --use_confidence_dev \
        --confidence \
        --file "$FILE" > "$DEV_LOG" 2>&1

      BEST_THRESH=$(grep "\[INFO\] Best confidence threshold on dev set:" "$DEV_LOG" | awk '{print $7}')
      if [[ -z "$BEST_THRESH" ]]; then
        echo "ERROR: Could not extract best confidence threshold for $FILE_NAME"
        cat "$DEV_LOG"
        rm "$DEV_LOG"
        continue
      fi
      echo "[INFO] Best confidence threshold for $FILE_NAME = $BEST_THRESH"
      rm "$DEV_LOG"
    fi

    # --- Step 2: Retrain / final evaluation ---
    echo "Running main.py (final single-label) on $FILE_NAME with threshold=$BEST_THRESH"
    python3 main.py \
      --data_dir "$DIR" \
      --setting single \
      --confidence \
      --threshold "$BEST_THRESH" \
      --file "$FILE" \
      $( [[ "$HYPERPARAM_SEARCH" = true ]] && echo "--hyperparam_search" )
  done
done

