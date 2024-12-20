# Modified script for ADASYN over-sampling
# script heavily adapted from Imbalanced-learn (imported as imblearn) is an open source, MIT-licensed library relying on scikit-learn

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import ADASYN

# Paths for input files
INPUT_EMBEDDINGS = "train_embeddings.npy"
INPUT_LABELS = "train_labels.npy"

# Load embeddings and labels
print("Loading embeddings and labels...")
X_train = np.load(INPUT_EMBEDDINGS)  # BERT embeddings
y_train = np.load(INPUT_LABELS)      # Labels (multi-class)

# Check class distributions before resampling
print("Class distribution before rebalancing ADASYN:")
for i, label in enumerate(['hd', 'cv', 'vo']):
    print(f"{label}: {Counter(y_train[:, i])}")

# Apply ADASYN over-sampling
print("\nApplying ADASYN...")
ada = ADASYN(random_state=42)

# For multi-class problems, ADASYN applies a one-vs-rest strategy.
X_resampled_list = []
y_resampled_list = []

for i, label in enumerate(['hd', 'cv', 'vo']):
    print(f"\nResampling for {label}...")
    # Create one-vs-rest labels for this class
    y_binary = (y_train[:, i] == 1).astype(int)

    # Apply ADASYN
    X_resampled, y_resampled = ada.fit_resample(X_train, y_binary)

    # Append results for this class
    X_resampled_list.append(X_resampled)
    y_resampled_list.append(y_resampled)

# Combine results back into a single dataset
X_resampled = np.concatenate(X_resampled_list, axis=0)
y_resampled = np.column_stack(y_resampled_list)

# Save the resampled data for further use
OUTPUT_EMBEDDINGS_RESAMPLED = "resampled_embeddings_adasyn.npy"
OUTPUT_LABELS_RESAMPLED = "resampled_labels_adasyn.npy"

print("\nSaving resampled data...")
np.save(OUTPUT_EMBEDDINGS_RESAMPLED, X_resampled)
np.save(OUTPUT_LABELS_RESAMPLED, y_resampled)

# Check class distributions after resampling
print("\nClass distribution after rebalancing using ADASYN:")
for i, label in enumerate(['hd', 'cv', 'vo']):
    print(f"{label}: {Counter(y_resampled[:, i])}")

