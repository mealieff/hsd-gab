# this is the script of 'RandomOverSampler' : random-oversampler.py
# script heavily adapted from Imbalanced-learn (imported as imblearn) is an open source, MIT-licensed library relying on scikit-learn

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Paths for input files
INPUT_EMBEDDINGS = "train_embeddings.npy"
INPUT_LABELS = "train_labels.npy"

# Load embeddings and labels
print("Loading embeddings and labels...")
X_train = np.load(INPUT_EMBEDDINGS)  # BERT embeddings
y_train = np.load(INPUT_LABELS)      # Labels (multi-class)

# Check class distributions before resampling
print("Class distribution before rebalancing:")
for i, label in enumerate(['hd', 'cv', 'vo']):
    print(f"{label}: {Counter(y_train[:, i])}")

# Apply Random Over Sampler (ROS)
print("Applying Random Over Sampler...")
ros = RandomOverSampler(random_state=42)

# ROS requires labels to be single-dimensional. To handle multi-label, process each column separately.
X_resampled_list = []
y_resampled_list = []

for i, label in enumerate(['hd', 'cv', 'vo']):
    print(f"\nResampling for {label}...")
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train[:, i])
    X_resampled_list.append(X_resampled)
    y_resampled_list.append(y_resampled)

# Combine results back into a single dataset
X_resampled = np.concatenate(X_resampled_list, axis=0)
y_resampled = np.column_stack(y_resampled_list)

# Save the resampled data for further use
OUTPUT_EMBEDDINGS_RESAMPLED = "resampled_embeddings.npy"
OUTPUT_LABELS_RESAMPLED = "resampled_labels.npy"

print("Saving resampled data...")
np.save(OUTPUT_EMBEDDINGS_RESAMPLED, X_resampled)
np.save(OUTPUT_LABELS_RESAMPLED, y_resampled)

# Check class distributions after resampling
print("\nClass distribution after rebalancing:")
for i, label in enumerate(['hd', 'cv', 'vo']):
    print(f"{label}: {Counter(y_resampled[:, i])}")

