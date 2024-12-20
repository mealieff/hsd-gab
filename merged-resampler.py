import numpy as np
import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN

# Paths for input and output files
INPUT_EMBEDDINGS = "train_embeddings.npy"
INPUT_LABELS = "train_labels.npy"
OUTPUT_DIR = "resampled_data/"

# Load embeddings and labels
print("Loading embeddings and labels...")
X_train = np.load(INPUT_EMBEDDINGS)
y_train = np.load(INPUT_LABELS)

# Define label structure for binary classification
# Binary: Non-hate ({},{VO}) vs Hate ({HD},{CV},{HD,CV},{HD,VO},{CV,VO},{HD,CV,VO})
def convert_to_binary(y):
    return np.any(y > 0, axis=1).astype(int)

# Define label structure for multiclass classification
def convert_to_multiclass(y):
    return np.array(["{}{}{}".format(*row) for row in y])

# Define binary and multiclass labels
y_binary = convert_to_binary(y_train)
y_multiclass = convert_to_multiclass(y_train)

# All resampling techniques can be called simultaneously, random states 42
resampling_techniques = {
    "RandomUnderSampler": RandomUnderSampler(random_state=42),
    "CondensedNearestNeighbour": CondensedNearestNeighbour(random_state=42),
    "TomekLinks": TomekLinks(),
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "RandomOverSampler": RandomOverSampler(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42),
}

# Resampling function
def resample_data(X, y, technique_name, technique):
    print(f"\nApplying {technique_name}...")
    X_resampled, y_resampled = technique.fit_resample(X, y)
    print(f"Resampled distribution for {technique_name}: {Counter(y_resampled)}")
    return X_resampled, y_resampled

# Perform resampling for binary labels
print("\nProcessing binary labels...")
for name, technique in resampling_techniques.items():
    X_resampled, y_resampled = resample_data(X_train, y_binary, name, technique)
    np.save(f"{OUTPUT_DIR}binary_{name}_embeddings.npy", X_resampled)
    np.save(f"{OUTPUT_DIR}binary_{name}_labels.npy", y_resampled)

# Perform resampling for multiclass labels
print("\nProcessing multiclass labels...")
for name, technique in resampling_techniques.items():
    X_resampled, y_resampled = resample_data(X_train, y_multiclass, name, technique)
    np.save(f"{OUTPUT_DIR}multiclass_{name}_embeddings.npy", X_resampled)
    np.save(f"{OUTPUT_DIR}multiclass_{name}_labels.npy", y_resampled)

print("\nResampling completed. Resampled files saved to:", OUTPUT_DIR)

