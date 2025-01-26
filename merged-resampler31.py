#Note: this script does not include Tomeklinks for 3:1 resampling ratio as it cannot take sampling_strategy as an attribute

import os
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN

# Paths for input and output files
INPUT_EMBEDDINGS = "train_embeddings.npy"
INPUT_LABELS = "train_labels.npy"
OUTPUT_DIR = "resampled_data3_1/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load embeddings and labels
print("Loading embeddings and labels...")
X_train = np.load(INPUT_EMBEDDINGS)
y_train = np.load(INPUT_LABELS)

# Define binary and multiclass label structures
def convert_to_binary(y):
    return np.any(y > 0, axis=1).astype(int)

def convert_to_multiclass(y):
    return np.array(["{}{}{}".format(*row) for row in y])

y_binary = convert_to_binary(y_train)
y_multiclass = convert_to_multiclass(y_train)

# Calculate the smallest class size for binary and multiclass
min_class_size_binary = Counter(y_binary).most_common()[-1][1]  # Size of the smallest binary class
min_class_size_multiclass = Counter(y_multiclass).most_common()[-1][1]  # Size of the smallest multiclass class

# Define 3:1 sampling strategy
def create_sampling_strategy(counter, ratio=3):
    """Creates a sampling strategy dictionary for a given ratio."""
    min_class_size = min(counter.values())
    return {key: min_class_size * ratio if count > min_class_size else min_class_size for key, count in counter.items()}

binary_sampling_strategy = create_sampling_strategy(Counter(y_binary), ratio=3)
multiclass_sampling_strategy = create_sampling_strategy(Counter(y_multiclass), ratio=3)

# Define resampling techniques
resampling_techniques = {
    "RandomUnderSampler": RandomUnderSampler(random_state=42, sampling_strategy=binary_sampling_strategy),
    "SMOTE": SMOTE(random_state=42, sampling_strategy=0.75),  # 3:1 for binary (0.75 minority:majority)
    "ADASYN": ADASYN(random_state=42, sampling_strategy=0.75),
    "RandomOverSampler": RandomOverSampler(random_state=42, sampling_strategy=0.75),
    "SMOTEENN": SMOTEENN(random_state=42),
}

# Resampling function
def resample_data(X, y, technique_name, technique):
    print(f"\nApplying {technique_name}...")
    try:
        X_resampled, y_resampled = technique.fit_resample(X, y)
        print(f"Resampled distribution for {technique_name}: {Counter(y_resampled)}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Error applying {technique_name}: {e}")
        return X, y

# Perform resampling for binary labels
print("\nProcessing binary labels...")
for name, technique in resampling_techniques.items():
    print(f"\nProcessing {name} for binary labels...")
    if isinstance(technique, RandomUnderSampler):
        technique.set_params(sampling_strategy=binary_sampling_strategy)
    X_resampled, y_resampled = resample_data(X_train, y_binary, name, technique)
    np.save(f"{OUTPUT_DIR}binary_{name}_embeddings.npy", X_resampled)
    np.save(f"{OUTPUT_DIR}binary_{name}_labels.npy", y_resampled)

# Perform resampling for multiclass labels
print("\nProcessing multiclass labels...")
for name, technique in resampling_techniques.items():
    print(f"\nProcessing {name} for multiclass labels...")
    if isinstance(technique, RandomUnderSampler):
        technique.set_params(sampling_strategy=multiclass_sampling_strategy)
    X_resampled, y_resampled = resample_data(X_train, y_multiclass, name, technique)
    np.save(f"{OUTPUT_DIR}multiclass_{name}_embeddings.npy", X_resampled)
    np.save(f"{OUTPUT_DIR}multiclass_{name}_labels.npy", y_resampled)

print("\nResampling completed. Resampled files saved to:", OUTPUT_DIR)

