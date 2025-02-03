import os
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN

# Paths for input and output files
INPUT_EMBEDDINGS = "train_embeddings.npy"
INPUT_LABELS = "train_labels.npy"
OUTPUT_DIR = "resampled_data_binary21/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load embeddings and labels
print("Loading embeddings and labels...")
X_train = np.load(INPUT_EMBEDDINGS)
y_train = np.load(INPUT_LABELS)

# Define binary and multiclass label structures
def convert_to_binary(y):
    y_binary = np.ones(len(y)) # initialize as 1 (hate)
    for i in range(len(y)):
        if y[i][0] == 0 and y[i][1] == 0 and y[i][2] == 0: # {}
            y_binary[i] = 0 # non_hate
        if y[i][0] == 0 and y[i][1] == 0 and y[i][2] == 1: # {VO}
            y_binary[i] = 0
    return y_binary

y_binary = convert_to_binary(y_train) # list
print(y_binary)
print(Counter(y_binary))
print(len(y_train))

# Define resampling techniques
resampling_techniques = {
    "RandomUnderSampler": RandomUnderSampler(random_state=42, sampling_strategy=1/3),
    #"CondensedNearestNeighbour": CondensedNearestNeighbour(random_state=42), # can't 3:1
    #"TomekLinks": TomekLinks(), # can't 3:1
    ## both of above weren't taking dict or float as sampling_strategy
    ## --> conclusion, can't do other than 1:1 for both
    "SMOTE": SMOTE(random_state=42, sampling_strategy=1/3),
    "ADASYN": ADASYN(random_state=42, sampling_strategy=1/3),
    "RandomOverSampler": RandomOverSampler(random_state=42, sampling_strategy=1/3),
    "SMOTEENN": SMOTEENN(random_state=42, sampling_strategy=1/3),

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
    X_resampled, y_resampled = resample_data(X_train, y_binary, name, technique)
    np.save(f"{OUTPUT_DIR}binary_{name}_embeddings.npy", X_resampled)
    np.save(f"{OUTPUT_DIR}binary_{name}_labels.npy", y_resampled)

print("\nResampling completed. Resampled files saved to:", OUTPUT_DIR)
