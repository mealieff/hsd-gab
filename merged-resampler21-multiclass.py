import os
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN

# Paths for input and output files
INPUT_EMBEDDINGS = "train_embeddings.npy"
INPUT_LABELS = "train_labels.npy"
OUTPUT_DIR = "resampled_data_multiclass21/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load embeddings and labels
print("Loading embeddings and labels...")
X_train = np.load(INPUT_EMBEDDINGS)
y_train = np.load(INPUT_LABELS)


def convert_to_multiclass(y):
    return np.array(["{}{}{}".format(*row) for row in y])


print(len(y_train))
y_multiclass = convert_to_multiclass(y_train)

undersampling_dict = {'000': 5340, '001': 729, '010': 56, '011': 28, '100': 1211, '101': 599, '110': 24, '111': 23}
ratio_2_1 = int(9683/2670)
oversampling_dict = {'000': 19366, '001': 729*ratio_2_1, '010': 56*ratio_2_1, '011': 28*ratio_2_1, '100': 1211*ratio_2_1, '101': 599*ratio_2_1, '110': 24*ratio_2_1, '111': 23*ratio_2_1}

# Define resampling techniques
resampling_techniques = {
    "RandomUnderSampler": RandomUnderSampler(random_state=42, sampling_strategy=undersampling_dict),
    #"CondensedNearestNeighbour": CondensedNearestNeighbour(random_state=42), # can't 2:1
    #"TomekLinks": TomekLinks(), # can't 2:1
    ## both of above weren't taking dict or float as sampling_strategy
    ## --> conclusion, can't do other than 1:1 for both
    "SMOTE": SMOTE(random_state=42, sampling_strategy=oversampling_dict),
    "ADASYN": ADASYN(random_state=42, sampling_strategy=oversampling_dict),
    "RandomOverSampler": RandomOverSampler(random_state=42, sampling_strategy=oversampling_dict),
    "SMOTEENN": SMOTEENN(random_state=42, sampling_strategy=oversampling_dict),
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



# Perform resampling for multiclass labels
print("\nProcessing multiclass labels...")
for name, technique in resampling_techniques.items():
    X_resampled, y_resampled = resample_data(X_train, y_multiclass, name, technique)
    np.save(f"{OUTPUT_DIR}multiclass_{name}_embeddings.npy", X_resampled)
    np.save(f"{OUTPUT_DIR}multiclass_{name}_labels.npy", y_resampled)

print("\nResampling completed. Resampled files saved to:", OUTPUT_DIR)
