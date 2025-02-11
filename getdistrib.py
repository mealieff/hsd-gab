<<<<<<< HEAD
# Extract binary labels per test/train file
# Binary nonhate  = [000] or [001]
# Binary hate = everything else
=======
import pandas as pd
from collections import Counter

TRAIN =  "ghc_train.tsv"
TEST = "ghc_test.tsv"

print("Loading dataset...")
test_data = pd.read_csv(TEST, sep='\t')
train_data = pd.read_csv(TRAIN, sep='\t')

# Separate text and labels
train_texts = train_data['text']
test_texts = test_data['text']
train_labels = train_data[['hd', 'cv', 'vo']].values
test_labels = test_data[['hd', 'cv', 'vo']].values

# Count train data
train_len = len(train_texts)
print("Number of training examples:", train_len)
>>>>>>> ca1894851a121608db2edc71651b0a4bed5063ad

def count_label_combinations(labels):
    label_counts = Counter(tuple(map(int, row)) for row in labels)
    return dict(label_counts)

<<<<<<< HEAD
# Extract multiclass labels per test/train file
# Multiclass labels = all iterations in the range of [000]-[111]

import pandas as pd

TRAIN =  "ghc_train.tsv"
TEST = "ghc_test.tsv"

def load_and_validate_dataset(file_path):
    """Load a dataset and ensure it contains the required columns."""
    data = pd.read_csv(file_path, sep='\t')
    required_columns = {'text', 'hd', 'cv', 'vo'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"The input file {file_path} must contain 'text', 'hd', 'cv', and 'vo' columns.")
    return data

def classify_binary_label(row):
    """Classify row as 'non-hate' or 'hate' based on binary label criteria."""
    return 'non-hate' if row.tolist() in [[0, 0, 0], [0, 0, 1]] else 'hate'

def classify_multiclass_label(row):
    """Classify row as a string representation of multiclass labels."""
    return ''.join(map(str, row.tolist()))

# Ensure the TSV file has the expected structure
if 'text' not in data.columns or not {'hd', 'cv', 'vo'}.issubset(data.columns):
    raise ValueError("The input file must contain 'text', 'hd', 'cv', and 'vo' columns.")

# Count specific label combinations
def count_label_combinations(data, dataset_name):
    print(f"\n{dataset_name} Specific Multiclass Label Counts:")
    label_map = {
        '{}': [0, 0, 0],
        '{HD}': [1, 0, 0],
        '{CV}': [0, 1, 0],
        '{VO}': [0, 0, 1],
        '{HD, CV}': [1, 1, 0],
        '{HD, VO}': [1, 0, 1],
        '{CV, VO}': [0, 1, 1],
        '{HD, CV, VO}': [1, 1, 1]
    }
    for label, combination in label_map.items():
        count = (data[['hd', 'cv', 'vo']].values.tolist().count(combination))
        print(f"{label}: {count}")


def print_label_distribution(data, dataset_name):
    """Print the distribution of binary and multiclass labels in the dataset."""
    print(f"\n{dataset_name} Label Distributions:")
    print("Binary Labels:")
    print(data['binary_label'].value_counts())
    print("\nMulticlass Labels:")
    print(data['multiclass_label'].value_counts())
    count_label_combinations(data, dataset_name)

def main():
    train_file = "ghc_train.tsv"
    test_file = "ghc_test.tsv"

    print("Loading datasets...")

    # Load and validate datasets
    train_data = load_and_validate_dataset(train_file)
    test_data = load_and_validate_dataset(test_file)

    # Apply classification
    train_data['binary_label'] = train_data[['hd', 'cv', 'vo']].apply(classify_binary_label, axis=1)
    test_data['binary_label'] = test_data[['hd', 'cv', 'vo']].apply(classify_binary_label, axis=1)
    train_data['multiclass_label'] = train_data[['hd', 'cv', 'vo']].apply(classify_multiclass_label, axis=1)
    test_data['multiclass_label'] = test_data[['hd', 'cv', 'vo']].apply(classify_multiclass_label, axis=1)

    # Print distributions
    print_label_distribution(train_data, "Training Set")
    print_label_distribution(test_data, "Test Set")

if __name__ == "__main__":
    main()


=======
# Count occurrences
label_counts = count_label_combinations(train_labels)

non_hate = 0
hate = 0

for label, count in sorted(label_counts.items()):
    # Print results
    print(f"{label}: {count}")
    if label == (0, 0, 0):
        non_hate += count
    if label == (0, 0, 1):
        non_hate += count

hate = train_len - non_hate

print(f"Non-hate: {non_hate}")
print(f"Hate: {hate}")
>>>>>>> ca1894851a121608db2edc71651b0a4bed5063ad

