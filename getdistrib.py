import pandas as pd
from collections import Counter

TRAIN =  "ghc_train.tsv"
TEST = "ghc_test.tsv"

print("Loading dataset...")
test_data = pd.read_csv(TRAIN, sep='\t')
train_data = pd.read_csv(TEST, sep='\t')

# Separate text and labels
train_texts = train_data['text']
test_texts = test_data['text']
train_labels = train_data[['hd', 'cv', 'vo']].values
test_labels = test_data[['hd', 'cv', 'vo']].values

# Count train data
train_len = len(train_texts)
print("Number of training examples:", train_len)

def count_label_combinations(labels):
    label_counts = Counter(tuple(map(int, row)) for row in labels)
    return dict(label_counts)

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

