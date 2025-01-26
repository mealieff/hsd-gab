import pandas as pd

TRAIN =  "ghc_train.tsv"
TEST = "ghc_test.tsv"


print("Loading dataset...")
test_data = pd.read_csv(TRAIN, sep='\t')
train_data = pd.read_csv(TEST, sep='\t')

# Ensure the TSV file has the expected structure
if 'text' not in data.columns or not {'hd', 'cv', 'vo'}.issubset(data.columns):
    raise ValueError("The input file must contain 'text', 'hd', 'cv', and 'vo' columns.")

# Separate text and labels
train_texts = train_data['text']
test_texts = test_data['text']
train_labels = train_data[['hd', 'cv', 'vo']].values
test_labels = test_data[['hd', 'cv', 'vo']].values

# Extract binary labels per test/train file
# Binary nonhate  = [000] or [001]
# Binary hate = everything else




# Extract multiclass labels per test/train file
# Multiclass labels = all iterations in the range of [000]-[111]

for i in range(0, len(texts)

