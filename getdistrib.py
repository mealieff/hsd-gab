import pandas as pd

TRAIN =  "ghc_train.tsv"
TEST = "ghc_test.tsv"


print("Loading dataset...")
data = pd.read_csv(INPUT_FILE, sep='\t')

# Ensure the TSV file has the expected structure
if 'text' not in data.columns or not {'hd', 'cv', 'vo'}.issubset(data.columns):
    raise ValueError("The input file must contain 'text', 'hd', 'cv', and 'vo' columns.")

# Separate text and labels
texts = data['text']
binary_labels = data[['hd', 'cv', 'vo']].values

for i in range(0, len(texts)

