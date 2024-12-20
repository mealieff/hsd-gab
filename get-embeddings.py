import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Paths for input and output files
<<<<<<< HEAD
#INPUT_FILE = "ghc_train.tsv"
#OUTPUT_EMBEDDINGS = "train_embeddings.npy"
#OUTPUT_LABELS = "train_labels.npy"

INPUT_FILE = "ghc_test.tsv"
OUTPUT_EMBEDDINGS = "test_embeddings.npy"
OUTPUT_LABELS = "test_labels.npy"
=======
INPUT_FILE = "ghc_train.tsv"
OUTPUT_EMBEDDINGS = "train_embeddings.npy"
OUTPUT_LABELS = "train_labels.npy"
>>>>>>> e891851236288748f229a4f51caed65a18c2d136

# Load the data
print("Loading dataset...")
data = pd.read_csv(INPUT_FILE, sep='\t')

# Ensure the TSV file has the expected structure
if 'text' not in data.columns or not {'hd', 'cv', 'vo'}.issubset(data.columns):
    raise ValueError("The input file must contain 'text', 'hd', 'cv', and 'vo' columns.")

# Separate text and labels
texts = data['text']
labels = data[['hd', 'cv', 'vo']].values

# Load pre-trained BERT tokenizer and model
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to compute embeddings
def compute_embeddings(texts, batch_size=32):
    embeddings = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computations for efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(list(batch_texts), return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            # Use mean pooling of token embeddings
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Generate embeddings
print("Generating embeddings...")
embeddings = compute_embeddings(texts)

# Save embeddings and labels
print("Saving embeddings and labels...")
np.save(OUTPUT_EMBEDDINGS, embeddings)
np.save(OUTPUT_LABELS, labels)

print(f"Embeddings saved to {OUTPUT_EMBEDDINGS}")
print(f"Labels saved to {OUTPUT_LABELS}")

