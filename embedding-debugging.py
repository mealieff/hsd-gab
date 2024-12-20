import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Paths for input and output files
#INPUT_FILE = "ghc_train.tsv"
#OUTPUT_EMBEDDINGS = "new_train_embeddings.npy"  # New name for the embeddings file
#OUTPUT_LABELS = "new_train_labels.npy"  # New name for the labels file
INPUT_FILE = "ghc_test.tsv"
OUTPUT_EMBEDDINGS = "test_embeddings.npy"
OUTPUT_LABELS = "test_labels.npy"


# Load the data via file pathway
print("Loading dataset...")
data = pd.read_csv(INPUT_FILE, sep='\t')

# Check the structure of the data
print("Dataset structure:")
print(data.head())

# Ensure the necessary columns are present
if 'text' not in data.columns or not {'hd', 'cv', 'vo'}.issubset(data.columns):
    raise ValueError("The input file must contain 'text', 'hd', 'cv', and 'vo' columns.")

# Separate text and labels
texts = data['text']
labels = data[['hd', 'cv', 'vo']].values

# Check for missing or empty texts
missing_texts = texts[texts.isna()]
print(f"Number of missing texts: {len(missing_texts)}")
if len(missing_texts) > 0:
    print("Missing texts:", missing_texts)

# Check if all texts are valid (non-empty strings)
empty_texts = texts[texts.str.strip() == '']
print(f"Number of empty texts: {len(empty_texts)}")
if len(empty_texts) > 0:
    print("Empty texts:", empty_texts)

# Check if the number of texts matches the number of labels
print(f"Number of texts: {len(texts)}")
print(f"Number of labels: {labels.shape[0]}")

# Load BERT-base-uncased
print("Loading BERT tokenizer and model...")
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
            print(f"Processed batch {i//batch_size + 1}, batch size: {batch_embeddings.shape[0]}")
    return np.vstack(embeddings)

# Generate embeddings
print("Generating embeddings...")
embeddings = compute_embeddings(texts)

# Check the shape of embeddings
print(f"Shape of embeddings: {embeddings.shape}")
print(f"Shape of labels: {labels.shape}")

# Debugging: Check if the embeddings and labels have the same number of rows
if embeddings.shape[0] != labels.shape[0]:
    print(f"Mismatch between embeddings ({embeddings.shape[0]}) and labels ({labels.shape[0]})")
    
    # Check the number of texts processed
    print(f"Number of texts processed: {len(texts)}")
    print(f"Number of embeddings generated: {embeddings.shape[0]}")
    
    # Check if any labels are missing or if there's any batch issue
    print(f"Number of missing texts in embeddings: {len(texts) - embeddings.shape[0]}")
    
    # Optionally trim embeddings or labels to match the sizes
    embeddings = embeddings[:labels.shape[0]]  # Trim embeddings to match labels
    print(f"Trimmed embeddings shape: {embeddings.shape}")

# Save embeddings and labels
print("Saving embeddings and labels...")
np.save(OUTPUT_EMBEDDINGS, embeddings)
np.save(OUTPUT_LABELS, labels)

# Print new file pathway
print(f"Embeddings saved to {OUTPUT_EMBEDDINGS}")
print(f"Labels saved to {OUTPUT_LABELS}")

