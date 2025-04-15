import numpy as np

tag_columns = ["HD", "CV", "VO"]

embeddings = np.load("train_embeddings.npy")
labels = np.load("train_labels.npy")

assert embeddings.shape[0] == labels.shape[0], "Embeddings and labels must have same number of rows."

output = []
num_duplicates = 0
num_none = 0

for i in range(embeddings.shape[0]):
    emb = embeddings[i]
    label_row = labels[i]

    tags_added = 0
    for j, tag in enumerate(label_row):
        if int(tag) == 1:  # Handle float or int labels
            output.append([emb, tag_columns[j]])
            tags_added += 1

    if tags_added > 1:
        num_duplicates += tags_added - 1  # count extras
    if tags_added == 0:
        output.append([emb, "NONE"])
        num_none += 1

output_array = np.array(output, dtype=object)
np.save("train_embeddings_single_label.npy", output_array)

print(f"Saved {len(output_array)} single-label rows to 'train_embeddings_single_label.npy'.")
print(f"Added {num_duplicates} duplicated rows due to multi-label entries.")
print(f"Found {num_none} 'NONE' label rows.")

