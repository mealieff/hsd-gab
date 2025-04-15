import numpy as np

tag_columns = ["HD", "CV", "VO"]

embeddings = np.load("train_embeddings.npy")
labels = np.load("train_labels.npy")

assert embeddings.shape[0] == labels.shape[0], "Embeddings and labels must have same number of rows."

output = []

for i in range(embeddings.shape[0]):
    emb = embeddings[i]
    label_row = labels[i]
    added = False
    for j, tag in enumerate(label_row):
        if tag == 1:
            output.append([emb, tag_columns[j]])
            added = True
    if not added:
        output.append([emb, "NONE"])

output_array = np.array(output, dtype=object)
np.save("train_embeddings_single_label.npy", output_array)

print(f"Saved {len(output_array)} single-label rows to 'train_embeddings_single_label.npy'.")

