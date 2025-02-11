import numpy as np
import pandas as pd
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from scipy.special import softmax  # For normalizing decision scores

# Define the resampled data directory
resampled_data_dir = 'resampled_data/'

# Define the list of methods and corresponding file paths
multiclass_files = [
    ("ADASYN", "multiclass_ADASYN_embeddings.npy", "multiclass_ADASYN_labels.npy"),
    ("CondensedNearestNeighbour", "multiclass_CondensedNearestNeighbour_embeddings.npy", "multiclass_CondensedNearestNeighbour_labels.npy"),
    ("RandomOverSampler", "multiclass_RandomOverSampler_embeddings.npy", "multiclass_RandomOverSampler_labels.npy"),
    ("RandomUnderSampler", "multiclass_RandomUnderSampler_embeddings.npy", "multiclass_RandomUnderSampler_labels.npy"),
    ("SMOTE", "multiclass_SMOTE_embeddings.npy", "multiclass_SMOTE_labels.npy"),
    ("SMOTEENN", "multiclass_SMOTEENN_embeddings.npy", "multiclass_SMOTEENN_labels.npy"),
    ("TomekLinks", "multiclass_TomekLinks_embeddings.npy", "multiclass_TomekLinks_labels.npy"),
]

# Define label mappings with linguistic feature tags
label_mapping = {
    "0_0_0": "{}",
    "0_0_1": "VO",
    "0_1_0": "CV",
    "0_1_1": "CV,VO",
    "1_0_0": "HD",
    "1_0_1": "HD,VO",
    "1_1_0": "HD,CV",
    "1_1_1": "HD,CV,VO"
}


# Function to encode multi-hot labels into single integer labels
def encode_multilabels(labels, encoder=None):
    if encoder is None:
        encoder = LabelEncoder()
    unique_labels = ["_".join(map(str, row)) for row in labels]
    single_labels = encoder.fit_transform(unique_labels)
    return single_labels, encoder


# Load test data
print("Loading test data...")
test_embeddings_file = 'test_embeddings.npy'
test_labels_file = 'test_labels.npy'

test_embeddings = np.load(test_embeddings_file)
test_labels = np.load(test_labels_file)

# Encode test labels
test_labels, label_encoder = encode_multilabels(test_labels)

# Store results for each method
all_results = []

for method_name, embeddings_file, labels_file in multiclass_files:
    print(f"\n=== Method: {method_name} ===")

    # Load training data
    train_embeddings_file = os.path.join(resampled_data_dir, embeddings_file)
    train_labels_file = os.path.join(resampled_data_dir, labels_file)

    print("Loading training data...")
    train_embeddings = np.load(train_embeddings_file)
    train_labels = np.load(train_labels_file)

    # Encode training labels using the same encoder as test labels
    train_labels, _ = encode_multilabels(train_labels)

    # Train LinearSVC
    lin_clf = LinearSVC()
    lin_clf.fit(train_embeddings, train_labels)

    # Predict on test data
    predictions = lin_clf.predict(test_embeddings)

    # Compute decision function (confidence scores)
    decision_scores = lin_clf.decision_function(test_embeddings)

    # Normalize decision scores using softmax
    confidence_scores = softmax(decision_scores, axis=1)  # Converts to probabilities-like values

    # Compute overall confidence score for the method
    overall_confidence = np.mean(confidence_scores)

    # Generate classification report for F1-score and precision
    report = classification_report(test_labels, predictions, output_dict=True, zero_division=0, target_names=label_encoder.classes_)

    method_results = []

    print("\nF1-Score per Class:")
    for class_label, feature_tag in label_mapping.items():
        f1_class = report.get(str(class_label), {}).get('f1-score', "N/A")
        print(f"{class_label} ({feature_tag}): {f1_class}")
        method_results.append({
            'Method': method_name,
            'Class': class_label,
            'Feature_Tag': feature_tag,
            'F1-Score': f1_class
        })

    print("\nPrecision per Class:")
    for class_label, feature_tag in label_mapping.items():
        precision_class = report.get(str(class_label), {}).get('precision', "N/A")
        print(f"{class_label} ({feature_tag}): {precision_class}")
        method_results.append({
            'Method': method_name,
            'Class': class_label,
            'Feature_Tag': feature_tag,
            'Precision': precision_class
        })

    print("\nConfidence Score per Class:")
    for idx, class_label in enumerate(label_encoder.classes_):
        avg_confidence = np.mean(confidence_scores[:, idx]) if idx < confidence_scores.shape[1] else "N/A"
        print(f"{class_label} ({label_mapping.get(class_label, 'Unknown')}): {avg_confidence}")
        method_results.append({
            'Method': method_name,
            'Class': class_label,
            'Feature_Tag': label_mapping.get(class_label, "Unknown"),
            'Confidence Score': avg_confidence
        })

    # Store overall confidence score per method
    print(f"\nOverall Confidence Score for {method_name}: {overall_confidence:.4f}")
    method_results.append({
        'Method': method_name,
        'Class': "Overall",
        'Feature_Tag': "N/A",
        'Confidence Score': overall_confidence
    })

    # Store all results
    all_results.extend(method_results)

# Convert results into a DataFrame
df_results = pd.DataFrame(all_results)

# Define output file path
output_file = os.path.join(resampled_data_dir, "classification_results-multi11.csv")

# Save DataFrame to CSV
df_results.to_csv(output_file, index=False)

print(f"\nResults saved successfully to: {output_file}")
