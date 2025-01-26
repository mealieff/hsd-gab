import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import os

def encode_multilabels(labels,encoder=None):
    """
    Encode multi-hot encoded labels into single integer labels and create a mapping.

    Args:
        labels (np.ndarray): Multi-hot encoded labels.
        encoder (LabelEncoder): Optional, existing LabelEncoder instance.

    Returns:
        np.ndarray: Single integer labels.
        LabelEncoder: Encoder to transform between multi-hot and single labels.
    """
    if encoder is None:
        encoder = LabelEncoder()
    # Convert multi-hot encoded labels to string representations
    unique_labels = ["_".join(map(str, row)) for row in labels]
    single_labels = encoder.fit_transform(unique_labels)
    return single_labels, encoder

if __name__ == "__main__":
    # Define the path to the resampled data directory
    resampled_data_dir = 'resampled_data/'

    # List of methods and corresponding file paths
    multiclass_files = [
    ("ADASYN", "resampled_data2_1/multiclass_ADASYN_embeddings.npy", "resampled_data2_1/multiclass_ADASYN_labels.npy"),
    ("CondensedNearestNeighbour", "resampled_data2_1/multiclass_CondensedNearestNeighbour_embeddings.npy", "resampled_data2_1/multiclass_CondensedNearestNeighbour_labels.npy"),
    ("RandomOverSampler", "resampled_data2_1/multiclass_RandomOverSampler_embeddings.npy", "resampled_data2_1/multiclass_RandomOverSampler_labels.npy"),
    ("RandomUnderSampler", "resampled_data2_1/multiclass_RandomUnderSampler_embeddings.npy", "resampled_data2_1/multiclass_RandomUnderSampler_labels.npy"),
    ("SMOTE", "resampled_data2_1/multiclass_SMOTE_embeddings.npy", "resampled_data2_1/multiclass_SMOTE_labels.npy"),
    ("SMOTEENN", "resampled_data2_1/multiclass_SMOTEENN_embeddings.npy", "resampled_data2_1/multiclass_SMOTEENN_labels.npy"),
    ("TomekLinks", "resampled_data2_1/multiclass_TomekLinks_embeddings.npy", "resampled_data2_1/multiclass_TomekLinks_labels.npy")
]

    # Load test data
    print("Loading test data...")
    test_embeddings_file = 'test_embeddings.npy'
    test_labels_file = 'test_labels.npy'
    test_embeddings = np.load(test_embeddings_file)
    test_labels = np.load(test_labels_file)

    # Encode test labels
    test_labels, label_encoder = encode_multilabels(test_labels)

    for method_name, embeddings_file, labels_file in multiclass_files:
        print(f"\n--- Method: {method_name} ---")

        # Load training data
        print("Loading training data...")
        train_embeddings_file = os.path.join(resampled_data_dir, embeddings_file)
        train_labels_file = os.path.join(resampled_data_dir, labels_file)
        train_embeddings = np.load(train_embeddings_file)
        train_labels = np.load(train_labels_file)

        # Encode training labels using the same encoder as test labels
        train_labels, _ = encode_multilabels(train_labels)

        # Train LinearSVC
        lin_clf = LinearSVC()
        lin_clf.fit(train_embeddings, train_labels)
       
        # Predict on test data
        predictions = lin_clf.predict(test_embeddings)
       
        # Evaluate on test data
        accuracy = lin_clf.score(test_embeddings, test_labels)
       
        # Generate classification report for precision, recall, and F1 score
        report = classification_report(test_labels, predictions, output_dict=True)

        # Extract metrics from the classification report
        macro_avg = report.get('macro avg', {})
        precision = macro_avg.get('precision', None)
        recall = macro_avg.get('recall', None)
        macro_f1 = macro_avg.get('f1-score', None)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro-Averaged): {precision:.4f}")
        print(f"Recall (Macro-Averaged): {recall:.4f}")
        print(f"F1 Score (Macro-Averaged): {macro_f1:.4f}")

