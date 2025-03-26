import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os


def encode_multilabels(labels, encoder=None):
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
    unique_labels = ["_".join(map(str, row)) for row in labels]
    single_labels = encoder.fit_transform(unique_labels)
    return single_labels, encoder


if __name__ == "__main__":
    # Define paths
    baseline_data_dir = 'results-hsd-gab'
    output_file = os.path.join(baseline_data_dir, 'svm-multi-baseline.txt')

    # Load baseline training data
    print("Loading training data...")
    train_embeddings_file = os.path.join(baseline_data_dir, 'train_embeddings.npy')
    train_labels_file = os.path.join(baseline_data_dir, 'train_labels.npy')

    train_embeddings = np.load(train_embeddings_file)
    train_labels = np.load(train_labels_file)

    # Load test data
    print("Loading test data...")
    test_embeddings_file = os.path.join(baseline_data_dir, 'test_embeddings.npy')
    test_labels_file = os.path.join(baseline_data_dir, 'test_labels.npy')

    test_embeddings = np.load(test_embeddings_file)
    test_labels = np.load(test_labels_file)

    # Encode labels
    train_labels, label_encoder = encode_multilabels(train_labels)
    test_labels, _ = encode_multilabels(test_labels, encoder=label_encoder)

    # Train LinearSVC
    lin_clf = LinearSVC()
    lin_clf.fit(train_embeddings, train_labels)

    # Predict on test data
    predictions = lin_clf.predict(test_embeddings)

    # Evaluate model
    accuracy = lin_clf.score(test_embeddings, test_labels)
    report = classification_report(test_labels, predictions, output_dict=True)

    macro_avg = report.get('macro avg', {})
    precision = macro_avg.get('precision', None)
    recall = macro_avg.get('recall', None)
    macro_f1 = macro_avg.get('f1-score', None)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro-Averaged): {precision:.4f}")
    print(f"Recall (Macro-Averaged): {recall:.4f}")
    print(f"F1 Score (Macro-Averaged): {macro_f1:.4f}")

    # Save results
    with open(output_file, 'w') as f:
        f.write("Baseline Model Results:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (Macro-Averaged): {precision:.4f}\n")
        f.write(f"Recall (Macro-Averaged): {recall:.4f}\n")
        f.write(f"F1 Score (Macro-Averaged): {macro_f1:.4f}\n")

    print(f"Results saved to {output_file}")
