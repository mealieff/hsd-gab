import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

# List of binary data files
binary_files = [
    ("binary_ADASYN", "resampled_data/binary_ADASYN_embeddings.npy", "resampled_data/binary_ADASYN_labels.npy"),
    ("binary_CondensedNearestNeighbour", "resampled_data/binary_CondensedNearestNeighbour_embeddings.npy", "resampled_data/binary_CondensedNearestNeighbour_labels.npy"),
    ("binary_RandomOverSampler", "resampled_data/binary_RandomOverSampler_embeddings.npy", "resampled_data/binary_RandomOverSampler_labels.npy"),
    ("binary_RandomUnderSampler", "resampled_data/binary_RandomUnderSampler_embeddings.npy", "resampled_data/binary_RandomUnderSampler_labels.npy"),
    ("binary_SMOTE", "resampled_data/binary_SMOTE_embeddings.npy", "resampled_data/binary_SMOTE_labels.npy"),
    ("binary_SMOTEENN", "resampled_data/binary_SMOTEENN_embeddings.npy", "resampled_data/binary_SMOTEENN_labels.npy"),
    ("binary_TomekLinks", "resampled_data/binary_TomekLinks_embeddings.npy", "resampled_data/binary_TomekLinks_labels.npy")
]

# Test data files
test_embeddings_file = "test_embeddings.npy"
test_labels_file = "test_labels.npy"

# Train and evaluate SVM
def train_and_evaluate_svm(train_embeddings, train_labels, test_embeddings, test_labels):
    print("Training SVM for Binary Classification...")
    # Ensure one-hot encoded labels are converted to integers
    if len(train_labels.shape) > 1:
        train_labels = np.argmax(train_labels, axis=1)
    if len(test_labels.shape) > 1:
        test_labels = np.argmax(test_labels, axis=1)

    svm_clf = SVC(kernel='linear', random_state=42, decision_function_shape='ovr')  # One-vs-Rest approach

    # Train the model
    svm_clf.fit(train_embeddings, train_labels)

    # Evaluate on test data
    print("Evaluating SVM on test data...")
    predictions = svm_clf.predict(test_embeddings)

    # Generate accuracy and classification report
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    return accuracy, precision, recall, macro_f1

# Main function
if __name__ == "__main__":
    # Load test data (this will be the same for all methods)
    print("Loading test data...")
    test_embeddings = np.load(test_embeddings_file)
    test_labels = np.load(test_labels_file)

    for method_name, embeddings_file, labels_file in binary_files:
        print(f"\n--- Method: {method_name} ---")

        # Load training data
        print("Loading training data...")
        train_embeddings = np.load(embeddings_file)
        train_labels = np.load(labels_file)

        # Train and evaluate SVM
        accuracy, precision, recall, macro_f1 = train_and_evaluate_svm(
            train_embeddings, train_labels, test_embeddings, test_labels
        )

        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro-Averaged): {precision:.4f}")
        print(f"Recall (Macro-Averaged): {recall:.4f}")
        print(f"F1 Score (Macro-Averaged): {macro_f1:.4f}")

