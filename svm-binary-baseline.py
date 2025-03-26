import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

baseline_data_dir = 'results-hsd-gab'
result_file = os.path.join(baseline_data_dir, "svm-binary-baseline.txt")

# Test data files
test_embeddings_file = os.path.join(baseline_data_dir, "test_embeddings.npy")
test_labels_file = os.path.join(baseline_data_dir, "test_labels.npy")

# Training data files (baseline, no resampling)
train_embeddings_file = os.path.join(baseline_data_dir, "train_embeddings.npy")
train_labels_file = os.path.join(baseline_data_dir, "train_labels.npy")

# Train and evaluate SVM
def train_and_evaluate_svm(train_embeddings, train_labels, test_embeddings, test_labels):
    print("Training SVM for Binary Classification...")

    if len(train_labels.shape) > 1:
        train_labels = np.argmax(train_labels, axis=1)
    if len(test_labels.shape) > 1:
        test_labels = np.argmax(test_labels, axis=1)

    svm_clf = SVC(kernel='linear', random_state=42, decision_function_shape='ovr')
    svm_clf.fit(train_embeddings, train_labels)

    print("Evaluating SVM on test data...")
    predictions = svm_clf.predict(test_embeddings)

    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    return accuracy, precision, recall, macro_f1

# Main function
if __name__ == "__main__":
    print("Loading test data...")
    with open(result_file, "w") as f:
        f.write("Loading test data...\n")

    test_embeddings = np.load(test_embeddings_file)
    test_labels = np.load(test_labels_file)

    print("Loading training data...")
    train_embeddings = np.load(train_embeddings_file)
    train_labels = np.load(train_labels_file)

    accuracy, precision, recall, macro_f1 = train_and_evaluate_svm(
        train_embeddings, train_labels, test_embeddings, test_labels
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro-Averaged): {precision:.4f}")
    print(f"Recall (Macro-Averaged): {recall:.4f}")
    print(f"F1 Score (Macro-Averaged): {macro_f1:.4f}")

    with open(result_file, "a") as f:
        f.write("\nBaseline Model Results:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (Macro-Averaged): {precision:.4f}\n")
        f.write(f"Recall (Macro-Averaged): {recall:.4f}\n")
        f.write(f"F1 Score (Macro-Averaged): {macro_f1:.4f}\n")

    print(f"Results saved in {result_file}")
