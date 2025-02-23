# I went crazy with the print statements feel free to remove 

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os

def encode_multilabels(labels, encoder=None):
    """
    Encode multi-hot encoded labels into single integer labels and create a mapping.
    """
    if encoder is None:
        encoder = LabelEncoder()
    
    unique_labels = ["_".join(map(str, row)) for row in labels]
    single_labels = encoder.fit_transform(unique_labels)
    return single_labels, encoder

def refine_labels(model, X_unlabeled, confidence_threshold):
    """
    Uses a trained model to predict and refine labels based on confidence scores.
    
    Args:
        model: Trained classifier.
        X_unlabeled (np.ndarray): Unlabeled feature embeddings.
        confidence_threshold (float): Minimum confidence required to assign a label.
    
    Returns:
        np.ndarray: Newly labeled data.
        np.ndarray: Indices of data points that were assigned labels.
    """
    decision_function = model.decision_function(X_unlabeled)  # Get confidence scores
    max_confidence = np.max(np.abs(decision_function), axis=1)  # Get highest confidence per sample
    new_labels = np.argmax(decision_function, axis=1)  # Predicted labels

    # Filter by confidence threshold
    confident_indices = max_confidence >= confidence_threshold
    return new_labels[confident_indices], np.where(confident_indices)[0]

if __name__ == "__main__":
    resampled_data_dir = ""

    # Add in pathway to baseline
    multiclass_files = [
        ("ADASYN", "resampled_data/multiclass_ADASYN_embeddings.npy", "resampled_data/multiclass_ADASYN_labels.npy"),
        ("CondensedNearestNeighbour", "resampled_data/multiclass_CondensedNearestNeighbour_embeddings.npy", "resampled_data/multiclass_CondensedNearestNeighbour_labels.npy"),
        ("RandomOverSampler", "resampled_data/multiclass_RandomOverSampler_embeddings.npy", "resampled_data/multiclass_RandomOverSampler_labels.npy"),
        ("RandomUnderSampler", "resampled_data/multiclass_RandomUnderSampler_embeddings.npy", "resampled_data/multiclass_RandomUnderSampler_labels.npy"),
        ("SMOTE", "resampled_data/multiclass_SMOTE_embeddings.npy", "resampled_data/multiclass_SMOTE_labels.npy"),
        ("SMOTEENN", "resampled_data/multiclass_SMOTEENN_embeddings.npy", "resampled_data/multiclass_SMOTEENN_labels.npy"),
        ("TomekLinks", "resampled_data/multiclass_TomekLinks_embeddings.npy", "resampled_data/multiclass_TomekLinks_labels.npy")
    ]

    print("Loading test data...")
    test_embeddings = np.load('test_embeddings.npy')
    test_labels = np.load('test_labels.npy')

    test_labels, label_encoder = encode_multilabels(test_labels)

    confidence_threshold = 0.7  # Confidence threshold for iterative labeling, might need to adjust given results

    for method_name, embeddings_file, labels_file in multiclass_files:
        print(f"\n=== Processing Resampling Method: {method_name} ===")

        # Load training data
        print(f"Loading training data for {method_name}...")
        train_embeddings = np.load(os.path.join(resampled_data_dir, embeddings_file))
        train_labels = np.load(os.path.join(resampled_data_dir, labels_file))

        # Encode training labels using the same encoder as test labels
        train_labels, _ = encode_multilabels(train_labels)

        print(f"Training initial LinearSVC model for {method_name}...")
        initial_model = LinearSVC()
        initial_model.fit(train_embeddings, train_labels)

        for iteration in range(3):  # 3 Iterations of Label Refinement
            print(f"\n--- Iteration {iteration + 1} ---")
            print("Predicting labels for unlabeled test data...")

            # Predict labels for unlabeled data
            new_labels, confident_indices = refine_labels(initial_model, test_embeddings, confidence_threshold)

            if len(confident_indices) == 0:
                print("No high-confidence labels found in this iteration. Stopping early.")
                break

            print(f"New high-confidence labels found: {len(confident_indices)}")
            print("Updating training data with new labels...")

            # Update training set with newly labeled data
            train_embeddings = np.vstack([train_embeddings, test_embeddings[confident_indices]])
            train_labels = np.hstack([train_labels, new_labels])

            print("Retraining model with updated dataset...")
            initial_model.fit(train_embeddings, train_labels)

        print("\nFinal model evaluation...")
        predictions = initial_model.predict(test_embeddings)
        accuracy = initial_model.score(test_embeddings, test_labels)
        report = classification_report(test_labels, predictions, output_dict=True)

        macro_avg = report.get('macro avg', {})
        precision = macro_avg.get('precision', None)
        recall = macro_avg.get('recall', None)
        macro_f1 = macro_avg.get('f1-score', None)

        print(f"\n=== Final Results for {method_name} ===")
        print(f"Final Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro-Averaged): {precision:.4f}")
        print(f"Recall (Macro-Averaged): {recall:.4f}")
        print(f"F1 Score (Macro-Averaged): {macro_f1:.4f}")
        print("=" * 50)

