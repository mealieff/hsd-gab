
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, jaccard_score
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
    resampled_data_dir = "results-hsd-gab/resampled_data_binary_multi_11"

    # Add in pathway to baseline
    multiclass_files = [
        ("ADASYN", "multiclass_ADASYN_embeddings.npy", "multiclass_ADASYN_labels.npy"),
        ("CondensedNearestNeighbour", "multiclass_CondensedNearestNeighbour_embeddings.npy", "multiclass_CondensedNearestNeighbour_labels.npy"),
        ("RandomOverSampler", "multiclass_RandomOverSampler_embeddings.npy", "multiclass_RandomOverSampler_labels.npy"),
        ("RandomUnderSampler", "multiclass_RandomUnderSampler_embeddings.npy", "multiclass_RandomUnderSampler_labels.npy"),
        ("SMOTE", "multiclass_SMOTE_embeddings.npy", "multiclass_SMOTE_labels.npy"),
        ("SMOTEENN", "multiclass_SMOTEENN_embeddings.npy", "multiclass_SMOTEENN_labels.npy"),
        ("TomekLinks", "multiclass_TomekLinks_embeddings.npy", "multiclass_TomekLinks_labels.npy")
    ]

    RESULTS_FILE = "results-hsd-gab/svm-multi-metrics-8labels.txt"
    CONFIDENCE_FILE = "results-hsd-gab/svm-multi-confidence-8labels.txt"

    with open(RESULTS_FILE, "w") as f:

        print("Loading test data...")
        f.write("Loading test data...\n")
        test_embeddings = np.load('trash/test_embeddings.npy')
        test_labels = np.load('trash/test_labels.npy')

        test_labels, label_encoder = encode_multilabels(test_labels)

        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for i in thresholds:
            confidence_threshold = i 

        for method_name, embeddings_file, labels_file in multiclass_files:
            print(f"\n=== Processing Resampling Method: {method_name} ===")
            f.write(f"\n=== Processing Resampling Method: {method_name} ===\n")
            # Load training data
            print(f"Loading training data for {method_name}...")
            f.write(f"Loading training data for {method_name}...\n")
            train_embeddings = np.load(os.path.join(resampled_data_dir, embeddings_file))
            train_labels = np.load(os.path.join(resampled_data_dir, labels_file))
            print(f"train_labels shape: {train_labels.shape}")

            #print(f"train_labels shape: {train_labels.shape}")

            # Encode training labels using the same encoder as test labels
            train_labels, _ = encode_multilabels(train_labels)

            print(f"Training initial LinearSVC model for {method_name}...")
            f.write(f"Training initial LinearSVC model for {method_name}...\n")
            initial_model = LinearSVC()
            initial_model.fit(train_embeddings, train_labels)

            for iteration in range(3):  # 3 Iterations of Label Refinement
                print(f"\n--- Iteration {iteration + 1} ---")
                f.write(f"\n--- Iteration {iteration + 1} ---\n")
                print("Predicting labels for unlabeled test data...")
                f.write("Predicting labels for unlabeled test data...\n")

                # Predict labels for unlabeled data
                new_labels, confident_indices = refine_labels(initial_model, test_embeddings, confidence_threshold)

                if len(confident_indices) == 0:
                    print("No high-confidence labels found in this iteration. Stopping early.")
                    f.write("No high-confidence labels found in this iteration. Stopping early.\n")
                    break

                print(f"New high-confidence labels found: {len(confident_indices)}")
                f.write(f"New high-confidence labels found: {len(confident_indices)}\n")
                print("Updating training data with new labels...")
                f.write("Updating training data with new labels...\n")

                # Update training set with newly labeled data
                train_embeddings = np.vstack([train_embeddings, test_embeddings[confident_indices]])
                train_labels = np.hstack([train_labels, new_labels])

                print("Retraining model with updated dataset...")
                f.write("Retraining model with updated dataset...\n")
                initial_model.fit(train_embeddings, train_labels)

            print("\nFinal model evaluation...")
            f.write("\nFinal model evaluation...\n")
            predictions = initial_model.predict(test_embeddings)
            accuracy = initial_model.score(test_embeddings, test_labels)
            report = classification_report(test_labels, predictions, output_dict=True)
            jaccardscore = jaccard_score(test_labels, predictions, average=None)

            macro_avg = report.get('macro avg', {})
            precision = macro_avg.get('precision', None)
            recall = macro_avg.get('recall', None)
            macro_f1 = macro_avg.get('f1-score', None)

            # Get decision function (confidence scores) for test data
            confidence_scores = initial_model.decision_function(test_embeddings)

            with open(CONFIDENCE_FILE, "a") as g:
                #print("\n\n\n\n ==========================================================")
                g.write("\n\n\n\n ==========================================================\n")
                #print(f"\n========= Confidence Scores for {method_name} =========")
                g.write(f"\n========= Confidence Scores for {method_name} =========\n")
                #print("\n ==========================================================")
                g.write("\n ==========================================================\n")
                # Print confidence scores per test sample (optional: can also save to file if needed)
                #print("\nConfidence Scores per Test Sample:")
                g.write("\nConfidence Scores per Test Sample:\n")
                for idx, (prediction, scores) in enumerate(zip(predictions, confidence_scores)):
                    confidence = max(scores)  # This is the confidence for the predicted class
                    #print(f"Sample {idx + 1}: Predicted Label = {prediction}, Confidence = {confidence:.4f}")
                    g.write(f"Sample {idx + 1}: Predicted Label = {prediction}, Confidence = {confidence:.4f}\n")

            mean_confidence = np.mean([max(scores) for scores in confidence_scores])
            print(f"Average Confidence Score for {method_name}: {mean_confidence:.4f}")
            f.write(f"Average Confidence Score for {method_name}: {mean_confidence:.4f}\n")
            print(f"\n=== Final Results for {method_name} ===")
            f.write(f"\n=== Final Results for {method_name} ===\n")
            print(f"Final Accuracy: {accuracy:.4f}")
            f.write(f"Final Accuracy: {accuracy:.4f}\n")
            print(f"Precision (Macro-Averaged): {precision:.4f}")
            f.write(f"Precision (Macro-Averaged): {precision:.4f}\n")
            print(f"Recall (Macro-Averaged): {recall:.4f}")
            f.write(f"Recall (Macro-Averaged): {recall:.4f}\n")
            print(f"F1 Score (Macro-Averaged): {macro_f1:.4f}")
            f.write(f"F1 Score (Macro-Averaged): {macro_f1:.4f}\n")
            print(f"Jaccard Score: {jaccardscore:.4f}")
            f.write(f"Jaccard Score: {jaccardscore:.4f}\n")
            print("=" * 50)
            f.write("=" * 50 + "\n")

