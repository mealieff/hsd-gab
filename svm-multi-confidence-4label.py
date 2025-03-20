import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import os


def refine_labels(model, X_unlabeled, confidence_threshold):
    """
    Predicts and refines labels for a single binary label (HD, CV, VO, no_label).

    Args:
        model: Trained LinearSVC.
        X_unlabeled (np.ndarray): Unlabeled feature embeddings.
        confidence_threshold (float): Minimum confidence required to assign a label.

    Returns:
        np.ndarray: Newly predicted binary labels (0 or 1).
        np.ndarray: Indices of data points with confident predictions.
    """
    decision_function = model.decision_function(X_unlabeled)
    max_confidence = np.abs(decision_function)  # Binary case, confidence is abs distance to hyperplane
    new_labels = (decision_function >= 0).astype(int)

    confident_indices = np.where(max_confidence >= confidence_threshold)[0]

    if len(confident_indices) == 0:  # Avoid empty array issues
        return np.array([]), confident_indices

    return new_labels[confident_indices], confident_indices


if __name__ == "__main__":
    resampled_data_dir = "results-hsd-gab/resampled_data_binary_multi_11"  # Adjust path if needed

    multiclass_files = [
        ("ADASYN", "multiclass_ADASYN_embeddings.npy", "multiclass_ADASYN_labels.npy"),
        ("CondensedNearestNeighbour", "multiclass_CondensedNearestNeighbour_embeddings.npy", "multiclass_CondensedNearestNeighbour_labels.npy"),
        ("RandomOverSampler", "multiclass_RandomOverSampler_embeddings.npy", "multiclass_RandomOverSampler_labels.npy"),
        ("RandomUnderSampler", "multiclass_RandomUnderSampler_embeddings.npy",
         "multiclass_RandomUnderSampler_labels.npy"),
        ("SMOTE", "multiclass_SMOTE_embeddings.npy", "multiclass_SMOTE_labels.npy"),
        ("SMOTEENN", "multiclass_SMOTEENN_embeddings.npy", "multiclass_SMOTEENN_labels.npy"),
        ("TomekLinks", "multiclass_TomekLinks_embeddings.npy", "multiclass_TomekLinks_labels.npy")

    ]

    RESULTS_FILE = "results-hsd-gab/svm-multi-metrics-4labels.txt"
    CONFIDENCE_FILE = "results-hsd-gab/svm-multi-confidence-4labels.txt"

    with open(RESULTS_FILE, "w") as f:
        print("Loading test data...")
        f.write("Loading test data...\n")
        test_embeddings = np.load('trash/test_embeddings.npy')
        test_labels = np.load('trash/test_labels.npy')

        print(f"test_labels shape: {test_labels.shape}")

        # Extract binary labels (HD, CV, VO, no_label)
        test_labels_HD = test_labels[:, 0]
        test_labels_CV = test_labels[:, 1]
        test_labels_VO = test_labels[:, 2]
        test_labels_no_label = (test_labels.sum(axis=1) == 0).astype(int)

        confidence_threshold = 0.7

        for method_name, embeddings_file, labels_file in multiclass_files:
            print(f"\n=== Processing Resampling Method: {method_name} ===")
            f.write(f"\n=== Processing Resampling Method: {method_name} ===\n")

            train_embeddings = np.load(os.path.join(resampled_data_dir, embeddings_file))
            train_labels = np.load(os.path.join(resampled_data_dir, labels_file))

            print(f"train_labels shape: {train_labels.shape}")

            # Convert string-based labels to binary format
            lst_of_lsts_train_labels = np.array([[int(char) for char in label] for label in train_labels])

            train_labels_HD = lst_of_lsts_train_labels[:, 0]
            train_labels_CV = lst_of_lsts_train_labels[:, 1]
            train_labels_VO = lst_of_lsts_train_labels[:, 2]
            train_labels_no_label = (lst_of_lsts_train_labels.sum(axis=1) == 0).astype(int)

            print(f"Training initial LinearSVC models for {method_name}...")
            f.write(f"Training initial LinearSVC models for {method_name}...\n")

            model_HD = LinearSVC(max_iter=5000).fit(train_embeddings, train_labels_HD)
            model_CV = LinearSVC(max_iter=5000).fit(train_embeddings, train_labels_CV)
            model_VO = LinearSVC(max_iter=5000).fit(train_embeddings, train_labels_VO)
            model_no_label = LinearSVC(max_iter=5000).fit(train_embeddings, train_labels_no_label)

            for iteration in range(3):
                print(f"\n--- Iteration {iteration + 1} ---")
                f.write(f"\n--- Iteration {iteration + 1} ---\n")

                # Step 1: Get confident labels for each category
                new_labels_HD, confident_indices_HD = refine_labels(model_HD, test_embeddings, confidence_threshold)
                new_labels_CV, confident_indices_CV = refine_labels(model_CV, test_embeddings, confidence_threshold)
                new_labels_VO, confident_indices_VO = refine_labels(model_VO, test_embeddings, confidence_threshold)
                new_labels_no_label, confident_indices_no_label = refine_labels(model_no_label, test_embeddings,
                                                                                confidence_threshold)

                # ✅ Insert Fix: Ensure all confident indices are collected properly
                all_confident_indices = np.concatenate([
                    confident_indices_HD, confident_indices_CV, confident_indices_VO, confident_indices_no_label
                ])
                all_confident_indices = np.unique(all_confident_indices)  # Remove duplicates

                if len(all_confident_indices) == 0:
                    print("No high-confidence labels found. Stopping early.")
                    f.write("No high-confidence labels found. Stopping early.\n")
                    break

                print("Updating training data with new labels...")
                f.write("Updating training data with new labels...\n")

                # Step 2: Expand training embeddings
                train_embeddings = np.vstack([train_embeddings, test_embeddings[all_confident_indices]])

                # Step 3: Initialize new training labels with consistent shape
                new_train_labels_HD = np.zeros(len(all_confident_indices))
                new_train_labels_CV = np.zeros(len(all_confident_indices))
                new_train_labels_VO = np.zeros(len(all_confident_indices))
                new_train_labels_no_label = np.zeros(len(all_confident_indices))

                # Step 4: Assign confident labels correctly
                new_train_labels_HD[np.isin(all_confident_indices, confident_indices_HD)] = new_labels_HD
                new_train_labels_CV[np.isin(all_confident_indices, confident_indices_CV)] = new_labels_CV
                new_train_labels_VO[np.isin(all_confident_indices, confident_indices_VO)] = new_labels_VO
                new_train_labels_no_label[
                    np.isin(all_confident_indices, confident_indices_no_label)] = new_labels_no_label

                # Step 5: Update labels
                train_labels_HD = np.hstack([train_labels_HD, new_train_labels_HD])
                train_labels_CV = np.hstack([train_labels_CV, new_train_labels_CV])
                train_labels_VO = np.hstack([train_labels_VO, new_train_labels_VO])
                train_labels_no_label = np.hstack([train_labels_no_label, new_train_labels_no_label])

                # Step 6: Retrain models on the updated dataset
                model_HD.fit(train_embeddings, train_labels_HD)
                model_CV.fit(train_embeddings, train_labels_CV)
                model_VO.fit(train_embeddings, train_labels_VO)
                model_no_label.fit(train_embeddings, train_labels_no_label)

            print("\nFinal evaluation...")
            f.write("\nFinal evaluation...\n")

            predictions_HD = model_HD.predict(test_embeddings)
            predictions_CV = model_CV.predict(test_embeddings)
            predictions_VO = model_VO.predict(test_embeddings)
            predictions_no_label = model_no_label.predict(test_embeddings)

            final_predictions = np.vstack([predictions_HD, predictions_CV, predictions_VO]).T
            accuracy = (final_predictions == test_labels[:, :3]).mean()

            report = classification_report(test_labels[:, :3], final_predictions, output_dict=True)

            macro_avg = report.get('macro avg', {})
            precision = macro_avg.get('precision', None)
            recall = macro_avg.get('recall', None)
            macro_f1 = macro_avg.get('f1-score', None)

            print(f"Accuracy: {accuracy:.4f}")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            print(f"Precision: {precision:.4f}")
            f.write(f"Precision: {precision:.4f}\n")
            print(f"Recall: {recall:.4f}")
            f.write(f"Recall: {recall:.4f}\n")
            print(f"F1-score: {macro_f1:.4f}")
            f.write(f"F1 Score: {macro_f1:.4f}\n")

            mean_confidence = np.mean([
                np.max(np.abs(model_HD.decision_function(test_embeddings))),
                np.max(np.abs(model_CV.decision_function(test_embeddings))),
                np.max(np.abs(model_VO.decision_function(test_embeddings))),
                np.max(np.abs(model_no_label.decision_function(test_embeddings)))
            ])

            with open(CONFIDENCE_FILE, "a") as g:
                g.write(f"\n=== Confidence Scores for {method_name} ===\n")
                for idx in range(len(test_embeddings)):
                    conf = np.mean([
                        abs(model_HD.decision_function([test_embeddings[idx]])[0]),
                        abs(model_CV.decision_function([test_embeddings[idx]])[0]),
                        abs(model_VO.decision_function([test_embeddings[idx]])[0]),
                        abs(model_no_label.decision_function([test_embeddings[idx]])[0]),
                    ])
                    g.write(f"Sample {idx}: Avg Confidence = {conf:.4f}\n")

            print(f"Mean Confidence: {mean_confidence:.4f}")
            f.write(f"Mean Confidence: {mean_confidence:.4f}\n")

