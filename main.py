import os
import numpy as np
from itertools import product
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, jaccard_score
from sklearn.model_selection import train_test_split
import argparse


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'max_iter': [1000, 5000, 10000],
    'tol': [1e-4, 1e-3, 1e-2],
    'loss': ['hinge', 'squared_hinge'],
    'penalty': ['l2'],  # l1 only with dual=False, might want to test that separately
    'dual': [True, False]
}

def refine_labels(model, X_unlabeled, confidence_threshold):
    decision_function = model.decision_function(X_unlabeled)  # shape: (n_samples, n_classes)
    # Get max confidence per sample
    if decision_function.ndim == 1:
        # falls back on binary classification
        max_confidence = np.abs(decision_function)
        predicted_labels = (decision_function >= 0).astype(int)
    else:
        max_confidence = np.max(decision_function, axis=1)  # max margin per sample
        predicted_labels = np.argmax(decision_function, axis=1)

    confident_indices = np.where(max_confidence >= confidence_threshold)[0]
    if len(confident_indices) == 0:
        return np.array([]), confident_indices
    return predicted_labels[confident_indices], confident_indices

def save_confidence_scores_to_files(models, test_embeddings, label_names=None, output_dir="confidence_scores"):
    """
    Save confidence scores (distance to decision boundary) to separate files for each label.

    Args:
        models (list): Trained LinearSVC models.
        test_embeddings (np.ndarray): Test embeddings.
        label_names (list): List of label names. Defaults to "Label 0", "Label 1", etc.
        output_dir (str): Directory to save output files.
    """
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(len(models))]

    os.makedirs(output_dir, exist_ok=True)

    for i, model in enumerate(models):
        scores = model.decision_function(test_embeddings)
        filename = os.path.join(output_dir, f"confidence_{label_names[i]}.txt")
        with open(filename, "w") as f:
            f.write(f"Confidence scores for label: {label_names[i]}\n")
            for j, score in enumerate(scores):
                f.write(f"Sample {j}: {score:.4f}\n")

    print(f"Confidence scores saved in: {os.path.abspath(output_dir)}")

def main(args):
    current_directory = os.getcwd()
    test_embeddings = np.load(os.path.join(current_directory, 'test_embeddings.npy'))
    test_labels = np.load(os.path.join(current_directory, 'test_labels.npy'))

    methods = load_methods(args.setting)

    for method_name, emb_file, label_file in methods:
        print(f"[INFO] Processing method: {method_name}")

        # Load training data based on method or baseline setting
        if method_name == "baseline" or args.setting == "baseline":
            if not args.baseline_data_dir:
                raise ValueError("Baseline data directory must be specified for baseline setting.")
            train_embeddings = np.load(os.path.join(args.baseline_data_dir, 'train_embeddings.npy'))
            train_labels = np.load(os.path.join(args.baseline_data_dir, 'train_labels.npy'))
        else:
            train_embeddings = np.load(emb_file)
            train_labels = np.load(label_file)

        y = np.array([[int(c) for c in label] for label in train_labels])

        if args.split_dev:
            train_emb, dev_emb, train_lbls, dev_lbls = train_test_split(
                train_embeddings, y, test_size=0.222, random_state=42, stratify=y if y.ndim == 1 else None
            )
            print(f"[INFO] Training samples: {len(train_emb)}")
            print(f"[INFO] Dev samples: {len(dev_emb)}")
        else:
            train_emb, train_lbls = train_embeddings, y
            dev_emb, dev_lbls = None, None
            print(f"[INFO] Training samples: {len(train_emb)}")

        print(f"[INFO] Test samples: {len(test_embeddings)}")

        # Train initial models
        models = []
        for i in range(args.labels):
            if i < train_lbls.shape[1]:
                y_i = train_lbls[:, i]
            else:
                y_i = (train_lbls.sum(axis=1) == 0).astype(int)
            model = LinearSVC(max_iter=10000, random_state=42).fit(train_emb, y_i)
            models.append(model)

        best_score = -1
        best_params = None
        best_models = None

        if dev_emb is not None:
            print("[INFO] Starting hyperparameter grid search on dev set...")

            for combo in product(
                param_grid['C'],
                param_grid['tol'],
                param_grid['loss'],
                param_grid['penalty'],
                param_grid['dual']
            ):
                params = {
                    'C': combo[0],
                    'tol': combo[1],
                    'loss': combo[2],
                    'penalty': combo[3],
                    'dual': combo[4]
                }

                try:
                    models_tuned = []
                    preds_dev = []

                    for i in range(args.labels):
                        if i < train_lbls.shape[1]:
                            y_train = train_lbls[:, i]
                            y_dev = dev_lbls[:, i]
                        else:
                            y_train = (train_lbls.sum(axis=1) == 0).astype(int)
                            y_dev = (dev_lbls.sum(axis=1) == 0).astype(int)

                        model = LinearSVC(**params, max_iter=10000, random_state=42).fit(train_emb, y_train)
                        models_tuned.append(model)
                        preds_dev.append(model.predict(dev_emb))

                    preds_dev = np.stack(preds_dev, axis=1)
                    report = classification_report(dev_lbls[:, :args.labels], preds_dev, output_dict=True, zero_division=0)
                    macro_f1 = report.get('macro avg', {}).get('f1-score', 0.0)

                    if macro_f1 > best_score:
                        best_score = macro_f1
                        best_params = params
                        best_models = models_tuned
                except Exception:
                    continue

            print(f"[INFO] Best params: {best_params}")
            print(f"[INFO] Best dev macro F1: {best_score:.4f}")

            # Retrain on combined set
            combined_emb = np.vstack([train_emb, dev_emb])
            combined_lbls = np.vstack([train_lbls, dev_lbls])
        else:
            print("[INFO] No dev set, training with default parameters")
            combined_emb = train_emb
            combined_lbls = train_lbls

        # Use best or fallback params
        params_to_use = best_params if best_params is not None else {
            'C': 1.0,
            'max_iter': 10000,
            'tol': 1e-4,
            'loss': 'squared_hinge',
            'penalty': 'l2',
            'dual': True
        }

        final_models = []
        for i in range(args.labels):
            if i < combined_lbls.shape[1]:
                y_i = combined_lbls[:, i]
            else:
                y_i = (combined_lbls.sum(axis=1) == 0).astype(int)

            model = LinearSVC(**params_to_use, random_state=42).fit(combined_emb, y_i)
            final_models.append(model)

        preds_test = np.stack([model.predict(test_embeddings) for model in final_models], axis=1)

        if preds_test.shape[1] != test_labels.shape[1]:
            min_labels = min(preds_test.shape[1], test_labels.shape[1])
            preds_test = preds_test[:, :min_labels]
            test_labels = test_labels[:, :min_labels]

        report = classification_report(test_labels, preds_test, output_dict=True, zero_division=0)
        print(classification_report(test_labels, preds_test, zero_division=0))
        print("Jaccard score:", jaccard_score(test_labels, preds_test, average='samples'))

        precision = report.get('macro avg', {}).get("precision", 0.0)
        recall = report.get('macro avg', {}).get("recall", 0.0)
        f1 = report.get('macro avg', {}).get("f1-score", 0.0)
        print(f"[INFO] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


def load_methods(setting):
    if setting == "multiclass":
        return [
            ("ADASYN", "resampled_data/multiclass_ADASYN_embeddings.npy", "resampled_data/multiclass_ADASYN_labels.npy"),
            ("CondensedNearestNeighbour", "resampled_data/multiclass_CondensedNearestNeighbour_embeddings.npy", "resampled_data/multiclass_CondensedNearestNeighbour_labels.npy"),
            ("RandomOverSampler", "resampled_data/multiclass_RandomOverSampler_embeddings.npy", "resampled_data/multiclass_RandomOverSampler_labels.npy"),
            ("RandomUnderSampler", "resampled_data/multiclass_RandomUnderSampler_embeddings.npy", "resampled_data/multiclass_RandomUnderSampler_labels.npy"),
            ("SMOTE", "resampled_data/multiclass_SMOTE_embeddings.npy", "resampled_data/multiclass_SMOTE_labels.npy"),
            ("SMOTEENN", "resampled_data/multiclass_SMOTEENN_embeddings.npy", "resampled_data/multiclass_SMOTEENN_labels.npy"),
            ("TomekLinks", "resampled_data/multiclass_TomekLinks_embeddings.npy", "resampled_data/multiclass_TomekLinks_labels.npy")
        ]
    elif setting == "binary":
        return [
            ("binary_ADASYN", "resampled_data/binary_ADASYN_embeddings.npy", "resampled_data/binary_ADASYN_labels.npy"),
            ("binary_CondensedNearestNeighbour", "resampled_data/binary_CondensedNearestNeighbour_embeddings.npy", "resampled_data/binary_CondensedNearestNeighbour_labels.npy"),
            ("binary_RandomOverSampler", "resampled_data/binary_RandomOverSampler_embeddings.npy", "resampled_data/binary_RandomOverSampler_labels.npy"),
            ("binary_RandomUnderSampler", "resampled_data/binary_RandomUnderSampler_embeddings.npy", "resampled_data/binary_RandomUnderSampler_labels.npy"),
            ("binary_SMOTE", "resampled_data/binary_SMOTE_embeddings.npy", "resampled_data/binary_SMOTE_labels.npy"),
            ("binary_SMOTEENN", "resampled_data/binary_SMOTEENN_embeddings.npy", "resampled_data/binary_SMOTEENN_labels.npy"),
            ("binary_TomekLinks", "resampled_data/binary_TomekLinks_embeddings.npy", "resampled_data/binary_TomekLinks_labels.npy")
        ]
    elif setting == "baseline":
    # Single entry for baseline since there's just one dataset
        return [
            ("baseline", "baseline_data/train_embeddings.npy", "baseline_data/train_labels.npy")
        ]
    else:
        raise ValueError(f"Unknown setting: {setting}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, choices=["baseline_data", "resampled_data", "resampled_data2_1", "resampled_data3_1", "all"],
                        help="Directory containing data. Use 'all' to include all directories.")
    parser.add_argument("--setting", choices=["binary", "multiclass", "baseline"], required=True, help="Setting for binary or multiclass classification.")
    parser.add_argument("--confidence", action="store_true", help="Enable confidence scoring.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold.")
    parser.add_argument("--labels", type=int, choices=[4, 8], default=4, help="Number of labels to classify (4 or 8).")
    parser.add_argument("--baseline_data_dir", type=str, help="Directory for baseline data.")
    parser.add_argument("--split_dev", action="store_true", help="Optionally split training set into training and dev sets (7:2 ratio).")

    args = parser.parse_args()

    if args.data_dir == "all":
        dirs = ["baseline_data", "resampled_data", "resampled_data2_1", "resampled_data3_1"]
        for directory in dirs:
            print(f"\n--- Processing directory: {directory} ---")
            args.data_dir = directory
            
            # If directory is baseline_data, and setting is baseline, set baseline_data_dir automatically
            if directory == "baseline_data" and args.setting == "baseline":
                args.baseline_data_dir = directory
            else:
                # Clear baseline_data_dir for other directories to avoid confusion
                args.baseline_data_dir = None
            
            main(args)
    else:
        main(args)

    # NEW: Optional dev split
    if args.split_dev:
        train_embeddings, dev_embeddings, train_labels, dev_labels = train_test_split(
            train_embeddings, train_labels, test_size=0.222, random_state=42, stratify=train_labels
        )
        print(f"[INFO] Training set: {len(train_embeddings)} samples")
        print(f"[INFO] Dev set:      {len(dev_embeddings)} samples")
    else:
        print(f"[INFO] Training set: {len(train_embeddings)} samples")
    print(f"[INFO] Test set:     {len(test_embeddings)} samples")

"""
sample usage for running a specific directory:
!python3 main.py --data_dir resampled_data2_1 --setting multiclass --confidence --threshold 0.8 --labels 4

sample usage for running all directiories:
!python3 main.py --data_dir all --setting binary --confidence --threshold 0.7 --labels 8

sample usage for running with a baseline directory:
!python3 main.py --data_dir resampled_data --setting multiclass --confidence --threshold 0.9 --labels 4 --baseline_data_dir /path/to/baseline_data
"""

