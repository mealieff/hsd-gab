import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import argparse

def refine_labels(model, X_unlabeled, confidence_threshold):
    """Return binary predictions based on decision function threshold."""
    decision_function = model.decision_function(X_unlabeled)
    preds = (decision_function >= confidence_threshold).astype(int)

    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    # Add "None" label if no label exceeds threshold
    none_mask = preds.sum(axis=1) == 0
    if none_mask.any():
        none_col = np.zeros((preds.shape[0], 1), dtype=int)
        none_col[none_mask, 0] = 1
        preds = np.hstack([preds, none_col])
    return preds

def save_confidence_scores_to_files(model, test_embeddings, label_names=None, output_dir="confidence_scores"):
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(len(model.estimators_))]
    os.makedirs(output_dir, exist_ok=True)
    for i, m in enumerate(model.estimators_):
        scores = m.decision_function(test_embeddings)
        filename = os.path.join(output_dir, f"confidence_{label_names[i]}.txt")
        with open(filename, "w") as f:
            f.write(f"Confidence scores for label: {label_names[i]}\n")
            for j, score in enumerate(scores):
                f.write(f"Sample {j}: {score:.4f}\n")
    print(f"[INFO] Confidence scores saved in: {os.path.abspath(output_dir)}")

def load_single_label_file(emb_file):
    """Load (embedding, label) pairs from .npy object array."""
    data = np.load(emb_file, allow_pickle=True)
    # Handle object arrays of (embedding, label) tuples
    if isinstance(data, np.ndarray) and data.dtype == object:
        if all(isinstance(x, (tuple, list)) and len(x) == 2 for x in data):
            embeddings = np.vstack([np.array(x[0], dtype=float) for x in data])
            labels = np.array([x[1] if isinstance(x[1], int) else 0 for x in data])
            return embeddings, labels
        else:
            raise ValueError(f"Could not parse object array in {emb_file}")
    elif isinstance(data, tuple) and len(data) == 2:
        return np.array(data[0], dtype=float), np.array(data[1])
    else:
        raise ValueError(f"Unrecognized structure in {emb_file}")

def main(args):
    current_directory = os.getcwd()
    test_embeddings = np.load(os.path.join(current_directory, "test_embeddings.npy"))
    test_labels = np.load(os.path.join(current_directory, "test_labels.npy"))

    label_scheme = {
        "HD": [1, 0, 0],
        "CV": [0, 1, 0],
        "VO": [0, 0, 1],
        "None": [0, 0, 0]
    }

    methods = [f.replace("_single_label.npy","") for f in os.listdir(args.data_dir) if f.endswith("_single_label.npy")]

    for method_name in methods:
        emb_file = os.path.join(args.data_dir, f"{method_name}_single_label.npy")
        print(f"[INFO] Processing method: {method_name}")

        train_embeddings, train_labels = load_single_label_file(emb_file)

        if args.split_dev:
            # Split for threshold selection
            train_emb, dev_emb, train_lbls, dev_lbls = train_test_split(
                train_embeddings, train_labels, test_size=0.2, random_state=42
            )
            print(f"[INFO] Training samples: {len(train_emb)}")
            print(f"[INFO] Dev samples: {len(dev_emb)}")

            # Fixed best hyperparameters
            best_params = {'C': 10, 'tol': 0.001, 'loss': 'squared_hinge', 'penalty': 'l2', 'dual': False}
            base_model = LinearSVC(**best_params, max_iter=10000, random_state=42)
            ovr_model = OneVsRestClassifier(base_model, n_jobs=1)
            ovr_model.fit(train_emb, train_lbls)

            # Compute predictions on dev to select threshold
            thresholds = [0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9]
            best_macro_f1 = -1
            best_thresh = 0.7
            for thr in thresholds:
                preds_thr = refine_labels(ovr_model, dev_emb, thr)
                macro_f1 = f1_score(dev_lbls, preds_thr, average="macro", zero_division=0)
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_thresh = thr
            print(f"[INFO] Selected best threshold: {best_thresh}")
        else:
            best_thresh = args.threshold

        # Train final model on full training set
        base_model = LinearSVC(**{'C': 10, 'tol': 0.001, 'loss': 'squared_hinge', 'penalty': 'l2', 'dual': False},
                               max_iter=10000, random_state=42)
        final_model = OneVsRestClassifier(base_model)
        final_model.fit(train_embeddings, train_labels)

        # Predict on test set
        preds_test = refine_labels(final_model, test_embeddings, best_thresh)

        # Evaluation
        micro_f1 = f1_score(test_labels, preds_test, average="micro", zero_division=0)
        macro_f1 = f1_score(test_labels, preds_test, average="macro", zero_division=0)
        print(f"[RESULT] Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}")

        if args.confidence:
            save_confidence_scores_to_files(final_model, test_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing data.")
    parser.add_argument("--confidence", action="store_true", help="Enable confidence scoring.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold.")
    parser.add_argument("--split_dev", action="store_true", help="Split training set into training and dev sets for threshold selection.")
    parser.add_argument("--setting", choices=["single"], required=True)
    args = parser.parse_args()

    main(args)

