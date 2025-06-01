import argparse
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, jaccard_score
from sklearn.model_selection import train_test_split  # NEW
from sklearn.metrics import f1_score


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
        print(f"\n=== Processing method: {method_name} ===")
        if method_name == "baseline":
            train_embeddings = np.load(os.path.join(args.baseline_data_dir, 'train_embeddings.npy'))
            train_labels = np.load(os.path.join(args.baseline_data_dir, 'train_labels.npy'))
        else:
            train_embeddings = np.load(emb_file)
            train_labels = np.load(label_file)

        y = np.array([[int(c) for c in label] for label in train_labels])

        # Optional dev split
        if args.split_dev:
            train_embeddings, dev_embeddings, y, dev_labels = train_test_split(
                train_embeddings, y, test_size=0.222, random_state=42, stratify=y
            )
            print(f"[INFO] Training set: {len(train_embeddings)} samples")
            print(f"[INFO] Dev set:      {len(dev_embeddings)} samples")
        else:
            print(f"[INFO] Training set: {len(train_embeddings)} samples")

        print(f"[INFO] Test set:     {len(test_embeddings)} samples")

        models = []
        for i in range(args.labels):
            y_i = y[:, i] if i < y.shape[1] else (y.sum(axis=1) == 0).astype(int)
            model = LinearSVC(max_iter=5000).fit(train_embeddings, y_i)
            models.append(model)

        for _ in range(3):
            confident_indices_all = []
            new_labels_all = []
            for i, model in enumerate(models):
                if args.confidence:
                    new_labels, confident_indices = refine_labels(model, test_embeddings, args.threshold)
                else:
                    new_labels = model.predict(test_embeddings)
                    confident_indices = np.arange(len(test_embeddings))
                confident_indices_all.append(confident_indices)
                new_labels_all.append((i, new_labels, confident_indices))

            all_confident = np.unique(np.concatenate(confident_indices_all))
            if len(all_confident) == 0:
                break

            train_embeddings = np.vstack([train_embeddings, test_embeddings[all_confident]])
            for i, new_labels, indices in new_labels_all:
                y_new = np.zeros(len(all_confident))
                y_new[np.isin(all_confident, indices)] = new_labels
                y_i = y[:, i] if i < y.shape[1] else (y.sum(axis=1) == 0).astype(int)
                y_i = np.hstack([y_i, y_new])
                models[i] = LinearSVC(max_iter=5000).fit(train_embeddings, y_i)

        preds = np.stack([model.predict(test_embeddings) for model in models], axis=1)
        gt = test_labels[:, :args.labels]
        report = classification_report(gt, preds, output_dict=True)
        macro_avg = report.get('macro avg', {})

        print(classification_report(gt, preds))
        print("Jaccard score:", jaccard_score(gt, preds, average='samples'))

        precision = macro_avg.get("precision", 0.0)
        recall = macro_avg.get("recall", 0.0)
        f1_score_val = macro_avg.get("f1-score", 0.0)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1_score_val:.4f}")

        # Save confidence scores
        label_names = ["HD", "CV", "VO", "None"] if args.labels == 4 else None
        save_confidence_scores_to_files(models, test_embeddings, label_names)


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
    else:
        raise ValueError(f"Unknown setting: {setting}")


def evaluate_on_threshold(models, embeddings, labels, threshold):
    preds = []
    for model in models:
        scores = model.decision_function(embeddings)
        pred = (scores >= threshold).astype(int)
        preds.append(pred)
    preds = np.stack(preds, axis=1)
    return f1_score(labels[:, :len(models)], preds, average="macro")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, choices=["baseline_data", "resampled_data", "resampled_data2_1", "resampled_data3_1", "all"],
                        help="Directory containing data. Use 'all' to include all directories.")
    parser.add_argument("--setting", choices=["binary", "multiclass"], required=True, help="Setting for binary or multiclass classification.")
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
            main(args)
    else:
        main(args)

"""
sample usage for running a specific directory:
!python3 main.py --data_dir resampled_data2_1 --setting multiclass --confidence --threshold 0.8 --labels 4

sample usage for running all directiories:
!python3 main.py --data_dir all --setting binary --confidence --threshold 0.7 --labels 8

sample usage for running with a baseline directory:
!python3 main.py --data_dir resampled_data --setting multiclass --confidence --threshold 0.9 --labels 4 --baseline_data_dir /path/to/baseline_data
"""


