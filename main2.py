import os
import numpy as np
from itertools import product
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import argparse
import ast

def refine_labels(model, X_unlabeled, confidence_threshold):
    decision_function = model.decision_function(X_unlabeled)
    if decision_function.ndim == 1:
        max_confidence = np.abs(decision_function)
        predicted_labels = (decision_function >= 0).astype(int)
    else:
        max_confidence = np.max(decision_function, axis=1)
        predicted_labels = np.argmax(decision_function, axis=1)
    confident_indices = np.where(max_confidence >= confidence_threshold)[0]
    if len(confident_indices) == 0:
        return np.array([]), confident_indices
    return predicted_labels[confident_indices], confident_indices

def save_confidence_scores_to_files(models, test_embeddings, label_names=None, output_dir="confidence_scores"):
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(len(models.estimators_))]
    os.makedirs(output_dir, exist_ok=True)
    for i, model in enumerate(models.estimators_):
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

    label_scheme = {
        "HD": [1, 0, 0],
        "CV": [0, 1, 0],
        "VO": [0, 0, 1],
        "None": [0, 0, 0]
    }

    for method_name, emb_file, label_file in methods:
        print(f"[INFO] Processing method: {method_name}")

        if method_name == "baseline" or args.setting == "baseline":
            if not args.baseline_data_dir:
                raise ValueError("Baseline data directory must be specified for baseline setting.")
            train_embeddings = np.load(os.path.join(args.baseline_data_dir, 'train_embeddings.npy'))
            train_labels = np.load(os.path.join(args.baseline_data_dir, 'train_labels.npy'))
        else:
            train_embeddings = np.load(emb_file)
            train_labels = np.load(label_file)

        y = np.array([ [int(c) for c in label] if isinstance(label, (list, np.ndarray)) else [int(label)] for label in train_labels])


        if args.split_dev:
            train_emb, dev_emb, train_lbls, dev_lbls = train_test_split(
                train_embeddings, y, test_size=0.222, random_state=42, stratify=None
            )
            print(f"[INFO] Training samples: {len(train_emb)}")
            print(f"[INFO] Dev samples: {len(dev_emb)}")
        else:
            train_emb, train_lbls = train_embeddings, y
            dev_emb, dev_lbls = None, None
            print(f"[INFO] Training samples: {len(train_emb)}")

        print(f"[INFO] Test samples: {len(test_embeddings)}")

        best_score = -1
        best_params = None

        if dev_emb is not None:
            print("[INFO] Starting hyperparameter grid search on dev set...")
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'tol': [1e-4, 1e-3, 1e-2],
                'loss': ['hinge', 'squared_hinge'],
                'penalty': ['l2'],
                'dual': [True, False]
            }
            for combo in product(*param_grid.values()):
                params = dict(zip(param_grid.keys(), combo))
                try:
                    base_model = LinearSVC(**params, max_iter=10000, random_state=42)
                    ovr_model = OneVsRestClassifier(base_model)
                    ovr_model.fit(train_emb, train_lbls)
                    preds_dev = ovr_model.predict(dev_emb)
                    macro_f1 = f1_score(dev_lbls, preds_dev, average='macro')
                    if macro_f1 > best_score:
                        best_score = macro_f1
                        best_params = params
                except Exception as e:
                    print(f"[WARN] Skipping parameter set {params} due to error: {str(e)}")
                    continue
            print(f"[INFO] Best params: {best_params}")
            print(f"[INFO] Best dev macro F1: {best_score:.4f}")
            combined_emb = np.vstack([train_emb, dev_emb])
            combined_lbls = np.vstack([train_lbls, dev_lbls])
        else:
            print("[INFO] No dev set, training with specified or default parameters")
            combined_emb = train_emb
            combined_lbls = train_lbls
            if args.model_params:
                best_params = ast.literal_eval(args.model_params)
            else:
                best_params = {'C': 10, 'tol': 0.001, 'loss': 'squared_hinge', 'penalty': 'l2', 'dual': False}

        base_model = LinearSVC(**best_params, max_iter=10000, random_state=42)
        final_model = OneVsRestClassifier(base_model)
        final_model.fit(combined_emb, combined_lbls)
        preds_test = final_model.predict(test_embeddings)

        preds_test = np.atleast_2d(preds_test)
        test_labels = np.atleast_2d(test_labels)

        if preds_test.shape[1] != test_labels.shape[1]:
            min_labels = min(preds_test.shape[1], test_labels.shape[1])
            preds_test = preds_test[:, :min_labels]
            test_labels = test_labels[:, :min_labels]

        label_names = ["HD", "CV", "VO", "None"]
        label_metrics = []

        print("\n[EVAL] Manual label-wise F1 scores:")
        for i in range(test_labels.shape[1]):
            tp = np.sum((preds_test[:, i] == 1) & (test_labels[:, i] == 1))
            fp = np.sum((preds_test[:, i] == 1) & (test_labels[:, i] == 0))
            fn = np.sum((preds_test[:, i] == 0) & (test_labels[:, i] == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            label = label_names[i] if i < len(label_names) else f"Label_{i}"
            label_metrics.append((precision, recall, f1))
            scheme = label_scheme.get(label, [0, 0, 0])
            print(f"{label}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Scheme={scheme}")

        avg_prec = np.mean([x[0] for x in label_metrics])
        avg_recall = np.mean([x[1] for x in label_metrics])
        avg_f1 = np.mean([x[2] for x in label_metrics])
        print(f"\n[AVG] Precision={avg_prec:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")

        print("\n[SKLEARN] Evaluation:")
        print("Micro F1:", f1_score(test_labels, preds_test, average='micro'))
        print("Macro F1:", f1_score(test_labels, preds_test, average='macro'))

        if args.confidence:
            label_names_all = [f"Label_{i}" for i in range(test_labels.shape[1])]
            save_confidence_scores_to_files(final_model, test_embeddings, label_names_all)

def load_methods(setting):
    if setting == "multiclass":
        return [
            ("ADASYN", "resampled_data/multiclass_ADASYN_embeddings.npy", "resampled_data/multiclass_ADASYN_labels.npy"),
        ]
    elif setting == "binary":
        return [
            ("binary_ADASYN", "resampled_data/binary_ADASYN_embeddings.npy", "resampled_data/binary_ADASYN_labels.npy"),
        ]
    elif setting == "baseline":
        return [("baseline", "baseline_data/train_embeddings.npy", "baseline_data/train_labels.npy")]
    else:
        raise ValueError(f"Unknown setting: {setting}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing data.")
    parser.add_argument("--setting", choices=["binary", "multiclass", "baseline"], required=True)
    parser.add_argument("--confidence", action="store_true", help="Enable confidence scoring.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold.")
    parser.add_argument("--labels", type=int, default=4, help="Number of labels to classify.")
    parser.add_argument("--baseline_data_dir", type=str, help="Directory for baseline data.")
    parser.add_argument("--split_dev", action="store_true", help="Split training set into training and dev sets.")
    parser.add_argument("--model_params", type=str, help="Custom model parameters as dictionary string.")
    args = parser.parse_args()

    if args.data_dir == "all":
        for directory in ["baseline_data", "resampled_data", "resampled_data2_1", "resampled_data3_1"]:
            print(f"\n--- Processing directory: {directory} ---")
            args.data_dir = directory
            if directory == "baseline_data" and args.setting == "baseline":
                args.baseline_data_dir = directory
            else:
                args.baseline_data_dir = None
            try:
                main(args)
            except Exception as e:
                print(f"[ERROR] Failed to process {directory}: {str(e)}")

    else:
        main(args)
