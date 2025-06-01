import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
from collections import defaultdict

from sklearn.metrics import classification_report, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.svm import LinearSVC

label_list = ["HD", "CV", "VO", "NONE"]

def load_training_data(path):
    data = np.load(path, allow_pickle=True)
    X = np.array([row[0] for row in data])
    y = np.array([row[1] for row in data])
    return X, y

def load_test_data(embedding_path, label_path):
    X = np.load(embedding_path)
    y_raw = np.load(label_path, allow_pickle=True)

    y = []
    for row in y_raw:
        labels = [label_list[i] for i, val in enumerate(row[:3]) if val == 1]
        y.append(labels if labels else ["NONE"])

    return X, y

def train_model(X_train, y_train):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_encoded)
    return clf, le

def evaluate_partial(clf, le, X_test, y_test, threshold=0.2, use_decision_function=False, verbose=True):
    if use_decision_function:
        # Use decision_function scores
        decision_scores = clf.decision_function(X_test)
        # decision_function returns shape (n_samples, n_classes)
        # We convert scores to a pseudo-confidence by normalizing scores per sample
        # so we can threshold them similarly.
        # Alternatively, just threshold raw scores as is.
        scores = decision_scores
        # For binary/multiclass, decision_function shape can be (n_samples,) or (n_samples, n_classes)
        if len(scores.shape) == 1:
            # binary case, convert to shape (n_samples, 2)
            scores = np.vstack([-scores, scores]).T

        # We will threshold the scores directly
        class_labels = le.inverse_transform(np.arange(len(le.classes_)))

        confidences = []
        predictions = []

        for i, score in enumerate(scores):
            conf_dict = {label: score[j] for j, label in enumerate(class_labels)}
            confidences.append(conf_dict)

            active_labels = [label for label in ["HD", "CV", "VO"] if conf_dict[label] >= threshold]

            if active_labels:
                predictions.append(active_labels)
            else:
                predictions.append(["NONE"])
    else:
        # Use predict_proba
        probas = clf.predict_proba(X_test)
        class_labels = le.inverse_transform(np.arange(len(le.classes_)))

        confidences = []
        predictions = []

        for i, prob in enumerate(probas):
            conf_dict = {label: prob[j] for j, label in enumerate(class_labels)}
            confidences.append(conf_dict)

            active_labels = [label for label in ["HD", "CV", "VO"] if conf_dict[label] >= threshold]

            if active_labels:
                predictions.append(active_labels)
            else:
                predictions.append(["NONE"])

    precision_sum = recall_sum = f1_sum = 0
    total = len(y_test)

    # Micro-average per label
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred_labels, true_labels in zip(predictions, y_test):
        pred_set = set(pred_labels)
        true_set = set(true_labels)

        overlap = len(pred_set & true_set)
        denom_p = len(pred_set) if pred_set else 1
        denom_r = len(true_set) if true_set else 1

        precision_sum += overlap / denom_p
        recall_sum += overlap / denom_r
        f1_sum += (2 * overlap) / (denom_p + denom_r) if (denom_p + denom_r) > 0 else 0

        for label in label_list:
            if label in pred_set and label in true_set:
                tp[label] += 1
            elif label in pred_set and label not in true_set:
                fp[label] += 1
            elif label not in pred_set and label in true_set:
                fn[label] += 1

    precision = precision_sum / total
    recall = recall_sum / total
    f1 = f1_sum / total

    if verbose:
        print("\nPartial Evaluation Results:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        print("\nMicroaveraged Scores per Label:")
        for label in label_list:
            p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
            r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
            f1_l = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            print(f"{label}: Precision={p:.3f} Recall={r:.3f} F1={f1_l:.3f}")

    output_data = []
    for i in range(len(X_test)):
        row = {
            "embedding": X_test[i],
            "confidences": confidences[i],
            "ground_truth": y_test[i],
            "predicted": predictions[i]
        }
        output_data.append(row)

    np.save("test_output_confidences.npy", output_data)
    print("Saved full confidence outputs to 'test_output_confidences.npy'.")

    return confidences, predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default="train_embeddings_single_label.npy")
    parser.add_argument('--test_embeddings', required=True)
    parser.add_argument('--test_labels', required=True)
    parser.add_argument("--split_dev", action="store_true")
    parser.add_argument("--use_decision_function", action="store_true", help="Use SVM decision_function instead of predict_proba")
    args = parser.parse_args()

    print("Loading training data...")
    X_all, y_all = load_training_data(args.train)

    print("Loading test data...")
    X_test, y_test = load_test_data(args.test_embeddings, args.test_labels)

    if args.split_dev:
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_all, y_all, test_size=0.222, random_state=42, stratify=y_all
        )
        print(f"[INFO] Training samples: {len(X_train)}")
        print(f"[INFO] Dev samples: {len(X_dev)}")
    else:
        X_train, y_train = X_all, y_all
        X_dev, y_dev = None, None
        print(f"[INFO] Training samples: {len(X_train)}")

    print("[INFO] Training model on training split...")
    clf, le = train_model(X_train, y_train)

    if args.split_dev:
        print("[INFO] Starting threshold sweep on dev set...")
        best_threshold = 0
        best_f1 = -1

        for threshold in np.arange(0.1, 1.01, 0.05):
            _, preds = evaluate_partial(clf, le, X_dev, y_dev, threshold=threshold, use_decision_function=args.use_decision_function, verbose=False)

            # Microaveraged F1 for HD, CV, VO
            tp = defaultdict(int)
            fp = defaultdict(int)
            fn = defaultdict(int)

            for pred_labels, true_labels in zip(preds, y_dev):
                pred_set = set(pred_labels)
                true_set = set(true_labels)

                for label in label_list:
                    if label in pred_set and label in true_set:
                        tp[label] += 1
                    elif label in pred_set and label not in true_set:
                        fp[label] += 1
                    elif label not in pred_set and label in true_set:
                        fn[label] += 1

            f1_scores = []
            for label in ["HD", "CV", "VO"]:
                p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
                r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
                f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
                f1_scores.append(f1)

            avg_f1 = np.mean(f1_scores)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = threshold

        print("\n==============================")
        print(f"Best threshold from dev set: {best_threshold:.2f}")
        print(f"Best avg F1 (HD, CV, VO):    {best_f1:.3f}")
        print("==============================")

        # Retrain on full data (train + dev)
        print("[INFO] Retraining model on full training data...")
        clf, le = train_model(X_all, y_all)
    else:
        best_threshold = 0.2  # default
        print("[INFO] No dev split; using default threshold 0.2")

    print("\n[INFO] Evaluating on test set with threshold", best_threshold)
    evaluate_partial(clf, le, X_test, y_test, threshold=best_threshold, use_decision_function=args.use_decision_function, verbose=True)


if __name__ == "__main__":
    main()
