import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import argparse
from collections import defaultdict

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
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    clf.fit(X_train, y_encoded)
    return clf, le

def evaluate_partial(clf, le, X_test, y_test, threshold=0.2, verbose=True):
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

    # Added: Micro-average per label
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

       # print("\nExample Confidence Scores:")
       # for i in range(min(14, len(confidences))):
        #    print(f"Test sample {i}:")
         #   for label in label_list:
          #      print(f"  {label}: {confidences[i].get(label, 0):.3f}")
           # print(f"  Ground truth: {y_test[i]}")
           # print(f"  Predicted:    {predictions[i]}")
           # print("")

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
    parser.add_argument('--threshold', type=float, default=0.2, help="Confidence threshold for label inclusion")
    args = parser.parse_args()

    print("Loading training data...")
    X_train, y_train = load_training_data(args.train)

    print("Loading test data...")
    X_test, y_test = load_test_data(args.test_embeddings, args.test_labels)

    print("Training model...")
    clf, le = train_model(X_train, y_train)

    print("Evaluating with threshold:", args.threshold)
    evaluate_partial(clf, le, X_test, y_test, threshold=args.threshold)

if __name__ == "__main__":
    main()

# python3 sing_train.py --test_embeddings test_embeddings.npy --test_labels test_labels.npy --threshold 0.3
# python3 sing_train.py --test_embeddings test_embeddings.npy --test_labels test_labels.npy

