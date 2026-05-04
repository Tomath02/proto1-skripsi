import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC


def parse_args():
    parser = argparse.ArgumentParser(description="Train TF-IDF + SVM baseline.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features-word", type=int, default=150000)
    parser.add_argument("--max-features-char", type=int, default=250000)
    return parser.parse_args()


def load_rows(path: Path):
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))


def metric_dict(y_true, prob_ai, threshold):
    y_pred = (prob_ai >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_ai)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, prob_ai)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "size": int(len(y_true)),
    }


def find_best_threshold(y_true, prob_ai):
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(y_true, (prob_ai >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.input_csv))
    if not rows:
        raise SystemExit("Input CSV is empty.")

    def pick(split_name):
        subset = [r for r in rows if r.get(args.split_col, "").strip().lower() == split_name]
        texts = [r.get(args.text_col, "") for r in subset]
        labels = np.array([int(r.get(args.label_col, 0)) for r in subset], dtype=np.int32)
        return texts, labels

    x_train, y_train = pick("train")
    x_val, y_val = pick("val")
    x_test, y_test = pick("test")
    x_ood, y_ood = pick("ood")
    if not x_train or not x_val or not x_test:
        raise SystemExit("Dataset must contain train, val, and test split.")

    vectorizer_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=args.max_features_word,
        sublinear_tf=True,
        lowercase=True,
    )
    vectorizer_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=args.max_features_char,
        sublinear_tf=True,
        lowercase=True,
    )

    xw_train = vectorizer_word.fit_transform(x_train)
    xc_train = vectorizer_char.fit_transform(x_train)
    x_train_matrix = hstack([xw_train, xc_train], format="csr")

    base_svm = LinearSVC(C=1.0, class_weight="balanced", random_state=args.seed)
    try:
        classifier = CalibratedClassifierCV(estimator=base_svm, cv=3, method="sigmoid")
    except TypeError:
        classifier = CalibratedClassifierCV(base_estimator=base_svm, cv=3, method="sigmoid")
    classifier.fit(x_train_matrix, y_train)

    def predict_prob(texts):
        xw = vectorizer_word.transform(texts)
        xc = vectorizer_char.transform(texts)
        matrix = hstack([xw, xc], format="csr")
        return classifier.predict_proba(matrix)[:, 1]

    val_prob = predict_prob(x_val)
    threshold = find_best_threshold(y_val, val_prob)

    test_prob = predict_prob(x_test)
    metrics = {
        "model": "tfidf_svm",
        "label_mapping": {"human": 0, "ai": 1},
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
        "ood_size": len(x_ood),
        "best_threshold_from_val": threshold,
        "val_metrics": metric_dict(y_val, val_prob, threshold),
        "test_metrics": metric_dict(y_test, test_prob, threshold),
    }
    if x_ood:
        ood_prob = predict_prob(x_ood)
        metrics["ood_metrics"] = metric_dict(y_ood, ood_prob, threshold)

    bundle = {
        "model_type": "tfidf_svm",
        "vectorizer_word": vectorizer_word,
        "vectorizer_char": vectorizer_char,
        "classifier": classifier,
        "threshold": threshold,
        "label_mapping": {"human": 0, "ai": 1},
        "text_col": args.text_col,
        "label_col": args.label_col,
    }
    with (output_dir / "baseline_bundle.pkl").open("wb") as file:
        pickle.dump(bundle, file)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label", "prob_ai", "pred_label"])
        predictions = (test_prob >= threshold).astype(int)
        for text, label, prob, pred in zip(x_test, y_test, test_prob, predictions):
            writer.writerow([text, int(label), float(prob), int(pred)])

    print(f"[done] output_dir={output_dir}")
    print(f"[done] threshold={threshold:.4f}")
    print(f"[done] test_f1={metrics['test_metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
