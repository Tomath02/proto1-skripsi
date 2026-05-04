import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline or IndoBERT model.")
    parser.add_argument("--model-type", choices=["baseline", "indobert"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--split-value", default="")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def metrics_at_threshold(y_true, prob_ai, threshold):
    pred = (prob_ai >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_ai)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, prob_ai)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "size": int(len(y_true)),
    }


def load_rows(path: Path):
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))


def predict_baseline(model_path: Path, texts, threshold):
    with model_path.open("rb") as file:
        bundle = pickle.load(file)
    vec_word = bundle["vectorizer_word"]
    vec_char = bundle["vectorizer_char"]
    classifier = bundle["classifier"]
    if threshold is None:
        threshold = float(bundle.get("threshold", 0.5))
    matrix = hstack([vec_word.transform(texts), vec_char.transform(texts)], format="csr")
    return classifier.predict_proba(matrix)[:, 1], threshold


def predict_indobert(model_dir: Path, texts, threshold, max_length: int, batch_size: int):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise SystemExit("Install torch and transformers first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if threshold is None:
        threshold_path = model_dir.parent / "threshold.json"
        if threshold_path.exists():
            threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
        else:
            threshold = 0.5

    probs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.extend(prob.tolist())
    return np.array(probs, dtype=np.float32), threshold


def main():
    args = parse_args()
    rows = load_rows(Path(args.input_csv))
    if args.split_value:
        rows = [
            r
            for r in rows
            if r.get(args.split_col, "").strip().lower() == args.split_value.lower()
        ]
    if not rows:
        raise SystemExit("No rows to evaluate.")

    texts = [r.get(args.text_col, "") for r in rows]
    y_true = np.array([int(r.get(args.label_col, 0)) for r in rows], dtype=np.int32)

    if args.model_type == "baseline":
        prob_ai, threshold = predict_baseline(Path(args.model_path), texts, args.threshold)
    else:
        prob_ai, threshold = predict_indobert(
            Path(args.model_path),
            texts,
            args.threshold,
            args.max_length,
            args.batch_size,
        )

    result = metrics_at_threshold(y_true, prob_ai, threshold)
    result["model_type"] = args.model_type
    result["model_path"] = args.model_path
    result["split_value"] = args.split_value or "all"
    result["label_mapping"] = {"human": 0, "ai": 1}

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
