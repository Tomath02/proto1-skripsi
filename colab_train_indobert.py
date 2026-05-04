"""
Colab-compatible IndoBERT training script.

Usage in Google Colab:

!pip install -q -r /content/drive/MyDrive/skripsi/pipeline2/requirements-colab.txt

%run /content/drive/MyDrive/skripsi/pipeline2/colab_train_indobert.py \
  --input-csv /content/drive/MyDrive/skripsi/pipeline2/artifacts/dataset_indobert256.csv \
  --output-dir /content/drive/MyDrive/skripsi/pipeline2/artifacts/indobert \
  --model-name indobenchmark/indobert-base-p1 \
  --max-length 256 \
  --epochs 3 \
  --batch-size 8
"""

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERT for AI text detection.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="indobenchmark/indobert-base-p1")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-zip", action="store_true", help="Export best_model as zip file")
    return parser.parse_args()


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_rows(path: Path):
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))


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


def find_best_threshold(y_true, prob_ai):
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(y_true, (prob_ai >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


@dataclass
class TextDataset:
    encodings: dict
    labels: list[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        import torch

        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise SystemExit("Install dependencies first: pip install torch transformers accelerate") from exc

    rows = load_rows(Path(args.input_csv))
    if not rows:
        raise SystemExit("Input CSV is empty.")

    def pick(split_name):
        subset = [r for r in rows if r.get(args.split_col, "").strip().lower() == split_name]
        texts = [r.get(args.text_col, "") for r in subset]
        labels = [int(r.get(args.label_col, 0)) for r in subset]
        return texts, labels

    x_train, y_train = pick("train")
    x_val, y_val = pick("val")
    x_test, y_test = pick("test")
    x_ood, y_ood = pick("ood")
    if not x_train or not x_val or not x_test:
        raise SystemExit("Dataset must contain train, val, and test split.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def encode(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=args.max_length,
        )

    train_dataset = TextDataset(encode(x_train), y_train)
    val_dataset = TextDataset(encode(x_val), y_val)
    test_dataset = TextDataset(encode(x_test), y_test)
    ood_dataset = TextDataset(encode(x_ood), y_ood) if x_ood else None

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

    training_args_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": 50,
        "report_to": [],
        "seed": args.seed,
    }
    try:
        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            **training_args_kwargs,
        )
    except TypeError:
        training_args = TrainingArguments(
            eval_strategy="epoch",
            **training_args_kwargs,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    def predict_prob(dataset):
        pred = trainer.predict(dataset)
        return torch.softmax(torch.tensor(pred.predictions), dim=1)[:, 1].numpy()

    val_prob = predict_prob(val_dataset)
    threshold = find_best_threshold(np.array(y_val), val_prob)
    test_prob = predict_prob(test_dataset)

    metrics = {
        "model": args.model_name,
        "label_mapping": {"human": 0, "ai": 1},
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
        "ood_size": len(x_ood),
        "best_threshold_from_val": threshold,
        "val_metrics": metrics_at_threshold(np.array(y_val), val_prob, threshold),
        "test_metrics": metrics_at_threshold(np.array(y_test), test_prob, threshold),
    }
    if ood_dataset is not None:
        ood_prob = predict_prob(ood_dataset)
        metrics["ood_metrics"] = metrics_at_threshold(np.array(y_ood), ood_prob, threshold)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "threshold.json").write_text(
        json.dumps({"threshold": threshold}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps({"human": 0, "ai": 1}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label", "prob_ai", "pred_label"])
        predictions = (test_prob >= threshold).astype(int)
        for text, label, prob, pred in zip(x_test, y_test, test_prob, predictions):
            writer.writerow([text, int(label), float(prob), int(pred)])

    if args.export_zip:
        best_model_dir = output_dir / "best_model"
        zip_path = output_dir / "best_model_export.zip"
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(best_model_dir))
        print(f"[done] exported_zip={zip_path}")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] best_model={output_dir / 'best_model'}")


if __name__ == "__main__":
    main()
