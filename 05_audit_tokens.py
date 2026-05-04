import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and optionally filter dataset by IndoBERT token length.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="", help="If set, write filtered CSV.")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--tokenizer-name", default="indobenchmark/indobert-base-p1")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--drop-over-limit", action="store_true")
    return parser.parse_args()


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = round((len(values) - 1) * p)
    return values[index]


def summarize(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def main():
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise SystemExit("Install transformers first: pip install transformers") from exc

    rows = list(csv.DictReader(Path(args.input_csv).open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise SystemExit("Input CSV is empty.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    kept = []
    token_lengths = []
    by_split = {}
    by_label = {}

    for row in rows:
        text = row.get(args.text_col, "")
        token_count = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        row["token_count"] = token_count
        token_lengths.append(token_count)
        split = row.get(args.split_col, "")
        label = row.get(args.label_col, "")
        by_split.setdefault(split, []).append(token_count)
        by_label.setdefault(label, []).append(token_count)
        if not args.drop_over_limit or token_count <= args.max_tokens:
            kept.append(row)

    report = {
        "tokenizer": args.tokenizer_name,
        "max_tokens": args.max_tokens,
        "drop_over_limit": args.drop_over_limit,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(rows) - len(kept),
        "overall": summarize(token_lengths),
        "by_split": {key: summarize(value) for key, value in sorted(by_split.items())},
        "by_label": {key: summarize(value) for key, value in sorted(by_label.items())},
    }

    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = list(kept[0].keys()) if kept else list(rows[0].keys()) + ["token_count"]
        with output.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)

    if args.report_json:
        output = Path(args.report_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
