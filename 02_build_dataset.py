import argparse
import csv
import hashlib
import math
import random
import re
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build human-vs-AI dataset for AI text detection.")
    parser.add_argument("--input", required=True, help="CSV from 01_extract_pdf_layout.py")
    parser.add_argument("--output", required=True, help="Output dataset CSV")
    parser.add_argument("--target-human", type=int, default=0, help="Sample N human rows; 0 means all")
    parser.add_argument("--ai-variants", type=int, default=1)
    parser.add_argument(
        "--generator",
        choices=["template", "openai", "gemini", "openrouter"],
        default="template",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--model-list",
        default="",
        help="Comma-separated model list for multi-model generation, e.g. 'openai/gpt-5.4-mini,google/gemini-2.0-flash'",
    )
    parser.add_argument(
        "--ai-model-mode",
        choices=["split", "full"],
        default="split",
        help="split: distribute AI samples across models; full: generate full ai_count for every model",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=650)
    parser.add_argument("--ai-min-words", type=int, default=100)
    parser.add_argument("--ai-max-words", type=int, default=150)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--ood-ratio", type=float, default=0.0)
    parser.add_argument("--ood-doc-regex", default="", help="Docs matching this regex are forced into OOD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-ms", type=int, default=250)
    parser.add_argument("--log-every-human", type=int, default=100, help="Log every N human rows processed")
    parser.add_argument("--log-every-doc", type=int, default=25, help="Log every N documents during AI generation")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output CSV if present")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w-]+\b", text, flags=re.UNICODE))


def stable_sort_key(key: str) -> str:
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def parse_model_list(raw_model_list: str, fallback_model: str) -> list[str]:
    models = [normalize_text(item) for item in (raw_model_list or "").split(",") if normalize_text(item)]
    if not models:
        models = [normalize_text(fallback_model)]
    return models


def build_doc_split_map(rows: list[dict], args) -> dict[str, str]:
    doc_ids = sorted({r.get("doc_id", "") for r in rows if r.get("doc_id", "")})
    if not doc_ids:
        raise SystemExit("No doc_id values found in input CSV.")

    forced_ood = set()
    if args.ood_doc_regex:
        forced_ood = {
            doc_id
            for doc_id in doc_ids
            if re.search(args.ood_doc_regex, doc_id, flags=re.IGNORECASE)
        }

    remaining = [doc_id for doc_id in doc_ids if doc_id not in forced_ood]
    remaining = sorted(remaining, key=lambda x: stable_sort_key(f"{args.seed}::{x}"))

    auto_ood_count = int(round(len(remaining) * args.ood_ratio))
    auto_ood = set(remaining[:auto_ood_count])
    remaining = remaining[auto_ood_count:]

    total = args.train_ratio + args.val_ratio + args.test_ratio
    train_ratio = args.train_ratio / total
    val_ratio = args.val_ratio / total

    n = len(remaining)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    n_test = max(0, n - n_train - n_val)

    split_map = {}
    for doc_id in forced_ood | auto_ood:
        split_map[doc_id] = "ood"
    for doc_id in remaining[:n_train]:
        split_map[doc_id] = "train"
    for doc_id in remaining[n_train : n_train + n_val]:
        split_map[doc_id] = "val"
    for doc_id in remaining[n_train + n_val : n_train + n_val + n_test]:
        split_map[doc_id] = "test"
    return split_map


def title_from_doc_id(doc_id: str) -> str:
    text = re.sub(r"^\d+_\d{4}_", "", doc_id)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().title()


def doc_metadata(rows: list[dict], doc_id: str) -> tuple[str, str]:
    for row in rows:
        if row.get("doc_id", "") == doc_id:
            title = normalize_text(row.get("doc_title", "")) or title_from_doc_id(doc_id)
            keywords = normalize_text(row.get("doc_keywords", ""))
            return title, keywords
    return title_from_doc_id(doc_id), ""


def template_generate(title: str, keywords: str, variant_index: int, min_words: int, max_words: int) -> str:
    keyword_part = f" dengan kata kunci {keywords}" if keywords else ""
    angles = [
        "latar belakang dan urgensi penelitian",
        "pendekatan metodologis yang dapat digunakan",
        "kontribusi dan implikasi akademik",
        "tantangan implementasi pada konteks terkait",
        "arah pengembangan penelitian lanjutan",
    ]
    angle = angles[(variant_index - 1) % len(angles)]
    text = (
        f"Topik mengenai {title}{keyword_part} dapat dikaji melalui {angle}. "
        "Pembahasan pada topik tersebut menekankan pentingnya perumusan masalah yang jelas, "
        "pemilihan metode yang sesuai, serta penyajian hasil yang dapat dipertanggungjawabkan secara akademik. "
        "Dalam konteks penelitian berbahasa Indonesia, uraian yang sistematis diperlukan agar hubungan antara konsep, "
        "data, dan kesimpulan dapat dipahami oleh pembaca. "
        "Selain itu, penggunaan istilah teknis perlu disesuaikan dengan ruang lingkup kajian sehingga argumentasi yang dibangun tetap konsisten dan relevan."
    )
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(" ,;") + "."
    elif len(words) < min_words:
        text += " Hal tersebut menunjukkan bahwa penulisan ilmiah tidak hanya menuntut kelengkapan informasi, tetapi juga ketepatan struktur dan koherensi antarbagian."
    return normalize_text(text)


def build_academic_prompt(
    title: str,
    keywords: str,
    variant_index: int,
    min_words: int,
    max_words: int,
) -> str:
    keyword_text = keywords if keywords else "tidak tersedia"
    focus_modes = [
        "latar belakang dan urgensi topik",
        "kerangka metodologis yang relevan",
        "implikasi praktis dan kontribusi ilmiah",
        "tantangan implementasi dan batasan studi",
        "arah pengembangan riset lanjutan",
    ]
    style_modes = [
        "Mulai tanpa frasa klise seperti 'Dalam era digital' atau 'Seiring perkembangan teknologi'.",
        "Gunakan variasi panjang kalimat dan hindari pola kalimat berulang.",
        "Gunakan diksi akademik natural, tidak terlalu generik atau templatis.",
        "Pertahankan koherensi logis antar kalimat dalam satu paragraf utuh.",
        "Hindari pembuka yang identik dengan variasi lain pada topik yang sama.",
    ]
    focus = focus_modes[(variant_index - 1) % len(focus_modes)]
    style = style_modes[(variant_index - 1) % len(style_modes)]

    return (
        "Tulis satu paragraf bergaya paper jurnal berbahasa Indonesia berdasarkan metadata artikel berikut. "
        "Paragraf harus formal, argumentatif, relevan dengan topik, dan tidak boleh menyebut bahwa teks dibuat dari metadata. "
        f"Panjang wajib {min_words}-{max_words} kata. "
        "Jangan membuat bullet, heading, abstrak, sitasi palsu, daftar pustaka, atau catatan tambahan. "
        "Keluarkan hanya satu paragraf final.\n\n"
        f"Judul: {title}\n"
        f"Kata kunci: {keyword_text}\n"
        f"Fokus variasi: {focus}\n"
        f"Aturan gaya: {style}\n"
        f"Nomor variasi: {variant_index}"
    )


def openai_generate(
    client,
    model: str,
    title: str,
    keywords: str,
    variant_index: int,
    min_words: int,
    max_words: int,
    temperature: float,
    max_output_tokens: int,
) -> str:
    prompt = build_academic_prompt(title, keywords, variant_index, min_words, max_words)
    response = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        input=prompt,
    )
    return normalize_text(response.output_text)


def gemini_generate(
    model,
    title: str,
    keywords: str,
    variant_index: int,
    min_words: int,
    max_words: int,
) -> str:
    prompt = build_academic_prompt(title, keywords, variant_index, min_words, max_words)
    response = model.generate_content(prompt)
    text = getattr(response, "text", None) or ""
    return normalize_text(text)


def main():
    args = parse_args()
    random.seed(args.seed)
    start_time = time.time()

    def log(message: str):
        elapsed = time.time() - start_time
        print(f"[{elapsed:8.1f}s] {message}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = parse_model_list(args.model_list, args.model)

    rows = list(csv.DictReader(input_path.open("r", encoding="utf-8-sig", newline="")))
    rows = [r for r in rows if normalize_text(r.get("text", ""))]
    if not rows:
        raise SystemExit("Input CSV is empty or has no text rows.")
    log(f"Loaded rows={len(rows)} from {input_path}")

    if args.target_human > 0 and args.target_human < len(rows):
        rows = random.sample(rows, args.target_human)
        log(f"Sampled target_human={len(rows)}")

    doc_split_map = build_doc_split_map(rows, args)
    split_counts = {}
    for _, split in doc_split_map.items():
        split_counts[split] = split_counts.get(split, 0) + 1
    log(f"Doc split map ready: {split_counts}")

    client = None
    openrouter_client = None
    gemini_model = None
    if args.generator == "openai":
        try:
            from openai import OpenAI
        except Exception as exc:
            raise SystemExit("Install OpenAI package first: pip install openai") from exc
        client = OpenAI()
        log(f"Generator=openai models={models}")
    elif args.generator == "openrouter":
        try:
            from openai import OpenAI
        except Exception as exc:
            raise SystemExit("Install OpenAI package first: pip install openai") from exc
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENROUTER_API_KEY first.")
        # OpenRouter uses an OpenAI-compatible API surface.
        openrouter_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        log(f"Generator=openrouter models={models}")
    elif args.generator == "gemini":
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise SystemExit(
                "Install Gemini SDK first: pip install google-generativeai"
            ) from exc
        api_key = (
            # Main expected env var
            __import__("os").environ.get("GEMINI_API_KEY")
            # Fallback for people who reuse Google naming
            or __import__("os").environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) first.")
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(models[0])
        log(f"Generator=gemini models={models}")
    else:
        log("Generator=template")

    fields = [
        "sample_id",
        "origin_segment_id",
        "doc_id",
        "source_file",
        "doc_title",
        "doc_keywords",
        "year",
        "page",
        "section_guess",
        "split",
        "label",
        "label_name",
        "generator",
        "generator_model",
        "word_count",
        "text",
    ]

    existing_keys = set()
    sample_id = 0
    existing_rows = 0
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8-sig", newline="") as file:
            for old in csv.DictReader(file):
                existing_rows += 1
                key = (old.get("origin_segment_id", ""), old.get("label_name", ""))
                existing_keys.add(key)
                try:
                    sample_id = max(sample_id, int(old.get("sample_id", "0") or 0))
                except Exception:
                    pass
        log(f"Resume mode: loaded existing rows={existing_rows}, max_sample_id={sample_id}")
    elif (not args.resume) and output_path.exists():
        output_path.unlink()
        log(f"Removed existing output: {output_path}")

    write_header = not output_path.exists()
    out_file = output_path.open("a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(out_file, fieldnames=fields)
    if write_header:
        writer.writeheader()
        out_file.flush()

    counts = {}

    def row_key(row):
        return (row.get("origin_segment_id", ""), row.get("label_name", ""))

    def write_row(row):
        nonlocal sample_id
        key = row_key(row)
        if key in existing_keys:
            return False
        sample_id += 1
        row["sample_id"] = sample_id
        writer.writerow(row)
        out_file.flush()
        existing_keys.add(key)
        k = (row.get("split", ""), row.get("label_name", ""))
        counts[k] = counts.get(k, 0) + 1
        return True

    rows_by_doc = {}
    for row in rows:
        rows_by_doc.setdefault(row.get("doc_id", ""), []).append(row)
    log(f"Unique docs for generation={len(rows_by_doc)}")

    log("Phase 1/2: writing human rows")
    for index, row in enumerate(rows, start=1):
        human_text = normalize_text(row.get("text", ""))
        doc_id = row.get("doc_id", "")
        split = doc_split_map.get(doc_id, "train")

        base_meta = {
            "origin_segment_id": row.get("segment_id", ""),
            "doc_id": doc_id,
            "source_file": row.get("source_file", ""),
            "doc_title": normalize_text(row.get("doc_title", "")) or title_from_doc_id(doc_id),
            "doc_keywords": normalize_text(row.get("doc_keywords", "")),
            "year": row.get("year", ""),
            "page": row.get("page", ""),
            "section_guess": row.get("section_guess", ""),
            "split": split,
        }

        write_row(
            {
                **base_meta,
                "label": 0,
                "label_name": "human",
                "generator": "human",
                "generator_model": "human",
                "word_count": word_count(human_text),
                "text": human_text,
            }
        )

        if args.log_every_human > 0 and index % args.log_every_human == 0:
            log(f"human_processed={index}/{len(rows)} current_sample_id={sample_id}")

    log("Phase 2/2: generating AI rows by document")
    for doc_index, (doc_id, doc_rows) in enumerate(sorted(rows_by_doc.items()), start=1):
        if not doc_id:
            continue
        title, keywords = doc_metadata(rows, doc_id)
        split = doc_split_map.get(doc_id, "train")
        human_count = len(doc_rows)
        ai_count = int(math.ceil(human_count * max(args.ai_variants, 1)))
        first_row = doc_rows[0]
        if args.ai_model_mode == "full":
            generation_plan = [(m, i) for m in models for i in range(1, ai_count + 1)]
        else:
            generation_plan = [(models[(i - 1) % len(models)], i) for i in range(1, ai_count + 1)]

        for model_name, variant_index in generation_plan:
            if args.generator == "openai":
                try:
                    ai_text = openai_generate(
                        client=client,
                        model=model_name,
                        title=title,
                        keywords=keywords,
                        variant_index=variant_index,
                        min_words=args.ai_min_words,
                        max_words=args.ai_max_words,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                    )
                except Exception as exc:
                    print(f"[warn] OpenAI generation failed for {doc_id} model {model_name} variant {variant_index}: {exc}")
                    ai_text = template_generate(title, keywords, variant_index, args.ai_min_words, args.ai_max_words)
            elif args.generator == "openrouter":
                try:
                    ai_text = openai_generate(
                        client=openrouter_client,
                        model=model_name,
                        title=title,
                        keywords=keywords,
                        variant_index=variant_index,
                        min_words=args.ai_min_words,
                        max_words=args.ai_max_words,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                    )
                except Exception as exc:
                    print(f"[warn] OpenRouter generation failed for {doc_id} model {model_name} variant {variant_index}: {exc}")
                    ai_text = template_generate(title, keywords, variant_index, args.ai_min_words, args.ai_max_words)
            elif args.generator == "gemini":
                try:
                    if model_name != models[0]:
                        gemini_model = genai.GenerativeModel(model_name)
                    ai_text = gemini_generate(
                        model=gemini_model,
                        title=title,
                        keywords=keywords,
                        variant_index=variant_index,
                        min_words=args.ai_min_words,
                        max_words=args.ai_max_words,
                    )
                except Exception as exc:
                    print(f"[warn] Gemini generation failed for {doc_id} model {model_name} variant {variant_index}: {exc}")
                    ai_text = template_generate(title, keywords, variant_index, args.ai_min_words, args.ai_max_words)
            else:
                ai_text = template_generate(title, keywords, variant_index, args.ai_min_words, args.ai_max_words)

            ai_text = normalize_text(ai_text)
            if not ai_text:
                continue

            model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)
            write_row(
                {
                    "origin_segment_id": f"{doc_id}__ai_title_keywords_{model_slug}_{variant_index}",
                    "doc_id": doc_id,
                    "source_file": first_row.get("source_file", ""),
                    "doc_title": title,
                    "doc_keywords": keywords,
                    "year": first_row.get("year", ""),
                    "page": "",
                    "section_guess": "ai_from_title_keywords",
                    "split": split,
                    "label": 1,
                    "label_name": "ai",
                    "generator": args.generator,
                    "generator_model": (
                        model_name
                        if args.generator in {"openai", "openrouter", "gemini"}
                        else "template_title_keywords_v1"
                    ),
                    "word_count": word_count(ai_text),
                    "text": ai_text,
                }
            )
            if args.generator in {"openai", "gemini", "openrouter"} and args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

        if args.log_every_doc > 0 and doc_index % args.log_every_doc == 0:
            log(f"ai_docs_processed={doc_index}/{len(rows_by_doc)} current_sample_id={sample_id}")

    out_file.close()

    # Recompute counts from final file so resume mode remains accurate.
    counts = {}
    final_rows = 0
    with output_path.open("r", encoding="utf-8-sig", newline="") as file:
        for item in csv.DictReader(file):
            final_rows += 1
            key = (item.get("split", ""), item.get("label_name", ""))
            counts[key] = counts.get(key, 0) + 1

    log("Dataset build completed. Split/label counts:")
    for key in sorted(counts):
        log(f"  {key[0]} / {key[1]} = {counts[key]}")
    log(f"final_rows={final_rows}")
    log(f"output={output_path}")


if __name__ == "__main__":
    main()
