import argparse
import csv
import re
from pathlib import Path


REFERENCE_HEADING_RE = re.compile(
    r"^\s*(daftar\s+pustaka|referensi|references|bibliography)\s*$",
    re.IGNORECASE,
)

NOISE_LINE_RE = re.compile(
    r"("
    r"^\s*\d+\s*$|"
    r"issn\s*:?\s*\d+|"
    r"e-issn|p-issn|"
    r"vol\.?\s*\d+|"
    r"volume\s+\d+|"
    r"no\.?\s*\d+|"
    r"jurnal\s+|"
    r"copyright|"
    r"doi\s*:|"
    r"email\s*:|"
    r"e-mail\s*:"
    r")",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract cleaner paragraph-like segments from Indonesian journal PDFs."
    )
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--recursive", action="store_true", help="Read PDFs recursively.")
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--max-per-doc", type=int, default=5)
    parser.add_argument("--top-margin-ratio", type=float, default=0.08)
    parser.add_argument("--bottom-margin-ratio", type=float, default=0.08)
    parser.add_argument("--include-abstract", action="store_true")
    parser.add_argument("--keep-english", action="store_true")
    return parser.parse_args()


def normalize_space(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w-]+\b", text, flags=re.UNICODE))


def parse_year_from_filename(filename: str) -> str:
    match = re.match(r"^\d+_(\d{4})_", filename)
    return match.group(1) if match else ""


def title_from_filename(path: Path) -> str:
    stem = re.sub(r"^\d+_\d{4}_", "", path.stem)
    stem = stem.replace("_", " ")
    stem = re.sub(r"\s+", " ", stem)
    return stem.strip().title()


def is_noise_text(text: str) -> bool:
    clean = normalize_space(text)
    if not clean:
        return True
    if len(clean) < 25:
        return True
    if NOISE_LINE_RE.search(clean) and word_count(clean) < 35:
        return True
    letters = len(re.findall(r"[A-Za-zÀ-ž]", clean))
    if letters == 0:
        return True
    non_letters = len(clean) - letters
    if non_letters / max(len(clean), 1) > 0.55:
        return True
    return False


def looks_english(text: str) -> bool:
    lowered = f" {text.lower()} "
    markers = [
        " the ",
        " and ",
        " this ",
        " that ",
        " research ",
        " study ",
        " result ",
        " abstract ",
        " keyword ",
    ]
    return sum(1 for marker in markers if marker in lowered) >= 3


def split_long_segment(text: str, min_words: int, max_words: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_words = 0
    for sentence in sentences:
        sw = word_count(sentence)
        if current and current_words + sw > max_words:
            chunk = normalize_space(" ".join(current))
            if word_count(chunk) >= min_words:
                chunks.append(chunk)
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sw
    if current:
        chunk = normalize_space(" ".join(current))
        if word_count(chunk) >= min_words:
            chunks.append(chunk)
    return chunks


def clean_metadata_line(line: str) -> str:
    line = normalize_space(line)
    line = re.sub(r"^(judul|title)\s*[:\-]\s*", "", line, flags=re.IGNORECASE)
    return line.strip(" -:;")


def looks_like_heading_line(text: str) -> bool:
    t = text.strip().lower()
    return bool(
        re.match(
            r"^(abstrak|abstract|pendahuluan|introduction|metode|metodologi|hasil|pembahasan|kesimpulan|references|daftar pustaka|\d+[\.\)]|i[\.\)])\b",
            t,
        )
    )


def sanitize_keywords(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"^(kata\s*kunci|keywords?)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .;:-")
    # Keep keyword-style chunk and cut before obvious paragraph continuation.
    text = re.split(r"\b(abstract|abstrak|pendahuluan|introduction|metode|metodologi|hasil|pembahasan|kesimpulan)\b", text, flags=re.IGNORECASE)[0]
    parts = [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
    # If separator is missing, split by short keyword n-grams heuristically.
    if len(parts) == 1 and len(parts[0].split()) > 14:
        words = parts[0].split()
        parts = [" ".join(words[i : i + 3]) for i in range(0, min(len(words), 18), 3)]
    # Limit to avoid grabbing full sentences.
    parts = parts[:8]
    out = ", ".join(parts).strip(" ,.;")
    # Last safety trim by word count.
    out_words = out.split()
    if len(out_words) > 24:
        out = " ".join(out_words[:24]).strip(" ,.;")
    return out


def extract_title_and_keywords(doc, path: Path) -> tuple[str, str]:
    # The downloaded corpus already encodes the article title in the filename.
    # For two-column PDFs, title extraction from layout blocks is often less stable
    # because author names, headers, or the first body sentence can be read first.
    title = title_from_filename(path)
    first_pages = []
    for page_index in range(min(2, len(doc))):
        first_pages.append(doc[page_index].get_text("text") or "")
    text = "\n".join(first_pages)
    lines = [clean_metadata_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line and not is_noise_text(line)]

    keyword_text = ""
    raw_lines = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
    for idx, line in enumerate(raw_lines):
        if re.match(r"^(kata\s*kunci|keywords?)\s*[:\-]?\s*", line, flags=re.IGNORECASE):
            candidate = line
            # Keywords sometimes continue to the next short line.
            if idx + 1 < len(raw_lines):
                next_line = raw_lines[idx + 1]
                if (
                    not looks_like_heading_line(next_line)
                    and len(next_line.split()) <= 12
                    and not re.search(r"[.!?]", next_line)
                ):
                    candidate = f"{candidate}, {next_line}"
            keyword_text = sanitize_keywords(candidate)
            break

    return normalize_space(title), normalize_space(keyword_text)


def page_blocks_layout_aware(page, top_margin_ratio: float, bottom_margin_ratio: float):
    width = float(page.rect.width)
    height = float(page.rect.height)
    top = height * top_margin_ratio
    bottom = height * (1.0 - bottom_margin_ratio)

    raw_blocks = page.get_text("blocks", sort=False)
    blocks = []
    for block in raw_blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        if y1 < top or y0 > bottom:
            continue
        text = normalize_space(text)
        if is_noise_text(text):
            continue
        center_x = (x0 + x1) / 2.0
        full_width = x0 < width * 0.18 and x1 > width * 0.82
        blocks.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "text": text,
                "center_x": center_x,
                "full_width": full_width,
            }
        )

    if not blocks:
        return []

    left = [b for b in blocks if (not b["full_width"] and b["center_x"] < width / 2.0)]
    right = [b for b in blocks if (not b["full_width"] and b["center_x"] >= width / 2.0)]
    full = [b for b in blocks if b["full_width"]]

    # Two-column journal pages are read left column first, then right column.
    # Full-width blocks such as title/abstract are kept before column blocks by vertical order.
    if len(left) >= 2 and len(right) >= 2:
        top_full = [b for b in full if b["y0"] < height * 0.35]
        bottom_full = [b for b in full if b["y0"] >= height * 0.35]
        ordered = (
            sorted(top_full, key=lambda b: (b["y0"], b["x0"]))
            + sorted(left, key=lambda b: (b["y0"], b["x0"]))
            + sorted(right, key=lambda b: (b["y0"], b["x0"]))
            + sorted(bottom_full, key=lambda b: (b["y0"], b["x0"]))
        )
    else:
        ordered = sorted(blocks, key=lambda b: (b["y0"], b["x0"]))
    return ordered


def section_guess(text: str) -> str:
    lowered = text.lower()
    if "abstrak" in lowered[:80] or "abstract" in lowered[:80]:
        return "abstract"
    if re.search(r"\bpendahuluan\b", lowered[:120]):
        return "pendahuluan"
    if re.search(r"\bmetode\b|\bmetodologi\b", lowered[:120]):
        return "metode"
    if re.search(r"\bhasil\b|\bpembahasan\b", lowered[:120]):
        return "hasil_pembahasan"
    if re.search(r"\bkesimpulan\b|\bsimpulan\b", lowered[:120]):
        return "kesimpulan"
    return ""


def extract_pdf(path: Path, args) -> list[dict]:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("Missing dependency PyMuPDF. Install with: pip install PyMuPDF") from exc

    doc = fitz.open(str(path))
    doc_title, doc_keywords = extract_title_and_keywords(doc, path)
    rows = []
    stop_doc = False

    for page_index in range(len(doc)):
        if stop_doc:
            break
        page = doc[page_index]
        blocks = page_blocks_layout_aware(page, args.top_margin_ratio, args.bottom_margin_ratio)

        for block_index, block in enumerate(blocks, start=1):
            text = block["text"]
            if REFERENCE_HEADING_RE.match(text):
                stop_doc = True
                break
            if not args.include_abstract and section_guess(text) == "abstract":
                continue
            if not args.keep_english and looks_english(text):
                continue

            wc = word_count(text)
            candidates = [text]
            if wc > args.max_words:
                candidates = split_long_segment(text, args.min_words, args.max_words)

            for candidate in candidates:
                candidate = normalize_space(candidate)
                wc2 = word_count(candidate)
                if wc2 < args.min_words or wc2 > args.max_words:
                    continue
                flags = []
                if re.search(r"\b(tabel|gambar|fig\.|table)\b", candidate, re.IGNORECASE):
                    flags.append("maybe_table_or_figure")
                if re.search(r"\[[0-9,\s-]+\]", candidate):
                    flags.append("contains_citation_marker")
                rows.append(
                    {
                        "segment_id": f"{path.stem}__p{page_index + 1}_b{block_index}_{len(rows) + 1}",
                        "doc_id": path.stem,
                        "source_file": path.name,
                        "doc_title": doc_title,
                        "doc_keywords": doc_keywords,
                        "year": parse_year_from_filename(path.name),
                        "page": page_index + 1,
                        "block": block_index,
                        "section_guess": section_guess(candidate),
                        "word_count": wc2,
                        "quality_flags": "|".join(flags),
                        "text": candidate,
                    }
                )
                if len(rows) >= args.max_per_doc:
                    stop_doc = True
                    break
            if stop_doc:
                break

    doc.close()
    return rows


def main():
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdfs = sorted(pdf_dir.glob(pattern), key=lambda p: p.name.lower())
    if not pdfs:
        raise SystemExit(f"No PDF files found in {pdf_dir}")

    all_rows = []
    for idx, pdf in enumerate(pdfs, start=1):
        try:
            rows = extract_pdf(pdf, args)
            all_rows.extend(rows)
            print(f"[ok] {idx}/{len(pdfs)} {pdf.name}: {len(rows)} segments")
        except Exception as exc:
            print(f"[warn] skip {pdf.name}: {exc}")

    fields = [
        "segment_id",
        "doc_id",
        "source_file",
        "doc_title",
        "doc_keywords",
        "year",
        "page",
        "block",
        "section_guess",
        "word_count",
        "quality_flags",
        "text",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[done] pdf_scanned={len(pdfs)}")
    print(f"[done] segments_written={len(all_rows)}")
    print(f"[done] output={output}")


if __name__ == "__main__":
    main()
