import json
import re
from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Deteksi Teks AI IndoBERT",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stTextArea textarea { font-size: 15px; line-height: 1.5; border-radius: 6px; }
    .stButton > button { width: 100%; height: 2.8rem; border-radius: 6px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


BASE_ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def pick_default_model_dir() -> Path:
    candidates = [
        BASE_ARTIFACTS / "indobert_multi" / "best_model",
        BASE_ARTIFACTS / "indobert" / "best_model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def pick_default_threshold() -> Path:
    candidates = [
        BASE_ARTIFACTS / "indobert_multi" / "threshold.json",
        BASE_ARTIFACTS / "indobert" / "threshold.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_MODEL_DIR = pick_default_model_dir()
DEFAULT_THRESHOLD = pick_default_threshold()


@st.cache_resource
def load_indobert(model_dir: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_indobert_hf(repo_id: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def load_threshold(path: Path) -> float:
    if path.exists():
        return float(json.loads(path.read_text(encoding="utf-8")).get("threshold", 0.5))
    return 0.5


@st.cache_resource
def load_threshold_hf(repo_id: str) -> float:
    from huggingface_hub import hf_hub_download

    threshold_file = hf_hub_download(repo_id=repo_id, filename="threshold.json", repo_type="model")
    return float(json.loads(Path(threshold_file).read_text(encoding="utf-8")).get("threshold", 0.5))


def predict_ai_probability(text: str, tokenizer, model, device, max_length: int):
    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        prob_ai = torch.softmax(logits, dim=1)[0, 1].item()
    return prob_ai


def _chunk_by_words(text: str, target_words: int = 140, overlap_words: int = 25):
    words = text.split()
    if len(words) <= target_words:
        return [text]
    out = []
    start = 0
    step = max(1, target_words - overlap_words)
    while start < len(words):
        part = words[start : start + target_words]
        if not part:
            break
        out.append(" ".join(part))
        if start + target_words >= len(words):
            break
        start += step
    return out


def split_paragraphs(text: str, min_words: int = 40):
    raw = text or ""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Primary split: blank lines.
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", raw) if c.strip()]
    chunks = [re.sub(r"\s+", " ", c).strip() for c in chunks]

    # 2) If OCR/PDF extraction collapses into one giant chunk, try line-based regrouping.
    if len(chunks) <= 1 and len(raw.split()) > 250:
        lines = [re.sub(r"\s+", " ", l).strip() for l in raw.split("\n") if l.strip()]
        grouped = []
        buf = []
        wc = 0
        for line in lines:
            lw = len(line.split())
            if wc + lw > 160 and buf:
                grouped.append(" ".join(buf))
                buf = [line]
                wc = lw
            else:
                buf.append(line)
                wc += lw
        if buf:
            grouped.append(" ".join(buf))
        chunks = grouped if grouped else chunks

    # 3) Final pass: enforce min words and split overlong chunks.
    cleaned = []
    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        w = len(chunk.split())
        if w < min_words:
            continue
        if w > 220:
            for sub in _chunk_by_words(chunk, target_words=140, overlap_words=25):
                if len(sub.split()) >= min_words:
                    cleaned.append(sub)
        else:
            cleaned.append(chunk)
    return cleaned


def extract_text_from_pdf(uploaded_file) -> str:
    import fitz

    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        txt = page.get_text("text") or ""
        pages.append(txt)
    return "\n\n".join(pages).strip()


def extract_text_from_txt(uploaded_file) -> str:
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except Exception:
            return raw.decode("latin-1", errors="ignore")
    return str(raw)


st.title("Deteksi Teks Hasil Generative AI")
st.caption("Prototipe deteksi teks AI pada paper jurnal berbahasa Indonesia menggunakan IndoBERT.")

with st.sidebar:
    st.header("Konfigurasi")
    model_source = st.radio("Sumber model", options=["Lokal", "Hugging Face"], horizontal=True)
    model_dir = ""
    threshold_path = ""
    hf_repo_id = "Tomath02/proto1"
    threshold_default = 0.5

    if model_source == "Lokal":
        model_dir = st.text_input("Path model IndoBERT", value=str(DEFAULT_MODEL_DIR))
        threshold_path = st.text_input("Path threshold", value=str(DEFAULT_THRESHOLD))
        threshold_default = load_threshold(Path(threshold_path))
    else:
        hf_repo_id = st.text_input("HF repo id (model)", value="Tomath02/proto1")
        try:
            threshold_default = load_threshold_hf(hf_repo_id)
        except Exception:
            threshold_default = 0.5

    threshold = st.slider(
        "Threshold AI",
        min_value=0.05,
        max_value=0.95,
        value=float(threshold_default),
        step=0.01,
    )
    max_length = st.select_slider("Max token length", options=[128, 256, 384, 512], value=256)

left, right = st.columns([1.45, 1], gap="large")

with left:
    st.subheader("Input")
    input_mode = st.radio("Sumber input", options=["Teks Manual", "Upload File"], horizontal=True)
    text = ""
    paragraphs = []
    if input_mode == "Teks Manual":
        text = st.text_area(
            "Masukkan paragraf paper jurnal berbahasa Indonesia",
            height=320,
            placeholder="Tempel paragraf artikel ilmiah di sini...",
        )
        paragraphs = split_paragraphs(text, min_words=20)
        if text.strip() and not paragraphs:
            paragraphs = [re.sub(r"\s+", " ", text).strip()]
    else:
        uploaded_file = st.file_uploader("Upload paper (.pdf / .txt)", type=["pdf", "txt"])
        min_words_pdf = st.slider("Minimal kata per paragraf (file)", min_value=20, max_value=120, value=40, step=5)
        if uploaded_file is not None:
            try:
                suffix = Path(uploaded_file.name).suffix.lower()
                if suffix == ".pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif suffix == ".txt":
                    text = extract_text_from_txt(uploaded_file)
                else:
                    raise ValueError("Format file belum didukung.")
                if len(text.split()) < 20:
                    raise ValueError(
                        "Teks dari file sangat sedikit/nihil. Kemungkinan PDF berupa hasil scan gambar tanpa layer teks (butuh OCR)."
                    )
                paragraphs = split_paragraphs(text, min_words=min_words_pdf)
                st.caption(
                    f"Teks file terambil: {len(text.split())} kata. Paragraf valid untuk inferensi: {len(paragraphs)}."
                )
            except Exception as exc:
                st.error(f"Gagal ekstrak file: {exc}")
    analyze = st.button("Jalankan Deteksi", type="primary")

with right:
    st.subheader("Hasil")
    if not analyze:
        st.info("Masukkan teks atau upload PDF, lalu jalankan deteksi.")
    elif not paragraphs:
        st.warning("Input belum valid atau paragraf yang dapat dianalisis belum tersedia.")
    elif model_source == "Lokal" and not Path(model_dir).exists():
        st.error(f"Model tidak ditemukan: {model_dir}")
    else:
        try:
            if model_source == "Lokal":
                tokenizer, model, device = load_indobert(model_dir)
            else:
                tokenizer, model, device = load_indobert_hf(hf_repo_id)
            probs = [predict_ai_probability(p, tokenizer, model, device, max_length) for p in paragraphs]
            prob_ai = sum(probs) / len(probs)
            pred = 1 if prob_ai >= threshold else 0
            prob_human = 1.0 - prob_ai

            if pred == 1:
                st.markdown("<h2 style='color:#dc2626'>AI</h2>", unsafe_allow_html=True)
                st.caption("Dokumen terindikasi sebagai hasil Generative AI berdasarkan rerata probabilitas paragraf.")
            else:
                st.markdown("<h2 style='color:#15803d'>Human</h2>", unsafe_allow_html=True)
                st.caption("Dokumen terindikasi sebagai tulisan manusia berdasarkan rerata probabilitas paragraf.")

            st.metric("Probabilitas AI", f"{prob_ai:.2%}")
            st.progress(float(prob_ai))
            st.metric("Jumlah Paragraf Dianalisis", f"{len(paragraphs)}")
            st.dataframe(
                {
                    "Label": ["Human", "AI"],
                    "Probabilitas": [f"{prob_human:.4f}", f"{prob_ai:.4f}"],
                },
                hide_index=True,
                use_container_width=True,
            )
            if len(paragraphs) > 1:
                preview_rows = []
                for idx, (para, score) in enumerate(zip(paragraphs, probs), start=1):
                    preview_rows.append(
                        {
                            "Paragraf": idx,
                            "Prob_AI": round(score, 4),
                            "Pred": "AI" if score >= threshold else "Human",
                            "Preview": para[:180] + ("..." if len(para) > 180 else ""),
                        }
                    )
                st.dataframe(preview_rows, hide_index=True, use_container_width=True)
            st.caption("Hasil bersifat probabilistik dan perlu digunakan sebagai indikator, bukan vonis final.")
        except Exception as exc:
            st.error(f"Gagal memuat atau menjalankan model: {exc}")
