# Pipeline 2 - Deteksi Teks Generative AI pada Paper Jurnal Berbahasa Indonesia

Pipeline ini adalah rework dari folder `pipeline` lama agar sesuai dengan arah skripsi:

- Objek data: paper jurnal/artikel ilmiah berbahasa Indonesia.
- Tugas: klasifikasi biner `human` vs `ai`.
- Baseline: TF-IDF + Support Vector Machine.
- Model utama: IndoBERT.
- Evaluasi: in-distribution dan out-of-distribution.
- Output akhir: artifact model yang dapat dipakai oleh Streamlit.

Label yang digunakan konsisten:

- `0` = teks manusia.
- `1` = teks hasil Generative AI.

## Struktur File

- `01_extract_pdf_layout.py`  
  Ekstraksi paragraf dari PDF jurnal dengan pendekatan layout-aware untuk halaman dua kolom.

- `02_build_dataset.py`  
  Membuat dataset berlabel human/AI dari hasil ekstraksi. Kelas AI dibuat dari judul dan kata kunci jurnal, bukan dari parafrase paragraf manusia.

- `03_train_baseline_svm.py`  
  Training baseline TF-IDF + SVM, memilih threshold dari validation set, dan menyimpan metrik.

- `04_evaluate_model.py`  
  Evaluasi baseline atau IndoBERT pada split tertentu, termasuk OOD.

- `05_audit_tokens.py`  
  Audit panjang token IndoBERT dan opsional memfilter sampel yang melebihi batas token.

- `colab_train_indobert.py`  
  Script Colab-compatible untuk fine-tuning IndoBERT. File ini dapat diunggah ke Google Colab atau dijalankan sebagai script di notebook.

- `streamlit_app.py`  
  Prototipe antarmuka Streamlit yang membaca artifact IndoBERT atau baseline.

- `notebooks/IndoBERT_Training_Colab_Template.ipynb`  
  Template notebook Colab untuk training IndoBERT dan export model.

- `notebooks/SVM_Training_Colab_Template.ipynb`  
  Template notebook Colab untuk training baseline SVM.

## Instalasi Lokal

```powershell
cd E:\Tito\skripsi\pipeline2
pip install -r requirements-local.txt
```

## Alur Minimum

### 1. Ekstrak PDF jurnal manusia

```powershell
python .\01_extract_pdf_layout.py `
  --pdf-dir "E:\Tito\skripsi\jurnal" `
  --output "E:\Tito\skripsi\pipeline2\artifacts\human_segments.csv" `
  --recursive `
  --min-words 80 `
  --max-words 180 `
  --max-per-doc 5
```

### 2. Buat dataset human vs AI

Smoke test tanpa API:

```powershell
python .\02_build_dataset.py `
  --input "E:\Tito\skripsi\pipeline2\artifacts\human_segments.csv" `
  --output "E:\Tito\skripsi\pipeline2\artifacts\dataset.csv" `
  --generator template `
  --target-human 300 `
  --ood-ratio 0.15 `
  --ai-min-words 100 `
  --ai-max-words 150
```

Eksperimen final dengan API:

```powershell
$env:OPENAI_API_KEY="isi_api_key"

python .\02_build_dataset.py `
  --input "E:\Tito\skripsi\pipeline2\artifacts\human_segments.csv" `
  --output "E:\Tito\skripsi\pipeline2\artifacts\dataset.csv" `
  --generator openai `
  --model gpt-4o-mini `
  --target-human 3000 `
  --ai-variants 1 `
  --ood-ratio 0.15 `
  --ai-min-words 100 `
  --ai-max-words 150
```

Alternatif Gemini API:

```powershell
$env:GEMINI_API_KEY="isi_api_key"

python .\02_build_dataset.py `
  --input "E:\Tito\skripsi\pipeline2\artifacts\human_segments.csv" `
  --output "E:\Tito\skripsi\pipeline2\artifacts\dataset.csv" `
  --generator gemini `
  --model gemini-1.5-flash `
  --target-human 3000 `
  --ai-variants 1 `
  --ood-ratio 0.15 `
  --ai-min-words 100 `
  --ai-max-words 150 `
  --sleep-ms 250
```

Alternatif OpenRouter API:

```powershell
$env:OPENROUTER_API_KEY="isi_api_key"

python .\02_build_dataset.py `
  --input "E:\Tito\skripsi\pipeline2\artifacts\human_segments.csv" `
  --output "E:\Tito\skripsi\pipeline2\artifacts\dataset.csv" `
  --generator openrouter `
  --model openai/gpt-5.4-mini `
  --target-human 3000 `
  --ai-variants 1 `
  --ood-ratio 0.15 `
  --ai-min-words 100 `
  --ai-max-words 150 `
  --sleep-ms 250
```

Mode aman (disarankan) agar progress tetap tersimpan dan bisa lanjut jika credit habis atau koneksi putus:

```powershell
$env:OPENROUTER_API_KEY="isi_api_key"

python .\02_build_dataset.py `
  --input "E:\Tito\skripsi\pipeline2\artifacts\human_segments.csv" `
  --output "E:\Tito\skripsi\pipeline2\artifacts\dataset_openrouter.csv" `
  --generator openrouter `
  --model openai/gpt-5.4-mini `
  --target-human 3000 `
  --ai-variants 1 `
  --ood-ratio 0.15 `
  --ai-min-words 100 `
  --ai-max-words 150 `
  --sleep-ms 300 `
  --log-every-human 50 `
  --log-every-doc 10 `
  --resume
```

Keterangan singkat:
- `--resume`: lanjut dari file output yang sudah ada, tidak mengulang baris yang sama.
- Script menulis baris per baris dan flush langsung ke disk, jadi hasil yang sudah jadi tetap aman.
- Jika gagal di tengah jalan, jalankan ulang command yang sama (tetap pakai `--resume`).

### 3. Audit panjang token IndoBERT

```powershell
python .\05_audit_tokens.py `
  --input-csv "E:\Tito\skripsi\pipeline2\artifacts\dataset.csv" `
  --output-csv "E:\Tito\skripsi\pipeline2\artifacts\dataset_indobert256.csv" `
  --report-json "E:\Tito\skripsi\pipeline2\artifacts\token_report.json" `
  --max-tokens 256 `
  --drop-over-limit
```

Gunakan `dataset_indobert256.csv` untuk training jika ingin memastikan seluruh sampel masuk batas token IndoBERT `max_length=256`.

### 4. Train baseline TF-IDF + SVM

```powershell
python .\03_train_baseline_svm.py `
  --input-csv "E:\Tito\skripsi\pipeline2\artifacts\dataset_indobert256.csv" `
  --output-dir "E:\Tito\skripsi\pipeline2\artifacts\baseline_svm"
```

### 5. Train IndoBERT di Google Colab

Upload ke Google Drive:

- `pipeline2/artifacts/dataset_indobert256.csv`
- `pipeline2/colab_train_indobert.py`

Di Colab, jalankan:

```python
%run /content/drive/MyDrive/skripsi/pipeline2/colab_train_indobert.py \
  --input-csv /content/drive/MyDrive/skripsi/pipeline2/artifacts/dataset_indobert256.csv \
  --output-dir /content/drive/MyDrive/skripsi/pipeline2/artifacts/indobert \
  --model-name indobenchmark/indobert-base-p1 \
  --max-length 256 \
  --epochs 3 \
  --batch-size 8 \
  --export-zip
```

### 6. Evaluasi OOD

Baseline:

```powershell
python .\04_evaluate_model.py `
  --model-type baseline `
  --model-path "E:\Tito\skripsi\pipeline2\artifacts\baseline_svm\baseline_bundle.pkl" `
  --input-csv "E:\Tito\skripsi\pipeline2\artifacts\dataset_indobert256.csv" `
  --split-value ood `
  --output-json "E:\Tito\skripsi\pipeline2\artifacts\baseline_svm\ood_metrics.json"
```

IndoBERT:

```powershell
python .\04_evaluate_model.py `
  --model-type indobert `
  --model-path "E:\Tito\skripsi\pipeline2\artifacts\indobert\best_model" `
  --input-csv "E:\Tito\skripsi\pipeline2\artifacts\dataset_indobert256.csv" `
  --split-value ood `
  --output-json "E:\Tito\skripsi\pipeline2\artifacts\indobert\ood_metrics.json"
```

## Catatan Metodologis

- Split dilakukan berbasis `doc_id`, bukan paragraf, untuk mengurangi data leakage.
- Unit data dibatasi sebagai paragraf pendek agar sesuai dengan kapasitas token IndoBERT.
- Kelas AI dibuat dari judul dan kata kunci jurnal agar tidak menjadi parafrase langsung dari paragraf manusia.
- OOD dapat dibuat dari dokumen yang berbeda berdasarkan hash `doc_id`, atau ditentukan lebih eksplisit dengan `--ood-doc-regex`.
- Ekstraksi PDF dua kolom tetap perlu audit manual sampel. Script ini mengurangi risiko urutan kolom kacau, tetapi tidak bisa menjamin semua PDF bersih.
- Output model adalah indikator probabilistik, bukan bukti absolut pelanggaran akademik.

## Artifact Siap Pakai Tanpa Retrain

Setelah training selesai, pakai artifact berikut untuk inferensi:

- Baseline SVM: `artifacts/baseline_svm/baseline_bundle.pkl`
- IndoBERT: `artifacts/indobert/best_model/`
- Threshold IndoBERT: `artifacts/indobert/threshold.json`
- Export zip IndoBERT: `artifacts/indobert/best_model_export.zip`
