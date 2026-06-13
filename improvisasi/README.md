# Improvisasi Eksperimen

Pipeline improvisasi dengan Focal Loss dan EDA augmentation untuk menangani class imbalance pada klasifikasi sentimen BERT.

## Struktur Folder

```
improvisasi/
├── run_baseline.py        # Kondisi A: BERT + CrossEntropy (baseline)
├── run_eda.py             # Kondisi B: BERT + EDA augmentation
├── run_focal.py           # Kondisi C: BERT + Focal Loss
├── run_combined.py        # Kondisi D: BERT + Focal Loss + EDA (FULL)
├── run_deepfm_eval.py      # DeepFM evaluation (5 variants)
├── requirements.txt         # Dependencies
├── config.yaml            # Konfigurasi
│
├── dataset/
│   ├── IMDB/
│   │   └── imdb_reviews.csv       # Dataset IMDb reviews
│   └── MovieLens_OneM/
│       ├── ratings.dat             # MovieLens ratings
│       ├── users.dat              # MovieLens users
│       └── movies.dat             # MovieLens movies
│
├── logs/sentiment_scores/        # Sentiment scores (output)
│   ├── sentiment_a.csv
│   ├── sentiment_b.csv
│   ├── sentiment_c.csv
│   └── sentiment_d.csv
│
├── models/                        # Model checkpoints (output)
│   ├── bert_a/
│   ├── bert_b/
│   ├── bert_c_best/
│   ├── bert_d_best/
│   └── deepfm/
│
├── results/                      # Hasil evaluasi (output)
│   ├── condition_a/
│   ├── condition_b/
│   ├── condition_c/
│   ├── condition_d/
│   ├── deepfm_comparison.json
│   └── deepfm_comparison.png
│
└── README.md
```

## Persiapan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt')"
```

### 3. Pastikan Dataset Tersedia

Pastikan file-file berikut ada di folder `dataset/`:
- `dataset/IMDB/imdb_reviews.csv`
- `dataset/MovieLens_OneM/ratings.dat`
- `dataset/MovieLens_OneM/users.dat`
- `dataset/MovieLens_OneM/movies.dat`

## Cara Menjalankan

### Fase 1: BERT Training (4 Kondisi)

Jalankan secara berurutan:

```bash
# Kondisi A: Baseline (CrossEntropy)
python run_baseline.py

# Kondisi B: EDA augmentation
python run_eda.py

# Kondisi C: Focal Loss (grid search gamma)
python run_focal.py

# Kondisi D: Focal Loss + EDA (FULL - kombinasi terbaik)
python run_combined.py
```

Setiap script menghasilkan:
- `models/bert_X/best_model.pt` — model checkpoint
- `logs/sentiment_scores/sentiment_X.csv` — sentiment scores untuk DeepFM
- `results/condition_X/classification_report.txt` — laporan klasifikasi
- `results/condition_X/confusion_matrix.png` — confusion matrix
- `results/condition_X/training_curve.png` — kurva training

### Fase 2: DeepFM Evaluation

Setelah semua sentiment scores generated:

```bash
python run_deepfm_eval.py
```

Menghasilkan:
- `results/deepfm_comparison.json` — perbandingan metrik
- `results/deepfm_comparison.png` — visualisasi
- `models/deepfm/*.pt` — 5 model DeepFM

## Estimasi Waktu

| Script | Estimasi Waktu |
|--------|--------------|
| run_baseline.py | ~20-30 menit |
| run_eda.py | ~20-30 menit |
| run_focal.py | ~45-60 menit (3 gamma × 5 epoch) |
| run_combined.py | ~45-60 menit (3 gamma × 5 epoch) |
| run_deepfm_eval.py | ~30-45 menit (5 variants) |

Total: **~3-4 jam** pada GPU lokal (NVIDIA A40/4090)

## Kondisi Eksperimen

| Kondisi | Loss | Augmentasi | Keterangan |
|---------|------|------------|-------------|
| A | CrossEntropy | Tidak | Baseline untuk perbandingan |
| B | CrossEntropy | Ya (EDA) | Isolasi EDA saja |
| C | Focal Loss | Tidak | Isolasi Focal Loss saja |
| D | Focal Loss | Ya (EDA) | **Konfigurasi penuh** |

## Hasil yang Diharapkan

### Target

| Metrik | Target Paper | Target |
|--------|-------------|--------|
| BERT Macro F1 | 0.75 | 0.75 |
| DeepFM ROC-AUC | 0.8447 | 0.8447 |

### Hasil Baseline (Tanpa Improvisasi)

| Metrik | Hasil |
|--------|-------|
| BERT Macro F1 | 0.6342 |
| DeepFM ROC-AUC | 0.8704 |

### Hasil Setelah Improvisasi (Target)

| Metrik | Target |
|--------|--------|
| BERT Macro F1 | ~0.65+ |
| DeepFM ROC-AUC | ~0.80+ |
| Delta Sentiment D | Positif |

## Hardware yang Digunakan

- GPU: NVIDIA A40 (atau GPU dengan VRAM ≥16GB)
- RAM: ≥32GB
- OS: Linux/Windows
