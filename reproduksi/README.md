# Reprodusi Eksperimen

Pipeline reproduksi eksperimen paper Agboola et al. (2026) pada Google Colab (Tesla T4 GPU).

## Catatan Penting

> **Reproduksi ini berjalan di Google Colab dengan GPU Tesla T4.**
>
> Clone repository ini ke Google Drive, buka di Google Colab, dan jalankan `10_run_all.py` atau step-by-step.

## Struktur Folder

```
reproduksi/
├── 01_setup_environment.py     # Setup environment, mount Google Drive
├── 02_bert_preprocessing.py    # Preprocess IMDb reviews untuk BERT
├── 03_bert_training.py        # Train BERT sentiment classifier
├── 04_bert_evaluation.py     # Evaluate BERT on test set
├── 05_generate_sentiment_features.py  # Generate sentiment scores
├── 06_deepfm_preprocessing.py    # Preprocess MovieLens untuk DeepFM
├── 07_deepfm_model.py           # DeepFM PyTorch model
├── 08_deepfm_training.py        # Train DeepFM (2 variants)
├── 09_deepfm_evaluation.py      # Evaluate DeepFM
├── 10_run_all.py               # Run all steps sequentially
│
├── kaggle.json                 # Kaggle API key (WAJIB di-isi sendiri)
│
├── dataset/
│   ├── IMDB/
│   │   └── imdb_reviews.csv    # Dataset IMDb reviews
│   └── MovieLens_OneM/
│       ├── README               # Cara download MovieLens
│       ├── ratings.dat
│       ├── users.dat
│       └── movies.dat
│
├── reprodusi-eksperimen/        # Output hasil (dibuat setelah run)
│   ├── logs/
│   ├── models/
│   └── results/
│
└── README.md
```

## Persiapan di Google Colab

### 1. Clone Repository

```bash
git https://github.com/FroggyJumpys/tugas-data-mining.git
cd tugas-data-mining/code/reproduksi
```

### 2. Setup Kaggle API

1. Buka [kaggle.com](https://www.kaggle.com) → Account → Create API Token
2. Download `kaggle.json`
3. Upload ke folder ini (Colab file browser)
4. Setup Kaggle:

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download Dataset MovieLens 1M

```bash
!kaggle datasets download -d jfreyberg/movielens-data1m -p dataset/MovieLens_OneM --unzip
```

Dataset IMDb (`imdb_reviews.csv`) sudah termasuk dalam repository.

### 4. Install Dependencies

```bash
!pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm nltk beautifulsoup4 lxml scipy PyYAML
```

## Cara Menjalankan

### Opsi 1: Run Semua Sekaligus

```bash
python 10_run_all.py
```

### Opsi 2: Step-by-Step (Direkomendasikan)

```bash
# Step 1: Setup environment
python 01_setup_environment.py

# Step 2: Preprocess BERT
python 02_bert_preprocessing.py

# Step 3: Train BERT
python 03_bert_training.py

# Step 4: Evaluate BERT
python 04_bert_evaluation.py

# Step 5: Generate sentiment scores
python 05_generate_sentiment_features.py

# Step 6: Preprocess DeepFM
python 06_deepfm_preprocessing.py

# Step 7: DeepFM model
python 07_deepfm_model.py

# Step 8: Train DeepFM
python 08_deepfm_training.py

# Step 9: Evaluate DeepFM
python 09_deepfm_evaluation.py
```

## Output yang Dihasilkan

```
reprodusi-eksperimen/
├── logs/
│   ├── bert_preprocessed/       # BERT tokenized data
│   ├── movie_sentiment_scores.csv
│   └── deepfm_preprocessed/     # DeepFM preprocessed data
│
├── models/
│   ├── bert_sentiment/
│   │   ├── best_model.pt         # BERT checkpoint
│   │   └── tokenizer/            # BERT tokenizer
│   └── deepfm/
│       ├── deepfm_without_sentiment.pt
│       └── deepfm_with_sentiment.pt
│
└── results/
    ├── deepfm_comparison.json
    ├── deepfm_comparison.png
    └── bert_classification_report.txt
```

## Estimasi Waktu di Colab (Tesla T4)

| Step | Estimasi |
|------|---------|
| 01 - Setup | ~1 menit |
| 02 - BERT Preprocessing | ~2 menit |
| 03 - BERT Training (5 epoch) | ~30-40 menit |
| 04 - BERT Evaluation | ~5 menit |
| 05 - Sentiment Scores | ~10 menit |
| 06 - DeepFM Preprocessing | ~5 menit |
| 07 - DeepFM Model | <1 menit |
| 08 - DeepFM Training (10 epoch × 2) | ~15-20 menit |
| 09 - DeepFM Evaluation | ~5 menit |

**Total: ~1.5-2 jam** di Google Colab Tesla T4

## Konfigurasi Hardware

| Komponen | Spesifikasi |
|----------|------------|
| GPU | NVIDIA Tesla T4 (16GB) |
| RAM | ~12GB (Colab free) |
| Disk | ~5GB gratis |

## Hasil Target

| Metrik | Target Paper | Hasil Repro |
|--------|-------------|-------------|
| BERT Accuracy | 75.81% | 78.36% ✅ |
| BERT Macro F1 | 0.75 | 0.6342 ❌ |
| DeepFM ROC-AUC | 0.8447 | 0.8704 ✅ |
| Delta Sentiment | +0.0151 | -0.0007 ❌ |

## Catatan Penting

1. **Macro F1 rendah** karena class imbalance parah pada IMDb dataset
2. **Delta sentimen negatif** karena sentiment tidak diskriminatif (global mean)
3. Lihat improvisasi/ untuk improvement dengan Focal Loss + EDA