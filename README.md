# Peningkatan Klasifikasi Sentimen pada Sistem Rekomendasi Film Berbasis BERT dan DeepFM Menggunakan Focal Loss dan Augmentasi Data untuk Mengatasi Ketidak-seimbangan Kelas

___

### Anggota Kelompok
- Ariefa Salsabilla Azzahra
- Asma Nur Patmi
- Dhea Novita Amelia Putri
- Muhammad Putra Ramadhan
___

## 1. Latar Belakang

Paper Agboola et al. (2026) mengusulkan sistem rekomendasi film hybrid yang mengintegrasikan analisis sentimen berbasis BERT ke dalam arsitektur DeepFM untuk meningkatkan akurasi rekomendasi. Paper mengklaim peningkatan ROC-AUC sebesar 1,51% (dari 0,8296 ke 0,8447) dengan penambahan fitur sentimen.

Tugas data mining ini bertujuan untuk **mereproduksi** hasil paper tersebut pada dataset MovieLens 1M + IMDb Reviews, kemudian **memperbaiki** keterbatasannya melalui Focal Loss dan EDA (Easy Data Augmentation) untuk menangani class imbalance pada klasifikasi sentimen.

## 2. Research Gap

Tiga research gap telah diidentifikasi:

1. **Class Imbalance Parah** — Macro F1 jauh di bawah target paper (0,6342 vs 0,75) akibat distribusi kelas yang tidak seimbang
2. **Delta Sentimen Negatif** — Sentimen tidak memberikan dampak positif pada DeepFM
3. **Computational Efficiency** — Model hybrid meningkatkan kompleksitas komputasi

## 3. Dataset

| Dataset | Records | Keterangan |
|---------|---------|-------------|
| IMDb Reviews | 57.534 | 5 kelas sentimen |
| MovieLens 1M | 1.000.209 | 6.040 users, 3.882 movies |

## 4. Metodologi

### Fase 1: Reproduksi Baseline
Replikasi pipeline paper dengan konfigurasi identik:
- BERT: `bert-base-uncased`, lr=2e-5, batch=16, epoch=5
- DeepFM: lr=1e-3, batch=256, DNN=[400,400,400], dropout=0.5

### Fase 2: Improvisasi
Penanganan class imbalance melalui 4 kondisi eksperimen:

| Kondisi | Loss | Augmentasi | Keterangan |
|---------|------|------------|-------------|
| A | CrossEntropy | Tidak | Baseline |
| B | CrossEntropy | Ya (EDA) | Isolasi EDA |
| C | Focal Loss | Tidak | Isolasi Focal Loss |
| D | Focal Loss | Ya (EDA) | **Konfigurasi penuh** |

## 5. Hasil Utama

### Reproduksi Baseline
| Metrik | Target Paper | Hasil | Status |
|--------|-------------|-------|--------|
| BERT Macro F1 | 0.75 | 0.6342 | ❌ |
| DeepFM ROC-AUC | 0.8447 | 0.8704 | ✅ |
| Delta Sentiment | +0.0151 | -0.0007 | ❌ |

### Improvisasi (Focal Loss + EDA)
| Metrik | Hasil | Perubahan vs Baseline |
|--------|-------|---------------------|
| BERT Macro F1 | 0.6433 | **+0.0091** ✅ |
| Delta Sentiment C | +0.0004 | **Positif** ✅ |
| Delta Sentiment D | +0.0012 | **Positif** ✅ |

## 6. Kesimpulan

Improvisasi berhasil meningkatkan Macro F1 dan mengubah delta sentimen menjadi positif untuk pertama kalinya. Focal Loss terbukti efektif dalam menangani class imbalance dan propagasinya ke downstream recommendation system. Hasil ini memvalidasi bahwa perbaikan fundamental pada kualitas sentimen memberikan dampak positif pada sistem rekomendasi film.

---

## AI Tools yang Digunakan
- Deepseek V4 Pro - untuk membantu dalam planning pembuatan dan debugging code.
- Claude Opus 4.8 - untuk eksekusi dan memberi masukan debugging code ketika ada error.

## Folder Code

- `reproduksi/` — Pipeline reproduksi baseline (Google Colab)
- `improvisasi/` — Pipeline perbaikan dengan Focal Loss + EDA (Local GPU)

## Paper Acuan

> Agboola A.O., Ladoja K.T., Onifade O.F.W. (2026). *Movie Recommendation System with Sentiment Analysis Using Deep Learning Algorithms*. Egyptian Informatics Journal, 33, 100905.
