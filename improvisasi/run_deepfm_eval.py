"""
run_deepfm_eval.py — Fase 2: DeepFM Evaluation (5 Variants)
============================================================
Fully self-contained — tidak ada external modules/
Train dan evaluate DeepFM dengan 5 konfigurasi:
  1. Without sentiment (baseline)
  2. With sentiment (Condition A)
  3. With sentiment (Condition B)
  4. With sentiment (Condition C)
  5. With sentiment (Condition D)

Output:
  - results/deepfm_comparison.json
  - results/deepfm_comparison.png
  - models/deepfm/deepfm_without_sentiment.pt
  - models/deepfm/deepfm_with_sentiment_a.pt
  - models/deepfm/deepfm_with_sentiment_b.pt
  - models/deepfm/deepfm_with_sentiment_c.pt
  - models/deepfm/deepfm_with_sentiment_d.pt
"""

# ============================================================
# SECTION 1: IMPORTS
# ============================================================
import json
import os
import re
from pathlib import Path
from datetime import datetime
from json import JSONEncoder

import numpy as np


def make_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    return obj
import pandas as pd
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.amp import GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================
# SECTION 2: CONFIG — Matched to original working implementation
# ============================================================
SEED = 42

DEEPFM_EMBED_DIM = 32           # Plan 1B: from 16 → 32 (4x embedding capacity)
DEEPFM_DNN = [400, 400, 400]     # Original: [400,400,400] — proven architecture
DEEPFM_DROPOUT = 0.5             # Original: 0.5
DEEPFM_BATCH = 256
DEEPFM_EPOCHS = 20               # Plan 1A: from 10 → 20 (slower LR needs more steps)
DEEPFM_LR = 5e-4                # Plan 1A: from 1e-3 → 5e-4 (smoother convergence)
DEEPFM_PATIENCE = 5             # Plan 1A: from 3 → 5 (prevent premature stop with slower LR)
DEEPFM_LABEL_SMOOTHING = 0.05   # Improv: reduces overconfidence, closes generalization gap
DEEPFM_GRAD_CLIP_NORM = 1.0     # Improv: prevents gradient explosion with larger embed dims
RATING_THRESHOLD_POS = 4

DEEPFM_SPLIT = (0.60, 0.20, 0.20)

# Multi-hot genres (18 genres in MovieLens 1M)
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]  # train/val/test

# ============================================================
# SECTION 3: UTILITY FUNCTIONS
# ============================================================
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# SECTION 4: DEEPFM MODEL — Ported from original 07_deepfm_model.py
# ============================================================
class DeepFM(nn.Module):
    """
    DeepFM from reproduction experiment (07_deepfm_model.py).
    Architecture: Linear + FM (second-order) + DNN → Sigmoid output.
    """
    def __init__(self, field_config, embed_dim=DEEPFM_EMBED_DIM, dnn_hidden_dims=None,
                 dropout=DEEPFM_DROPOUT, use_sentiment=False):
        super().__init__()

        if dnn_hidden_dims is None:
            dnn_hidden_dims = DEEPFM_DNN

        self.field_config = field_config
        self.embed_dim = embed_dim
        self.use_sentiment = use_sentiment

        # ── Embedding Layer ─────────────────────────────────────────────
        sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip_code']
        self.embeddings = nn.ModuleDict()
        for feat_name in sparse_features:
            num_embeddings = field_config['feature_dims'][feat_name]
            self.embeddings[feat_name] = nn.Embedding(num_embeddings, embed_dim)

        # ── Linear Component (first-order) ──────────────────────────────
        self.linear_embeddings = nn.ModuleDict()
        for feat_name in sparse_features:
            num_embeddings = field_config['feature_dims'][feat_name]
            self.linear_embeddings[feat_name] = nn.Embedding(num_embeddings, 1)

        # Genres linear (multi-hot → scalar)
        self.genres_linear = nn.Linear(field_config['feature_dims']['genres'], 1, bias=True)

        # Sentiment linear (dense feature)
        if use_sentiment:
            self.sentiment_linear = nn.Linear(1, 1, bias=True)

        # ── DNN Component ───────────────────────────────────────────────
        num_sparse_features = len(sparse_features)  # 6
        num_genre_features = field_config['feature_dims']['genres']  # 18
        dnn_input_dim = num_sparse_features * embed_dim + num_genre_features
        if use_sentiment:
            dnn_input_dim += 1

        self.dnn = nn.ModuleList()
        prev_dim = dnn_input_dim
        for hidden_dim in dnn_hidden_dims:
            self.dnn.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.dnn_dropout = nn.Dropout(dropout)
        self.dnn_output = nn.Linear(prev_dim, 1)

        # ── Weight Initialization ────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, sparse_features, dense_features=None):
        batch_size = sparse_features['user_id'].size(0)
        sparse_feat_names = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip_code']

        # ── Linear Component ────────────────────────────────────────────
        linear_output = sum(
            self.linear_embeddings[fn](sparse_features[fn]).squeeze(-1)
            for fn in sparse_feat_names
        )
        linear_output += self.genres_linear(sparse_features['genres']).squeeze(-1)
        if self.use_sentiment and dense_features is not None and 'sentiment' in dense_features:
            linear_output += self.sentiment_linear(dense_features['sentiment']).squeeze(-1)

        # ── FM Component (second-order) ─────────────────────────────────
        embed_list = [self.embeddings[fn](sparse_features[fn]) for fn in sparse_feat_names]
        embed_matrix = torch.stack(embed_list, dim=1)  # (batch, 6, embed_dim)

        first_order = embed_matrix.sum(dim=1)            # (batch, embed_dim)
        sum_square = (embed_matrix.sum(dim=1)) ** 2      # (batch, embed_dim)
        square_sum = (embed_matrix ** 2).sum(dim=1)      # (batch, embed_dim)
        second_order = 0.5 * (sum_square - square_sum)   # (batch, embed_dim)
        fm_output = first_order + second_order           # (batch, embed_dim)

        # ── DNN Component ────────────────────────────────────────────────
        embed_flat = embed_matrix.view(batch_size, -1)   # (batch, 6*embed_dim)
        dnn_input = torch.cat([embed_flat, sparse_features['genres']], dim=1)
        if self.use_sentiment and dense_features is not None and 'sentiment' in dense_features:
            dnn_input = torch.cat([dnn_input, dense_features['sentiment']], dim=1)

        for layer in self.dnn:
            dnn_input = F.relu(layer(dnn_input))
            dnn_input = self.dnn_dropout(dnn_input)

        dnn_output = self.dnn_output(dnn_input).squeeze(-1)  # (batch,)

        # ── Combine (raw logit, sigmoid fused into BCEWithLogitsLoss) ──
        output = linear_output + fm_output.sum(dim=1) + dnn_output
        return output


class DeepFMDataset(Dataset):
    """Dataset matching original 07_deepfm_model.py format."""
    def __init__(self, data_dict, use_sentiment=True):
        self.data = data_dict
        self.use_sentiment = use_sentiment

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):
        sparse_features = {
            'user_id': torch.tensor(self.data['user_id'][idx], dtype=torch.long),
            'movie_id': torch.tensor(self.data['movie_id'][idx], dtype=torch.long),
            'gender': torch.tensor(self.data['gender'][idx], dtype=torch.long),
            'age': torch.tensor(self.data['age'][idx], dtype=torch.long),
            'occupation': torch.tensor(self.data['occupation'][idx], dtype=torch.long),
            'zip_code': torch.tensor(self.data['zip_code'][idx], dtype=torch.long),
            'genres': torch.tensor(self.data['genres'][idx], dtype=torch.float32)
        }
        dense_features = None
        if self.use_sentiment and self.data.get('sentiment') is not None:
            dense_features = {
                'sentiment': torch.tensor([self.data['sentiment'][idx]], dtype=torch.float32)
            }
        label = torch.tensor(self.data['labels'][idx], dtype=torch.float32)
        return sparse_features, dense_features, label


def collate_fn(batch):
    """Custom collate — matches original 07_deepfm_model.py."""
    sparse_list, dense_list, labels_list = [], [], []
    for sparse, dense, label in batch:
        sparse_list.append(sparse)
        dense_list.append(dense)
        labels_list.append(label)

    # Stack sparse features
    stacked_sparse = {}
    for key in sparse_list[0].keys():
        stacked_sparse[key] = torch.stack([s[key] for s in sparse_list]).squeeze(-1)

    # Stack dense features
    stacked_dense = None
    if dense_list[0] is not None:
        stacked_dense = {}
        for key in dense_list[0].keys():
            stacked_dense[key] = torch.stack([d[key] for d in dense_list])

    labels = torch.stack(labels_list)
    return stacked_sparse, stacked_dense, labels


# ============================================================
# SECTION 5: DATA LOADING
# ============================================================
def load_movielens_data(dataset_path):
    """Load MovieLens 1M — matching original column names."""
    ratings_file = dataset_path / "ratings.dat"
    users_file = dataset_path / "users.dat"
    movies_file = dataset_path / "movies.dat"

    # Original column names from 06_deepfm_preprocessing.py
    ratings_df = pd.read_csv(ratings_file, sep='::', engine='python',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'], header=None)
    users_df = pd.read_csv(users_file, sep='::', engine='python',
                            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'], header=None)
    movies_df = pd.read_csv(movies_file, sep='::', engine='python',
                             names=['movie_id', 'title', 'genres'], header=None, encoding='latin-1')

    return ratings_df, users_df, movies_df


def encode_categorical(df, column):
    """Encode categorical column to 0-indexed integer IDs. Returns (df, num_unique)."""
    unique_values = df[column].unique()
    value_to_id = {v: i for i, v in enumerate(unique_values)}
    df[column + '_encoded'] = df[column].map(value_to_id)
    return df, len(unique_values)


def parse_genres_multi_hot(genres_str):
    """Parse genre string and return multi-hot vector (18 dims)."""
    if pd.isna(genres_str):
        return [0] * len(ALL_GENRES)
    genres = genres_str.split('|')
    return [1 if g in genres else 0 for g in ALL_GENRES]


def preprocess_deepfm_data(ratings_df, users_df, movies_df, sentiment_df=None,
                            rating_threshold_pos=RATING_THRESHOLD_POS, random_seed=SEED):
    """
    Preprocess MovieLens data — ported from original 06_deepfm_preprocessing.py.
    Key differences from our previous version:
    - Multi-hot genres (18 dims) instead of single genre_idx
    - zip_code included (3439 values)
    - 0-indexed IDs (encoded)
    - Returns field_config for model initialization
    """
    np.random.seed(random_seed)

    # ── Encode user features (work on copy to avoid in-place mutation across calls) ──
    users_df = users_df.copy()
    users_df, _ = encode_categorical(users_df, 'gender')
    users_df['zip_code'] = users_df['zip_code'].astype(str)
    users_df, _ = encode_categorical(users_df, 'zip_code')

    # ── Encode movie_id (work on copy) ─────────────────────────────────────
    movies_df = movies_df.copy()
    movies_df, _ = encode_categorical(movies_df, 'movie_id')
    movies_df['genre_vector'] = movies_df['genres'].apply(parse_genres_multi_hot)

    # ── Merge datasets ────────────────────────────────────────────────────
    df = ratings_df.merge(users_df, on='user_id', how='left')
    df = df.merge(movies_df[['movie_id', 'movie_id_encoded', 'genre_vector']], on='movie_id', how='left')

    # ── Merge sentiment per movie ────────────────────────────────────────
    if sentiment_df is not None:
        sent_col = 'sentiment_score' if 'sentiment_score' in sentiment_df.columns else \
                   'sentiment' if 'sentiment' in sentiment_df.columns else \
                   [c for c in sentiment_df.columns if c not in ['user_id', 'movieId', 'review_text']][0]
        sent_renamed = sentiment_df.rename(columns={sent_col: 'sentiment'})
        merge_key = 'movie_id' if 'movie_id' in sent_renamed.columns else \
                    'movieId' if 'movieId' in sent_renamed.columns else None
        if merge_key:
            df = df.merge(sent_renamed[[merge_key, 'sentiment']], on=merge_key, how='left')
        df['sentiment'] = df['sentiment'].fillna(0.5)
    else:
        df['sentiment'] = 0.5

    # ── Binarize ratings ─────────────────────────────────────────────────
    labels = (df['rating'] >= rating_threshold_pos).astype(int).values

    # ── Encode age with pd.cut bins (matching original) ───────────────────
    age_bins = [0, 18, 25, 35, 45, 55, 100]
    age_labels = list(range(len(age_bins) - 1))
    df['age_encoded'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True).astype(int)

    # ── Encode occupation (enumerate unique values) ───────────────────────
    df, _ = encode_categorical(df, 'occupation')

    # ── Build field_config ────────────────────────────────────────────────
    # user_id: NOT encoded — use raw column from ratings (already 0-indexed in MovieLens)
    max_user_id = int(df['user_id'].max() + 1)
    # movie_id: encoded in movies_df → movie_id_encoded after merge
    max_movie_id = int(df['movie_id_encoded'].max() + 1)
    max_gender_id = int(df['gender_encoded'].max() + 1)
    max_age_id = int(df['age_encoded'].max() + 1)
    max_occupation_id = int(df['occupation_encoded'].max() + 1)
    max_zipcode_id = int(df['zip_code_encoded'].max() + 1)
    num_genres = len(ALL_GENRES)

    field_config = {
        'feature_dims': {
            'user_id': max_user_id,
            'movie_id': max_movie_id,
            'gender': max_gender_id,
            'age': max_age_id,
            'occupation': max_occupation_id,
            'zip_code': max_zipcode_id,
            'genres': num_genres,
        }
    }

    # ── Build data arrays ─────────────────────────────────────────────────
    data = {
        'user_id': df['user_id'].values,
        'movie_id': df['movie_id_encoded'].values,
        'gender': df['gender_encoded'].values,
        'age': df['age_encoded'].values,
        'occupation': df['occupation_encoded'].values,
        'zip_code': df['zip_code_encoded'].values,
        'genres': np.array(df['genre_vector'].tolist()),
        'sentiment': df['sentiment'].values,
        'labels': labels
    }

    return data, field_config


# ============================================================
# SECTION 6: TRAINING & EVALUATION
# ============================================================
def train_deepfm_variant(train_data, val_data, field_config, use_sentiment,
                          variant_name, device, n_gpus, model_path=None, use_amp=True):
    """
    Train DeepFM variant — ported from original 08_deepfm_training.py.
    Key: uses BCELoss (model outputs sigmoid), lr=1e-3, early stopping on val_loss.
    """
    train_ds = DeepFMDataset(train_data, use_sentiment=use_sentiment)
    val_ds = DeepFMDataset(val_data, use_sentiment=use_sentiment)

    train_loader = DataLoader(train_ds, batch_size=DEEPFM_BATCH, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=DEEPFM_BATCH, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    model = DeepFM(
        field_config=field_config,
        embed_dim=DEEPFM_EMBED_DIM,
        dnn_hidden_dims=DEEPFM_DNN,
        dropout=DEEPFM_DROPOUT,
        use_sentiment=use_sentiment
    ).to(device)

    print("  DeepFM model loaded.")

    # BCEWithLogitsLoss — combines sigmoid+BCE, safe under AMP autocast
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=DEEPFM_LR)
    scaler = GradScaler('cuda') if (use_amp and torch.cuda.is_available()) else None

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    training_history = []

    for epoch in tqdm(range(DEEPFM_EPOCHS), desc=f"  [{variant_name}] Epoch", unit="epoch", position=0):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for sparse, dense, labels in tqdm(train_loader, desc="Training", leave=False, position=1):
            sparse = {k: v.to(device) for k, v in sparse.items()}
            if dense is not None:
                dense = {k: v.to(device) for k, v in dense.items()}
            labels = labels.to(device)

            optimizer.zero_grad()

            if DEEPFM_LABEL_SMOOTHING > 0:
                target = labels * (1 - DEEPFM_LABEL_SMOOTHING) + 0.5 * DEEPFM_LABEL_SMOOTHING
            else:
                target = labels

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    preds = model(sparse, dense)
                loss = criterion(preds, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), DEEPFM_GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(sparse, dense)
                loss = criterion(preds, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), DEEPFM_GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_roc_auc = roc_auc_score(np.array(all_labels), np.array(all_preds))

        # ── Validate ──────────────────────────────────────────────────────
        val_metrics = evaluate_deepfm(model, val_loader, device)

        print(f"  [{variant_name}] Epoch {epoch+1}/{DEEPFM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} AUC: {train_roc_auc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['roc_auc']:.4f}")

        # ── Early stopping on val_loss (matching original) ─────────────────
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= DEEPFM_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)

    return model, {'best_val_loss': best_val_loss, 'training_history': training_history, 'variant_name': variant_name}


def evaluate_deepfm(model, test_loader, device):
    """
    Evaluate DeepFM — model outputs sigmoid (probabilities).
    Ported from original 09_deepfm_evaluation.py.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False, position=1):
            sparse, dense, labels = batch
            sparse = {k: v.to(device) for k, v in sparse.items()}
            if dense is not None:
                dense = {k: v.to(device) for k, v in dense.items()}
            labels = labels.to(device)

            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    preds = model(sparse, dense)
            else:
                preds = model(sparse, dense)

            all_preds.extend(preds.cpu().float().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    preds_proba = 1 / (1 + np.exp(-all_preds))  # sigmoid

    preds_binary = (preds_proba >= 0.5).astype(int)
    labels_binary = all_labels.astype(int)

    try:
        roc_auc = roc_auc_score(labels_binary, preds_proba)
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(labels_binary, preds_proba)
    except ValueError:
        pr_auc = 0.0

    bal_acc = balanced_accuracy_score(labels_binary, preds_binary)
    loss = nn.BCEWithLogitsLoss()(torch.tensor(all_preds), torch.tensor(all_labels).float()).item()

    top_k_metrics = compute_topk_metrics(preds_proba, all_labels, k=10)

    return {
        'loss': loss,
        'balanced_accuracy': bal_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision_at_10': top_k_metrics['precision_at_10'],
        'recall_at_10': top_k_metrics['recall_at_10'],
        'map_at_10': top_k_metrics['map_at_10'],
        'ndcg_at_10': top_k_metrics['ndcg_at_10']
    }


def compute_topk_metrics(probs, labels, k=10):
    n = len(probs)
    sorted_indices = np.argsort(probs)[::-1][:k]
    top_k_labels = labels[sorted_indices]

    precision = top_k_labels.sum() / k

    total_positives = labels.sum()
    recall = top_k_labels.sum() / total_positives if total_positives > 0 else 0.0

    ap = 0.0
    num_hits = 0
    for i, label in enumerate(top_k_labels):
        if label == 1:
            num_hits += 1
            ap += num_hits / (i + 1)
    map_score = ap / max(num_hits, 1)

    dcg = 0.0
    for i, label in enumerate(top_k_labels):
        dcg += label / np.log2(i + 2)

    ideal_sorted = np.sort(labels)[::-1][:k]
    idcg = 0.0
    for i, label in enumerate(ideal_sorted):
        idcg += label / np.log2(i + 2)

    ndcg = dcg / max(idcg, 1e-8)

    return {
        'precision_at_10': precision,
        'recall_at_10': recall,
        'map_at_10': map_score,
        'ndcg_at_10': ndcg
    }


def plot_deepfm_comparison(results, save_path):
    metrics = ['roc_auc', 'pr_auc', 'balanced_accuracy', 'loss']
    variants = ['without_sentiment', 'with_sentiment_a', 'with_sentiment_b', 'with_sentiment_c', 'with_sentiment_d']
    variant_labels = ['No Sentiment', 'Cond. A (CE)', 'Cond. B (EDA)', 'Cond. C (FL)', 'Cond. D (FL+EDA)']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[v].get(metric, 0) for v in variants]
        bars = ax.bar(variant_labels, values, color=colors, edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_title(metric.replace('_', ' ').upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)

    plt.suptitle('DeepFM Comparison: With vs Without Sentiment Features\n'
                 'Conditions A/B/C/D (from improvement proposal)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison chart saved: {save_path}")


# ============================================================
# SECTION 0: ENVIRONMENT DETECTION
# ============================================================
def get_environment():
    """Auto-detect Kaggle vs Colab vs Local environment."""
    import os
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    elif os.path.exists("/content/drive"):
        return "colab"
    else:
        return "local"

ENV = get_environment()
print(f"[ENV] Detected environment: {ENV}")

# ============================================================
# SECTION 7: MAIN EXECUTION
# ============================================================
def main():
    print("=" * 60)
    print("FASE 2 — DeepFM Evaluation (5 Variants)")
    print("=" * 60)
    print()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Enable cuDNN benchmark for faster training with fixed input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if n_gpus > 1:
        print(f"{n_gpus} GPUs detected — using single GPU for stability")
        n_gpus = 1
    print(f"Device: {device}")
    print()

    result_dir = Path("results")
    model_dir = Path("models/deepfm")
    sentiment_dir = Path("logs/sentiment_scores")
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect dataset path
    if os.path.exists("/kaggle/input"):
        dataset_path = Path("/kaggle/input/datasets/froggyjumpys/datase/MovieLens_OneM")
    elif os.path.exists("/workspace/dataset/MovieLens_OneM"):
        dataset_path = Path("/workspace/dataset/MovieLens_OneM")
    elif os.path.exists("/content/drive"):
        dataset_path = Path("/content/drive/MyDrive/dataset/MovieLens_OneM")
    else:
        dataset_path = Path("dataset/MovieLens_OneM")  # fallback

    if not (dataset_path / "ratings.dat").exists():
        print(f"[ERROR] File not found: {dataset_path / 'ratings.dat'}")
        print(f"[ERROR] Available files: {list(dataset_path.glob('*')) if dataset_path.exists() else 'Directory not found'}")
        return
    print(f"[DATASET] Using: {dataset_path}")

    ratings_df, users_df, movies_df = load_movielens_data(dataset_path)
    print(f"  Ratings: {len(ratings_df)}, Users: {len(users_df)}, Movies: {len(movies_df)}")
    print()

    # 2. Load sentiment scores from all 4 conditions
    print("[2/4] Loading sentiment scores from conditions A/B/C/D...")
    sentiment_files = {
        'a': sentiment_dir / "sentiment_a.csv",
        'b': sentiment_dir / "sentiment_b.csv",
        'c': sentiment_dir / "sentiment_c.csv",
        'd': sentiment_dir / "sentiment_d.csv",
    }

    sentiment_dfs = {}
    for cond, path in sentiment_files.items():
        if path.exists():
            sentiment_dfs[cond] = pd.read_csv(path)
            print(f"  Condition {cond.upper()}: {len(sentiment_dfs[cond])} samples loaded")
        else:
            print(f"  WARNING: {path} not found — please run BERT scripts first")

    if len(sentiment_dfs) < 4:
        print("ERROR: Not all sentiment files found. Please run all BERT scripts first.")
        return
    print()

    # 3. Preprocess data
    print("[3/4] Preprocessing DeepFM data...")

    train_ratings, temp_ratings = train_test_split(
        ratings_df, test_size=(DEEPFM_SPLIT[1] + DEEPFM_SPLIT[2]),
        random_state=SEED
    )
    val_ratio = DEEPFM_SPLIT[1] / (DEEPFM_SPLIT[1] + DEEPFM_SPLIT[2])
    val_ratings, test_ratings = train_test_split(
        temp_ratings, test_size=(1 - val_ratio),
        random_state=SEED
    )
    print(f"  Ratings split: Train={len(train_ratings)}, Val={len(val_ratings)}, Test={len(test_ratings)}")

    base_train, field_config = preprocess_deepfm_data(train_ratings, users_df, movies_df, sentiment_df=None, random_seed=SEED)
    base_val, _ = preprocess_deepfm_data(val_ratings, users_df, movies_df, sentiment_df=None, random_seed=SEED)
    base_test, _ = preprocess_deepfm_data(test_ratings, users_df, movies_df, sentiment_df=None, random_seed=SEED)

    print(f"  Field dims: {field_config['feature_dims']}")
    print()

    # 4. Train and evaluate 5 variants
    print("[4/4] Training and evaluating 5 DeepFM variants...")
    print()

    all_results = {}

    # Variant 0: Without sentiment
    print("  [Variant 0/4] DeepFM WITHOUT sentiment...")
    model_path = model_dir / "deepfm_without_sentiment.pt"
    model, result = train_deepfm_variant(
        base_train, base_val, field_config,
        use_sentiment=False,
        variant_name="without_sentiment",
        device=device, n_gpus=n_gpus, model_path=model_path
    )

    test_ds = DeepFMDataset(base_test, use_sentiment=False)
    test_loader = DataLoader(test_ds, batch_size=DEEPFM_BATCH, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    test_metrics = evaluate_deepfm(model, test_loader, device)
    all_results['without_sentiment'] = test_metrics
    print(f"    ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}")
    print()

    # Variants 1-4: With sentiment (A/B/C/D)
    conditions = ['a', 'b', 'c', 'd']

    for cond in tqdm(conditions, desc="DeepFM Variants", unit="variant", position=0):
        cond_key = f"with_sentiment_{cond}"
        print(f"  [Variant {conditions.index(cond)+1}/4] DeepFM WITH sentiment (Condition {cond.upper()})...")

        sent_df = sentiment_dfs[cond].copy()
        sent_col = 'sentiment_score' if 'sentiment_score' in sent_df.columns else \
                   'sentiment' if 'sentiment' in sent_df.columns else \
                   [c for c in sent_df.columns if c not in ['review_text']][0]
        global_sentiment = sent_df[sent_col].mean()
        print(f"    Global mean sentiment (Cond {cond.upper()}): {global_sentiment:.4f}")

        # Use per-movie (not per-rating) to avoid many-to-many merge explosion
        sent_per_movie = movies_df[['movie_id']].copy()
        sent_per_movie['sentiment'] = global_sentiment

        train_sent, _ = preprocess_deepfm_data(train_ratings, users_df, movies_df,
                                                sentiment_df=sent_per_movie, random_seed=SEED)
        val_sent, _ = preprocess_deepfm_data(val_ratings, users_df, movies_df,
                                              sentiment_df=sent_per_movie, random_seed=SEED)
        test_sent, _ = preprocess_deepfm_data(test_ratings, users_df, movies_df,
                                               sentiment_df=sent_per_movie, random_seed=SEED)

        model_path = model_dir / f"deepfm_with_sentiment_{cond}.pt"
        model, _ = train_deepfm_variant(
            train_sent, val_sent, field_config,
            use_sentiment=True,
            variant_name=cond_key,
            device=device, n_gpus=n_gpus, model_path=model_path
        )

        test_ds_sent = DeepFMDataset(test_sent, use_sentiment=True)
        test_loader_sent = DataLoader(test_ds_sent, batch_size=DEEPFM_BATCH, shuffle=False,
                                      collate_fn=collate_fn, num_workers=0)
        test_metrics = evaluate_deepfm(model, test_loader_sent, device)
        all_results[cond_key] = test_metrics
        print(f"    ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}")
        print()

    # 5. Calculate deltas
    print("[5/5] Calculating deltas...")
    deltas = {}
    for cond in conditions:
        cond_key = f"with_sentiment_{cond}"
        delta = {}
        for metric in ['loss', 'balanced_accuracy', 'roc_auc', 'pr_auc',
                       'precision_at_10', 'recall_at_10', 'map_at_10', 'ndcg_at_10']:
            delta[metric] = all_results[cond_key].get(metric, 0) - all_results['without_sentiment'].get(metric, 0)
        deltas[f"{cond}_vs_without"] = delta

    # 6. Save comparison JSON
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'embed_dim': DEEPFM_EMBED_DIM,
            'dnn_dims': DEEPFM_DNN,
            'dropout': DEEPFM_DROPOUT,
            'batch_size': DEEPFM_BATCH,
            'epochs': DEEPFM_EPOCHS,
            'learning_rate': DEEPFM_LR
        },
        'without_sentiment': all_results['without_sentiment'],
        'with_sentiment_a': all_results.get('with_sentiment_a', {}),
        'with_sentiment_b': all_results.get('with_sentiment_b', {}),
        'with_sentiment_c': all_results.get('with_sentiment_c', {}),
        'with_sentiment_d': all_results.get('with_sentiment_d', {}),
        'deltas': deltas,
        'target_metrics': {'loss': 0.5998, 'balanced_accuracy': 0.7636, 'roc_auc': 0.8447, 'pr_auc': 0.8214},
        'success_criteria': {'bert_macro_f1_target': 0.75, 'recall_minority_improvement': 0.05, 'deepfm_roc_auc_target': 0.8447}
    }

    comparison_path = result_dir / "deepfm_comparison.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(comparison), f, indent=2)
    print(f"  Comparison JSON saved: {comparison_path}")
    print()

    # 7. Generate visualization
    plot_deepfm_comparison(all_results, result_dir / "deepfm_comparison.png")

    # Print summary table
    print()
    print("=" * 80)
    print("DEEPFM COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<30} {'ROC-AUC':>10} {'PR-AUC':>10} {'Bal.Acc':>10} {'Δ ROC-AUC':>10}")
    print("-" * 80)
    print(f"{'Without Sentiment':<30} {all_results['without_sentiment']['roc_auc']:>10.4f} "
          f"{all_results['without_sentiment']['pr_auc']:>10.4f} "
          f"{all_results['without_sentiment']['balanced_accuracy']:>10.4f} {'—':>10}")
    for cond in conditions:
        key = f"with_sentiment_{cond}"
        if key in all_results:
            delta = deltas[f"{cond}_vs_without"]['roc_auc']
            print(f"{'With Sentiment (Cond ' + cond.upper() + ')':<30} "
                  f"{all_results[key]['roc_auc']:>10.4f} "
                  f"{all_results[key]['pr_auc']:>10.4f} "
                  f"{all_results[key]['balanced_accuracy']:>10.4f} "
                  f"{delta:>+10.4f}")
    print("-" * 80)
    print(f"Target ROC-AUC: 0.8447")
    print(f"Success: Δ ROC-AUC positive for at least one condition")
    print()
    print("=" * 60)
    print("DEEPFM EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()