"""
run_combined.py — Kondisi D: BERT + Focal Loss + EDA (Full Combination)
========================================================================
Fully self-contained — tidak ada external modules/
EDA Synonym Replacement + Focal Loss grid search gamma ∈ {1.0, 2.0, 3.0}.

Output:
  - models/bert_d_best/best_model.pt
  - logs/bert_d_grid_search_log.json
  - logs/bert_d_best_training_log.json
  - logs/sentiment_scores/sentiment_d.csv
  - results/condition_d/classification_report.txt
  - results/condition_d/confusion_matrix.png
  - results/condition_d/training_curve.png
  - results/condition_d/grid_search_results.json
"""

# ============================================================
# SECTION 1: IMPORTS
# ============================================================
import json
import os
import sys
import re
import html
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel as nn_parallel
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import nltk
for resource in ['wordnet', 'stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords, wordnet
STOPWORDS = set(stopwords.words('english'))

try:
    from nltk.stem import WordNetLemmatizer
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    LEMMATIZER = None

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
# SECTION 2: CONFIG
# ============================================================
SEED = 42
BERT_MODEL = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
PATIENCE = 3
NUM_LABELS = 5
LABEL_NAMES = ["Negative", "Slightly Negative", "Neutral", "Slightly Positive", "Positive"]
BERT_SPLIT = (0.70, 0.15, 0.15)

GAMMA_VALUES = [1.0, 2.0, 3.0]

EDA_ALPHA = 0.1
EDA_TARGET_CLASSES = [1, 2, 3]
EDA_N_AUG = {1: 2, 2: 2, 3: 1}
EDA_SEED = 42

# ============================================================
# SECTION 3: UTILITY FUNCTIONS
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if LEMMATIZER:
        words = [LEMMATIZER.lemmatize(w) for w in words]
    return ' '.join(words)

def rating_to_label(rating):
    if rating >= 8: return 4
    elif rating >= 6: return 3
    elif rating == 5: return 2
    elif rating >= 3: return 1
    else: return 0

def load_imdb_data(dataset_path):
    csv_files = list(Path(dataset_path).glob("*.csv")) + list(Path(dataset_path).glob("**/*.csv"))
    imdb_file = None
    for f in csv_files:
        if 'imdb' in f.name.lower() or 'review' in f.name.lower():
            imdb_file = f
            break
    if imdb_file is None:
        csv_files = list(Path(dataset_path).glob("*.csv"))
        if csv_files:
            imdb_file = csv_files[0]
        else:
            raise FileNotFoundError(f"No CSV file found in {dataset_path}")

    print(f"  Loading: {imdb_file}")
    df = pd.read_csv(imdb_file)
    print(f"  Columns: {df.columns.tolist()}")

    review_col_idx = None
    rating_col_idx = None
    for i, col in enumerate(df.columns):
        cl = col.lower().strip()
        if review_col_idx is None and 'review' in cl and 'text' in cl:
            review_col_idx = i
        if rating_col_idx is None and cl == 'rating':
            rating_col_idx = i

    if review_col_idx is None:
        raise ValueError(f"Could not find review text column. Available: {df.columns.tolist()}")
    if rating_col_idx is None:
        raise ValueError(f"Could not find rating column. Available: {df.columns.tolist()}")

    review_series = df.iloc[:, review_col_idx].astype(str).apply(clean_text)
    rating_series = df.iloc[:, rating_col_idx]

    df_clean = pd.DataFrame({
        'review_text': review_series,
        'label': rating_series.apply(lambda r: rating_to_label(int(float(r))) if pd.notna(r) else 2),
        'rating': rating_series
    })

    df_clean = df_clean[df_clean['review_text'].str.strip().str.len() > 0].reset_index(drop=True)
    return df_clean

def split_data_stratified(df, train_ratio, val_ratio, test_ratio, seed=SEED, label_col='label'):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), stratify=df[label_col], random_state=seed)
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_fraction), stratify=temp_df[label_col], random_state=seed)
    return train_df, val_df, test_df

def tokenize_texts(texts, tokenizer, max_length=MAX_LENGTH):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

# ============================================================
# SECTION 4: FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        fl = self.alpha * focal_weight * ce_loss
        if self.reduction == 'mean':
            return fl.mean()
        elif self.reduction == 'sum':
            return fl.sum()
        return fl

# ============================================================
# SECTION 5: EDA AUGMENTATION (OPTIMIZED)
# ============================================================
# Pre-compute synonyms cache to avoid repeated WordNet lookups
_SYNONYM_CACHE = {}

def get_synonyms(word):
    if word.lower() in _SYNONYM_CACHE:
        return _SYNONYM_CACHE[word.lower()]

    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word.lower() and len(synonym) > 2:
                synonyms.append(synonym)
    seen = set()
    unique = []
    for s in synonyms:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    _SYNONYM_CACHE[word.lower()] = unique
    return unique

def synonym_replacement(text, alpha=EDA_ALPHA, n=None, seed=EDA_SEED):
    random.seed(seed)
    words = text.split()
    if len(words) == 0:
        return text
    if n is None:
        n_replace = max(1, int(len(words) * alpha))
    replaceable_indices = [i for i, w in enumerate(words) if len(w) > 3 and get_synonyms(w)]
    if not replaceable_indices:
        return text
    n_replace = min(n_replace, len(replaceable_indices))
    indices_to_replace = random.sample(replaceable_indices, n_replace)
    for idx in indices_to_replace:
        word = words[idx]
        syns = get_synonyms(word)
        if syns:
            words[idx] = random.choice(syns)
    return ' '.join(words)

def augment_dataset(df, target_classes, n_aug, alpha=EDA_ALPHA, seed=EDA_SEED, text_col='review_text', label_col='label'):
    random.seed(seed)
    augmented_rows = []
    for label in target_classes:
        class_df = df[df[label_col] == label].copy()
        n_copies = n_aug.get(label, 0)
        for _, row in class_df.iterrows():
            original_text = row[text_col]
            for aug_idx in range(n_copies):
                aug_seed = seed + aug_idx + label * 1000
                aug_text = synonym_replacement(original_text, alpha=alpha, seed=aug_seed)
                if aug_text != original_text and len(aug_text.strip()) > 0:
                    new_row = row.copy()
                    new_row[text_col] = aug_text
                    augmented_rows.append(new_row)
    if augmented_rows:
        return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    return df

# ============================================================
# SECTION 6: BERT DATASET & TRAINING
# ============================================================
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
        }
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(dataloader, desc=" Training", leave=False, position=1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {'loss': total_loss / len(dataloader), 'accuracy': correct / total}

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=" Validating", leave=False, position=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return {'loss': total_loss / len(dataloader), 'accuracy': accuracy, 'macro_f1': macro_f1}

def train_bert(model, train_loader, val_loader, device, criterion=None, output_dir=None, condition_name="baseline", use_amp=True):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Use bfloat16 for better numerical stability (A40 supports natively)
    scaler = GradScaler() if (use_amp and torch.cuda.is_available()) else None

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    training_log = []

    for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch", position=0):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        val_metrics = validate(model, val_loader, criterion, device)
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1']
        }
        training_log.append(log_entry)
        print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}")

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch + 1
            patience_counter = 0
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
                break

    if output_dir and (output_dir / 'best_model.pt').exists():
        model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))

    return {'best_epoch': best_epoch, 'best_val_f1': best_val_f1, 'training_log': training_log}

def evaluate_bert(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=" Evaluating", leave=False, position=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu().numpy())
    all_probs = np.vstack(all_probs)
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True, digits=4)
    report_text = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': all_probs,
        'classification_report': report,
        'classification_report_text': report_text,
        'confusion_matrix': cm
    }

def predict_sentiment(model, tokenizer, texts, device, max_length=MAX_LENGTH, batch_size=64):
    model.eval()
    all_probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Sentiment", unit="batch", position=0):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(batch_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().float().numpy())
    all_probs = np.vstack(all_probs)
    class_weights = np.arange(NUM_LABELS, dtype=np.float32)
    weighted_scores = np.dot(all_probs, class_weights)
    return weighted_scores

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title('Confusion Matrix — Condition D (Focal Loss + EDA)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_training_curve(training_log, save_path):
    epochs = [e['epoch'] for e in training_log]
    train_loss = [e['train_loss'] for e in training_log]
    val_loss = [e['val_loss'] for e in training_log]
    val_f1 = [e['val_macro_f1'] for e in training_log]
    train_acc = [e['train_accuracy'] for e in training_log]
    val_acc = [e['val_accuracy'] for e in training_log]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', marker='o')
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, val_f1, 'g-', label='Val Macro F1', marker='^')
    axes[1].plot(epochs, train_acc, 'b--', label='Train Acc', marker='o')
    axes[1].plot(epochs, val_acc, 'r--', label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Macro F1 & Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ============================================================
# SECTION 7: MAIN EXECUTION
# ============================================================
def main():
    print("=" * 60)
    print("KONDISI D — BERT + Focal Loss + EDA (Full Combination)")
    print("=" * 60)
    print()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Enable cuDNN benchmark for faster training with fixed input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Disable DataParallel to avoid CUDA index issues with AMP
    # Single GPU is more stable for BERT training
    if n_gpus > 1:
        print(f"GPUs detected: {n_gpus}x {torch.cuda.get_device_name(0)} — using single GPU for stability")
        n_gpus = 1  # Force single GPU mode
        print(f"GPUs detected: {n_gpus}x {torch.cuda.get_device_name(0)} — using DataParallel")
    else:
        print(f"Device: {device}")
    print()

    # Auto-detect dataset path
    if os.path.exists("/kaggle/input"):
        dataset_path = Path("/kaggle/input/datasets/froggyjumpys/datase/IMDB")
    elif os.path.exists("/workspace/dataset/IMDB"):
        dataset_path = Path("/workspace/dataset/IMDB")
    elif os.path.exists("/content/drive"):
        dataset_path = Path("/content/drive/MyDrive/dataset/IMDB")
    else:
        dataset_path = Path("dataset/IMDB")  # fallback

    if not (dataset_path / "imdb_reviews.csv").exists():
        print(f"[ERROR] File not found: {dataset_path / 'imdb_reviews.csv'}")
        print(f"[ERROR] Available files: {list(dataset_path.glob('*')) if dataset_path.exists() else 'Directory not found'}")
        return
    print(f"[DATASET] Using: {dataset_path}")

    # Output paths (working directory)
    model_dir = Path("models/bert_d_best")
    log_dir = Path("logs")
    sentiment_dir = log_dir / "sentiment_scores"
    result_dir = Path("results/condition_d")
    for d in [model_dir, log_dir, sentiment_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load and split data
    print("[1/7] Loading IMDb data...")
    df = load_imdb_data(dataset_path)
    print(f"  Total samples: {len(df)}")
    print(f"  Class distribution before augmentation:\n{df['label'].value_counts().sort_index()}")

    train_df, val_df, test_df = split_data_stratified(df, BERT_SPLIT[0], BERT_SPLIT[1], BERT_SPLIT[2], seed=SEED)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print()

    # 2. EDA Augmentation (train set only)
    print("[2/7] Applying EDA augmentation on training set...")
    print(f"  Target classes: {EDA_TARGET_CLASSES}")
    print(f"  Augmentation factor: {EDA_N_AUG}")
    print(f"  Alpha (replacement rate): {EDA_ALPHA}")

    train_df_aug = augment_dataset(
        train_df,
        target_classes=EDA_TARGET_CLASSES,
        n_aug=EDA_N_AUG,
        alpha=EDA_ALPHA,
        seed=EDA_SEED,
        text_col='review_text',
        label_col='label'
    )

    print(f"  Train samples after augmentation: {len(train_df_aug)}")
    print(f"  Class distribution after augmentation:\n{train_df_aug['label'].value_counts().sort_index()}")
    print()

    # 3. Tokenize
    print("[3/7] Tokenizing...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    train_enc = tokenize_texts(train_df_aug['review_text'].tolist(), tokenizer)
    val_enc = tokenize_texts(val_df['review_text'].tolist(), tokenizer)
    test_enc = tokenize_texts(test_df['review_text'].tolist(), tokenizer)

    train_ds = IMDbDataset(train_enc, train_df_aug['label'].tolist())
    val_ds = IMDbDataset(val_enc, val_df['label'].tolist())
    test_ds = IMDbDataset(test_enc, test_df['label'].tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print("  Tokenization complete.")
    print()

    # 4. Grid search over gamma values
    print("[4/7] Grid search over gamma values...")
    print(f"  Searching gamma ∈ {GAMMA_VALUES}")
    print()

    grid_search_results = []

    for gamma in tqdm(GAMMA_VALUES, desc="Gamma Search", unit="gamma", position=0):
        print(f"  --- Training with gamma={gamma} ---")
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS).to(device)
        print("  BERT model loaded (single GPU mode for stability).")
        criterion = FocalLoss(gamma=gamma, reduction='mean')

        temp_model_dir = model_dir.parent / f"bert_d_gamma_{gamma}"
        training_result = train_bert(
            model, train_loader, val_loader, device,
            criterion=criterion,
            output_dir=temp_model_dir,
            condition_name=f"D_gamma_{gamma}"
        )

        grid_search_results.append({
            'gamma': gamma,
            'best_val_f1': training_result['best_val_f1'],
            'best_epoch': training_result['best_epoch'],
            'training_log': training_result['training_log']
        })
        print()

    # 5. Select best gamma
    best_result = max(grid_search_results, key=lambda x: x['best_val_f1'])
    best_gamma = best_result['gamma']
    print(f"[5/7] Best gamma: {best_gamma} (val macro F1: {best_result['best_val_f1']:.4f})")
    print()

    # Save grid search results
    grid_search_json = {
        'condition': 'D',
        'loss_type': 'FocalLoss',
        'augmentation': True,
        'eda_config': {'target_classes': EDA_TARGET_CLASSES, 'n_aug': EDA_N_AUG, 'alpha_sr': EDA_ALPHA},
        'train_samples_before_aug': len(train_df),
        'train_samples_after_aug': len(train_df_aug),
        'gamma_values': GAMMA_VALUES,
        'best_gamma': best_gamma,
        'selection_metric': 'val_macro_f1',
        'results': [
            {'gamma': r['gamma'], 'best_val_f1': r['best_val_f1'], 'best_epoch': r['best_epoch']}
            for r in grid_search_results
        ]
    }
    with open(result_dir / "grid_search_results.json", 'w') as f:
        json.dump(grid_search_json, f, indent=2)
    print(f"  Grid search results saved.")

    # Load best model
    print(f"[6/7] Loading best model (gamma={best_gamma})...")
    best_temp_dir = model_dir.parent / f"bert_d_gamma_{best_gamma}"
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS).to(device)
    print("  BERT model loaded (single GPU mode for stability).")
    model.load_state_dict(torch.load(best_temp_dir / 'best_model.pt', map_location=device))
    print("  Best model loaded.")
    torch.save(model.state_dict(), model_dir / 'best_model.pt')

    # Save training logs
    best_log_data = {
        'condition': 'D',
        'loss_type': f'FocalLoss(gamma={best_gamma})',
        'augmentation': True,
        'best_gamma': best_gamma,
        'best_val_f1': best_result['best_val_f1'],
        'best_epoch': best_result['best_epoch'],
        'training_log': best_result['training_log'],
        'timestamp': datetime.now().isoformat()
    }
    with open(log_dir / "bert_d_best_training_log.json", 'w') as f:
        json.dump(best_log_data, f, indent=2)
    with open(log_dir / "bert_d_grid_search_log.json", 'w') as f:
        json.dump(grid_search_json, f, indent=2)
    print()

    # 7. Evaluate on test set
    print("[7/7] Evaluating on test set...")
    eval_result = evaluate_bert(model, test_loader, device)

    report_path = result_dir / "classification_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("BERT SENTIMENT CLASSIFICATION — CONDITION D\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration:\n")
        f.write("  - Condition: D (Full Combination)\n")
        f.write(f"  - Loss: FocalLoss(gamma={best_gamma})\n")
        f.write("  - Augmentation: EDA Synonym Replacement\n")
        f.write(f"  - EDA target classes: {EDA_TARGET_CLASSES}\n")
        f.write(f"  - EDA alpha_sr: {EDA_ALPHA}\n")
        f.write(f"  - EDA n_aug: {EDA_N_AUG}\n")
        f.write(f"  - Best gamma (grid search): {best_gamma}\n")
        f.write(f"  - Seed: {SEED}\n")
        f.write(f"  - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Overall Accuracy: {eval_result['classification_report']['accuracy']:.4f}\n")
        f.write(f"Macro F1-Score: {eval_result['classification_report']['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted F1-Score: {eval_result['classification_report']['weighted avg']['f1-score']:.4f}\n\n")
        f.write("-" * 60 + "\n")
        f.write(eval_result['classification_report_text'])
    print("  Classification report saved.")

    plot_confusion_matrix(eval_result['confusion_matrix'], result_dir / "confusion_matrix.png")
    plot_training_curve(best_result['training_log'], result_dir / "training_curve.png")
    print("  Visualizations saved.")

    # Generate sentiment scores
    all_texts = df['review_text'].tolist()
    sentiment_scores = predict_sentiment(model, tokenizer, all_texts, device)
    sentiment_df = pd.DataFrame({'review_text': all_texts, 'sentiment_score': sentiment_scores})
    sentiment_path = sentiment_dir / "sentiment_d.csv"
    sentiment_df.to_csv(sentiment_path, index=False)
    print(f"  Sentiment scores saved: {sentiment_path}")
    print()

    print()
    print("=" * 60)
    print("KONDISI D COMPLETE")
    print("=" * 60)
    print(f"Best Gamma: {best_gamma}")
    print(f"Val Macro F1: {best_result['best_val_f1']:.4f}")
    print(f"Test Macro F1: {eval_result['classification_report']['macro avg']['f1-score']:.4f}")
    print(f"Test Accuracy: {eval_result['classification_report']['accuracy']:.4f}")

if __name__ == '__main__':
    main()