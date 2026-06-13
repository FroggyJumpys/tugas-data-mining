"""
02_bert_preprocessing.py
========================
BERT preprocessing for IMDb sentiment classification.

Handles:
- Load IMDB CSV data
- Convert ratings to 5-class sentiment labels
- Text cleaning (lowercase, remove punctuation/stopwords/HTML, lemmatization)
- BERT tokenization (bert-base-uncased)
- Train/test split (70/30, stratified)

Usage:
    python 02_bert_preprocessing.py

Output:
    - logs/bert_preprocessed/train.pt
    - logs/bert_preprocessed/test.pt
    - logs/bert_preprocessed/tokenizer/
    - logs/bert_preprocessed/preprocessing_log.json
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import load_config


# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
MAX_LENGTH = 256
TRAIN_RATIO = 0.7
BATCH_SIZE = 16

# Rating to sentiment label mapping
RATING_TO_LABEL = {
    (8, 10): 4,   # Positive
    (6, 7): 3,    # Slightly Positive
    (5, 5): 2,    # Neutral
    (3, 4): 1,    # Slightly Negative
    (1, 2): 0,    # Negative
}


# ============================================================================
# Text Cleaning Functions
# ============================================================================

def download_nltk_resources():
    """Download required NLTK resources with force=True."""
    import nltk
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True, force=True)
        except Exception as e:
            print(f"  [WARNING] Could not download {resource}: {e}")


def clean_text(text):
    """
    Clean text according to paper specifications:
    - Lowercasing
    - Remove punctuation
    - Remove stop words
    - Remove hashtags, HTML tags
    - Lemmatization (with fallback to simple processing if NLTK fails)
    """
    if not isinstance(text, str):
        return ""

    import re

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove punctuation (keep only alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Try NLTK lemmatization, fallback to simple processing
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        # Ensure resources are downloaded
        try:
            stop_words = set(stopwords.words('english'))
        except:
            download_nltk_resources()
            stop_words = set(stopwords.words('english'))

        lemmatizer = WordNetLemmatizer()

        # Tokenize, remove stopwords, lemmatize
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]

        return ' '.join(words)

    except Exception as e:
        # Fallback: simple processing without lemmatization
        words = text.split()
        # Simple stopword removal (common English stopwords)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose'
        }
        words = [w for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(words)


def rating_to_sentiment_label(rating):
    """
    Convert IMDb rating (1-10) to 5-class sentiment label.

    Args:
        rating: int, IMDb rating (1-10)

    Returns:
        int: Sentiment label (0-4)
    """
    for (min_val, max_val), label in RATING_TO_LABEL.items():
        if min_val <= rating <= max_val:
            return label
    return 2  # Default to Neutral for any edge cases


# ============================================================================
# Dataset Classes
# ============================================================================

class IMDbDataset(Dataset):
    """PyTorch Dataset for IMDb reviews."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item


# ============================================================================
# Main Preprocessing Function
# ============================================================================

def preprocess_bert_data(config):
    """
    Main preprocessing pipeline for BERT.

    Args:
        config: dict, Configuration from setup

    Returns:
        dict: Paths to saved preprocessed data
    """
    print("\n" + "=" * 60)
    print("BERT PREPROCESSING")
    print("=" * 60 + "\n")

    # Set random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Download NLTK resources before processing
    print("[0/6] Downloading NLTK resources...")
    download_nltk_resources()
    print("  NLTK resources ready")

    # Setup paths
    base_path = Path(config['base_path'])
    logs_path = Path(config['logs_path'])
    preprocessed_path = logs_path / 'bert_preprocessed'
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    dataset_path = base_path / 'dataset' / 'IMDB' / 'imdb_reviews.csv'

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("[1/6] Loading IMDB dataset...")
    df = pd.read_csv(dataset_path)
    print(f"  Loaded {len(df)} reviews")
    print(f"  Columns: {list(df.columns)}")

    # =========================================================================
    # Step 2: Handle Missing Values
    # =========================================================================
    print("\n[2/6] Handling missing values...")
    initial_count = len(df)
    df = df.dropna(subset=['review_text'])
    print(f"  Dropped {initial_count - len(df)} rows with missing review_text")
    print(f"  Remaining: {len(df)} reviews")

    # =========================================================================
    # Step 3: Convert Ratings to Labels
    # =========================================================================
    print("\n[3/6] Converting ratings to 5-class sentiment labels...")
    df['sentiment_label'] = df['rating'].apply(rating_to_sentiment_label)

    # Print label distribution
    label_counts = df['sentiment_label'].value_counts().sort_index()
    label_names = ['Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Positive']
    print("  Label distribution:")
    for label, count in label_counts.items():
        print(f"    {label} ({label_names[label]}): {count} ({count/len(df)*100:.1f}%)")

    # =========================================================================
    # Step 4: Text Cleaning
    # =========================================================================
    print("\n[4/6] Cleaning text...")
    print("  - Lowercasing")
    print("  - Removing HTML tags, hashtags, URLs")
    print("  - Removing punctuation")
    print("  - Removing stopwords")
    print("  - Lemmatization")

    # Combine review title and review text
    df['cleaned_text'] = df.apply(
        lambda row: f"{row.get('review_title', '')} {row['review_text']}",
        axis=1
    )

    # Clean each text
    cleaned_texts = []
    for text in tqdm(df['cleaned_text'], desc="  Cleaning texts"):
        cleaned_texts.append(clean_text(text))

    df['cleaned_text'] = cleaned_texts

    # Remove empty texts after cleaning
    initial_count = len(df)
    df = df[df['cleaned_text'].str.len() > 0]
    print(f"  Removed {initial_count - len(df)} empty texts after cleaning")
    print(f"  Remaining: {len(df)} reviews")

    # =========================================================================
    # Step 5: Train/Test Split
    # =========================================================================
    print("\n[5/6] Splitting data (70% train, 30% test)...")

    X = df['cleaned_text'].values
    y = df['sentiment_label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1 - TRAIN_RATIO,
        stratify=y,
        random_state=RANDOM_SEED
    )

    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # =========================================================================
    # Step 6: BERT Tokenization
    # =========================================================================
    print("\n[6/6] Tokenizing with BERT (bert-base-uncased)...")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Padding: max_length")
    print(f"  Truncation: True")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize train set
    print("  Tokenizing train set...")
    train_encodings = tokenizer(
        list(X_train),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Tokenize test set
    print("  Tokenizing test set...")
    test_encodings = tokenizer(
        list(X_test),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # =========================================================================
    # Save Preprocessed Data
    # =========================================================================
    print("\n[Saving] Saving preprocessed data...")

    # Save as PyTorch tensors
    train_data = {
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': torch.tensor(y_train)
    }
    test_data = {
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': torch.tensor(y_test)
    }

    torch.save(train_data, preprocessed_path / 'train.pt')
    torch.save(test_data, preprocessed_path / 'test.pt')
    print(f"  Saved train.pt: {preprocessed_path / 'train.pt'}")
    print(f"  Saved test.pt: {preprocessed_path / 'test.pt'}")

    # Save tokenizer
    tokenizer_path = preprocessed_path / 'tokenizer'
    tokenizer.save_pretrained(tokenizer_path)
    print(f"  Saved tokenizer: {tokenizer_path}")

    # Save preprocessing log
    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': RANDOM_SEED,
        'max_length': MAX_LENGTH,
        'train_ratio': TRAIN_RATIO,
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'label_distribution': {label_names[k]: int(v) for k, v in label_counts.items()},
        'rating_to_label_mapping': {f"{k[0]}-{k[1]}": v for k, v in RATING_TO_LABEL.items()}
    }

    log_path = preprocessed_path / 'preprocessing_log.json'
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"  Saved preprocessing_log.json: {log_path}")

    print("\n" + "=" * 60)
    print("BERT PREPROCESSING COMPLETE")
    print("=" * 60)

    return {
        'preprocessed_path': str(preprocessed_path),
        'train_path': str(preprocessed_path / 'train.pt'),
        'test_path': str(preprocessed_path / 'test.pt'),
        'tokenizer_path': str(tokenizer_path),
        'log_path': str(log_path),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }


if __name__ == '__main__':
    # Load config from setup
    config = load_config()
    result = preprocess_bert_data(config)

    # Print summary
    print("\nPreprocessing Summary:")
    for key, value in result.items():
        print(f"  {key}: {value}")
