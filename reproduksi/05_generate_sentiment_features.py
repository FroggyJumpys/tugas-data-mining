"""
05_generate_sentiment_features.py
===================================
Generate sentiment features from BERT model predictions.

Steps:
1. Load best BERT model
2. Predict sentiment for ALL57,534 IMDb reviews
3. Aggregate sentiment scores by movie_id (mean)
4. Normalize to [0, 1]
5. Save to CSV for joining with MovieLens

Usage:
    python 05_generate_sentiment_features.py

Output:
    - logs/movie_sentiment_scores.csv
    - logs/sentiment_generation_log.json
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config


# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 32
NUM_LABELS = 5


# ============================================================================
# Dataset Class
# ============================================================================

class IMDbFullDataset:
    """Dataset for all IMDb reviews (for sentiment prediction)."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item


# ============================================================================
# Main Function
# ============================================================================

def generate_sentiment_features(config):
    """
    Generate sentiment features for all movies.

    Args:
        config: dict, Configuration from setup

    Returns:
        dict: Paths to saved sentiment features
    """
    print("\n" + "=" * 60)
    print("GENERATE SENTIMENT FEATURES")
    print("=" * 60 + "\n")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Setup paths
    base_path = Path(config['base_path'])
    logs_path = Path(config['logs_path'])
    models_path = Path(config['models_path'])

    model_path = models_path / 'bert_sentiment'
    imdb_path = base_path / 'dataset' / 'IMDB' / 'imdb_reviews.csv'

    # =========================================================================
    # Step 1: Load BERT Model
    # =========================================================================
    print("[1/5] Loading BERT model...")

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(model_path / 'best_model.pt', map_location=device))
    model.to(device)
    model.eval()
    print("  Model loaded successfully")

    # =========================================================================
    # Step 2: Load Full IMDB Dataset
    # =========================================================================
    print("\n[2/5] Loading full IMDB dataset...")

    df = pd.read_csv(imdb_path)
    print(f"  Total reviews: {len(df)}")

    # Handle missing values
    df = df.dropna(subset=['review_text'])
    print(f"  After removing missing: {len(df)}")

    # =========================================================================
    # Step 3: Tokenize All Reviews
    # =========================================================================
    print("\n[3/5] Tokenizing all reviews...")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Combine title and text
    df['full_text'] = df.apply(
        lambda row: f"{row.get('review_title', '')} {row['review_text']}",
        axis=1
    )

    # Tokenize
    encodings = tokenizer(
        list(df['full_text']),
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    print(f"  Tokenized {len(encodings['input_ids'])} reviews")

    # =========================================================================
    # Step 4: Predict Sentiment
    # =========================================================================
    print("\n[4/5] Predicting sentiment for all reviews...")

    dataset = IMDbFullDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_sentiment_scores = []
    all_movie_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get probabilities (softmax)
            probs = torch.softmax(outputs.logits, dim=1)

            # Calculate weighted sentiment score
            # Labels: 0=Negative, 1=Slightly Negative, 2=Neutral, 3=Slightly Positive, 4=Positive
            weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
            sentiment_score = (probs * weights).sum(dim=1)

            all_sentiment_scores.extend(sentiment_score.cpu().numpy())
            all_movie_ids.extend(df['movie_id'].iloc[
                len(all_sentiment_scores) - len(sentiment_score):
 ].tolist())

    # Actually, we need to track indices properly
    # Let me redo this properly
    all_sentiment_scores = []
    all_movie_ids = df['movie_id'].tolist()  # Get all movie IDs first

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="  Predicting")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get probabilities (softmax)
            probs = torch.softmax(outputs.logits, dim=1)

            # Calculate weighted sentiment score
            weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
            sentiment_score = (probs * weights).sum(dim=1)

            all_sentiment_scores.extend(sentiment_score.cpu().numpy())

    df['sentiment_score'] = all_sentiment_scores

    print(f"  Predicted sentiment for {len(df)} reviews")

    # =========================================================================
    # Step 5: Aggregate by Movie
    # =========================================================================
    print("\n[5/5] Aggregating sentiment by movie...")

    # Group by movie_id and calculate mean sentiment
    movie_sentiment = df.groupby('movie_id').agg({
        'sentiment_score': 'mean',
        'review_text': 'count'  # Number of reviews
    }).reset_index()

    movie_sentiment.columns = ['movie_id', 'avg_sentiment_score', 'num_reviews']

    # Normalize to [0, 1]
    min_score = movie_sentiment['avg_sentiment_score'].min()
    max_score = movie_sentiment['avg_sentiment_score'].max()

    movie_sentiment['normalized_sentiment'] = (
        (movie_sentiment['avg_sentiment_score'] - min_score) /
        (max_score - min_score)
    )

    print(f"  Total movies with sentiment: {len(movie_sentiment)}")
    print(f"  Sentiment score range: [{min_score:.4f}, {max_score:.4f}]")
    print(f"  Normalized range: [0, 1]")

    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n[Saving] Saving sentiment features...")

    # Save to CSV
    sentiment_path = logs_path / 'movie_sentiment_scores.csv'
    movie_sentiment.to_csv(sentiment_path, index=False)
    print(f"  Saved: {sentiment_path}")

    # Save log
    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_reviews': len(df),
        'total_movies': len(movie_sentiment),
        'sentiment_score_range': {
            'min': float(min_score),
            'max': float(max_score)
        },
        'movie_review_counts': {
            'min': int(movie_sentiment['num_reviews'].min()),
            'max': int(movie_sentiment['num_reviews'].max()),
            'mean': float(movie_sentiment['num_reviews'].mean())
        }
    }

    log_path = logs_path / 'sentiment_generation_log.json'
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: {log_path}")

    # Print sample
    print("\nSample sentiment scores:")
    print(movie_sentiment.head(10).to_string())

    print("\n" + "=" * 60)
    print("SENTIMENT FEATURES GENERATED")
    print("=" * 60)

    return {
        'sentiment_path': str(sentiment_path),
        'log_path': str(log_path),
        'total_movies': len(movie_sentiment),
        'total_reviews': len(df)
    }


if __name__ == '__main__':
    config = load_config()
    result = generate_sentiment_features(config)

    print("\nSentiment Generation Summary:")
    for key, value in result.items():
        print(f"  {key}: {value}")
