"""
06_deepfm_preprocessing.py
===========================
DeepFM preprocessing for MovieLens 1M dataset.

Handles:
- Load MovieLens 1M (ratings, users, movies)
- Join sentiment features from IMDB
- Feature engineering (user features, item features, context features)
- Train/val/test split (60/20/20)
- Save preprocessed data for DeepFM training

Usage:
    python 06_deepfm_preprocessing.py

Output:
    - logs/deepfm_preprocessed/train.pt
    - logs/deepfm_preprocessed/val.pt
    - logs/deepfm_preprocessed/test.pt
    - logs/deepfm_preprocessed/field_config.json
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config


# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# MovieLens 1M column names
RATINGS_COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
USERS_COLS = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
MOVIES_COLS = ['movie_id', 'title', 'genres']

# All genres in MovieLens 1M
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_movielens(base_path):
    """Load MovieLens 1M dataset."""
    print("[1/6] Loading MovieLens 1M dataset...")

    ratings_path = base_path / 'dataset' / 'MovieLens_OneM' / 'ratings.dat'
    users_path = base_path / 'dataset' / 'MovieLens_OneM' / 'users.dat'
    movies_path = base_path / 'dataset' / 'MovieLens_OneM' / 'movies.dat'

    # Load ratings
    ratings = pd.read_csv(
        ratings_path,
        sep='::',
        names=RATINGS_COLS,
        engine='python',
        encoding='latin-1'
    )
    print(f"  Ratings: {len(ratings)} records")

    # Load users
    users = pd.read_csv(
        users_path,
        sep='::',
        names=USERS_COLS,
        engine='python',
        encoding='latin-1'
    )
    print(f"  Users: {len(users)} records")

    # Load movies
    movies = pd.read_csv(
        movies_path,
        sep='::',
        names=MOVIES_COLS,
        engine='python',
        encoding='latin-1'
    )
    print(f"  Movies: {len(movies)} records")

    return ratings, users, movies


def encode_categorical(df, column, prefix=''):
    """Encode categorical column to integer IDs."""
    unique_values = df[column].unique()
    value_to_id = {v: i for i, v in enumerate(unique_values)}
    df[f'{prefix}{column}_encoded'] = df[column].map(value_to_id)
    return df, len(unique_values)


def parse_genres(genres_str):
    """Parse genre string and return multi-hot vector."""
    if pd.isna(genres_str):
        return [0] * len(ALL_GENRES)
    genres = genres_str.split('|')
    return [1 if g in genres else 0 for g in ALL_GENRES]


# ============================================================================
# Main Preprocessing Function
# ============================================================================

def preprocess_deepfm(config):
    """
    Main DeepFM preprocessing pipeline.

    Args:
        config: dict, Configuration from setup

    Returns:
        dict: Paths to saved preprocessed data
    """
    print("\n" + "=" * 60)
    print("DEEPFM PREPROCESSING - MovieLens 1M")
    print("=" * 60 + "\n")

    set_seed(RANDOM_SEED)

    # Setup paths
    base_path = Path(config['base_path'])
    logs_path = Path(config['logs_path'])
    preprocessed_path = logs_path / 'deepfm_preprocessed'
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load MovieLens
    # =========================================================================
    ratings, users, movies = load_movielens(base_path)

    # =========================================================================
    # Step 2: Load Sentiment Features
    # =========================================================================
    print("\n[2/6] Loading sentiment features...")

    sentiment_path = logs_path / 'movie_sentiment_scores.csv'
    if sentiment_path.exists():
        sentiment_df = pd.read_csv(sentiment_path)
        print(f"  Loaded sentiment for {len(sentiment_df)} movies")
    else:
        print("  [WARNING] Sentiment file not found, using default sentiment=0.5")
        sentiment_df = pd.DataFrame({
            'movie_id': movies['movie_id'],
            'normalized_sentiment': [0.5] * len(movies)
        })

    # =========================================================================
    # Step 3: Feature Engineering
    # =========================================================================
    print("\n[3/6] Feature engineering...")

    # Encode user features
    print("  Encoding user features...")
    users, num_genders = encode_categorical(users, 'gender')
    users, num_ages = encode_categorical(users, 'age')
    users, num_occupations = encode_categorical(users, 'occupation')

    # Encode zip_code
    users['zip_code'] = users['zip_code'].astype(str)
    users, num_zipcodes = encode_categorical(users, 'zip_code')

    print(f"    Genders: {num_genders}")
    print(f"    Ages: {num_ages}")
    print(f"    Occupations: {num_occupations}")
    print(f"    Zip codes: {num_zipcodes}")

    # Encode movie_id
    movies, num_movies = encode_categorical(movies, 'movie_id')

    # Parse genres (multi-hot)
    print("  Parsing genres...")
    movies['genre_vector'] = movies['genres'].apply(parse_genres)

    # =========================================================================
    # Step 4: Merge Datasets
    # =========================================================================
    print("\n[4/6] Merging datasets...")

    # Merge ratings with users
    df = ratings.merge(users, on='user_id', how='left')

    # Merge with movies
    df = df.merge(movies, on='movie_id', how='left')

    # Merge with sentiment
    df = df.merge(
        sentiment_df[['movie_id', 'normalized_sentiment']],
        on='movie_id',
        how='left'
    )

    # Fill missing sentiment with 0.5 (neutral)
    df['normalized_sentiment'] = df['normalized_sentiment'].fillna(0.5)

    print(f"  Final dataset: {len(df)} records")

    # =========================================================================
    # Step 5: Prepare Features
    # =========================================================================
    print("\n[5/6] Preparing features...")

    # Get max IDs for embedding
    max_user_id = df['user_id'].max() + 1
    max_movie_id = df['movie_id'].max() + 1
    max_gender_id = df['gender_encoded'].max() + 1
    max_age_id = df['age_encoded'].max() + 1
    max_occupation_id = df['occupation_encoded'].max() + 1
    max_zipcode_id = df['zip_code_encoded'].max() + 1

    # Create feature dictionary
    features = {
        'user_id': df['user_id'].values,
        'movie_id': df['movie_id_encoded'].values,
        'gender': df['gender_encoded'].values,
        'age': df['age_encoded'].values,
        'occupation': df['occupation_encoded'].values,
        'zip_code': df['zip_code_encoded'].values,
        'genres': np.array(df['genre_vector'].tolist()),
        'sentiment': df['normalized_sentiment'].values,
        'timestamp': df['timestamp'].values
    }

    # Binarize rating (>=4 ->1, <=3 -> 0)
    labels = (df['rating'] >= 4).astype(int).values

    print(f"  User IDs: {max_user_id}")
    print(f"  Movie IDs: {max_movie_id}")
    print(f"  Positive samples: {labels.sum()} ({labels.mean()*100:.1f}%)")
    print(f"  Negative samples: {len(labels) - labels.sum()} ({(1-labels.mean())*100:.1f}%)")

    # =========================================================================
    # Step 6: Train/Val/Test Split
    # =========================================================================
    print("\n[6/6] Splitting data (60% train, 20% val, 20% test)...")

    # First split: 80% train_val, 20% test
    n_total = len(labels)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    train_val_size = int(n_total * (TRAIN_RATIO + VAL_RATIO))
    train_val_idx = indices[:train_val_size]
    test_idx = indices[train_val_size:]

    # Second split: 60% train, 20% val (from train_val)
    train_size = int(n_total * TRAIN_RATIO)
    train_idx = train_val_idx[:train_size]
    val_idx = train_val_idx[train_size:]

    def prepare_split(split_idx):
        """Prepare data for a split."""
        return {
            'user_id': features['user_id'][split_idx],
            'movie_id': features['movie_id'][split_idx],
            'gender': features['gender'][split_idx],
            'age': features['age'][split_idx],
            'occupation': features['occupation'][split_idx],
            'zip_code': features['zip_code'][split_idx],
            'genres': features['genres'][split_idx],
            'sentiment': features['sentiment'][split_idx],
            'timestamp': features['timestamp'][split_idx],
            'labels': labels[split_idx]
        }

    train_data = prepare_split(train_idx)
    val_data = prepare_split(val_idx)
    test_data = prepare_split(test_idx)

    print(f"  Train: {len(train_data['labels'])} samples")
    print(f"  Val: {len(val_data['labels'])} samples")
    print(f"  Test: {len(test_data['labels'])} samples")

    # =========================================================================
    # Save Preprocessed Data
    # =========================================================================
    print("\n[Saving] Saving preprocessed data...")

    torch.save(train_data, preprocessed_path / 'train.pt')
    torch.save(val_data, preprocessed_path / 'val.pt')
    torch.save(test_data, preprocessed_path / 'test.pt')

    print(f"  Saved train.pt: {len(train_data['labels'])} samples")
    print(f"  Saved val.pt: {len(val_data['labels'])} samples")
    print(f"  Saved test.pt: {len(test_data['labels'])} samples")

    # Save field configuration
    field_config = {
        'max_user_id': int(max_user_id),
        'max_movie_id': int(max_movie_id),
        'max_gender_id': int(max_gender_id),
        'max_age_id': int(max_age_id),
        'max_occupation_id': int(max_occupation_id),
        'max_zipcode_id': int(max_zipcode_id),
        'num_genres': len(ALL_GENRES),
        'feature_dims': {
            'user_id': int(max_user_id),
            'movie_id': int(max_movie_id),
            'gender': int(max_gender_id),
            'age': int(max_age_id),
            'occupation': int(max_occupation_id),
            'zip_code': int(max_zipcode_id),
            'genres': len(ALL_GENRES),
            'sentiment': 1
        },
        'all_genres': ALL_GENRES
    }

    config_path = preprocessed_path / 'field_config.json'
    with open(config_path, 'w') as f:
        json.dump(field_config, f, indent=2)
    print(f"  Saved field_config.json")

    # Save preprocessing log
    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': RANDOM_SEED,
        'total_records': len(df),
        'train_samples': len(train_data['labels']),
        'val_samples': len(val_data['labels']),
        'test_samples': len(test_data['labels']),
        'field_config': field_config
    }

    log_path = preprocessed_path / 'preprocessing_log.json'
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"  Saved preprocessing_log.json")

    print("\n" + "=" * 60)
    print("DEEPFM PREPROCESSING COMPLETE")
    print("=" * 60)

    return {
        'preprocessed_path': str(preprocessed_path),
        'train_path': str(preprocessed_path / 'train.pt'),
        'val_path': str(preprocessed_path / 'val.pt'),
        'test_path': str(preprocessed_path / 'test.pt'),
        'config_path': str(config_path),
        'field_config': field_config
    }


if __name__ == '__main__':
    config = load_config()
    result = preprocess_deepfm(config)

    print("\nPreprocessing Summary:")
    for key, value in result.items():
        if key != 'field_config':
            print(f"  {key}: {value}")
