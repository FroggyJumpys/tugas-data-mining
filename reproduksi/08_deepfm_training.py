"""
08_deepfm_training.py
=======================
DeepFM training for MovieLens1M recommendation.

Trains two variants:
- Variant A: Without sentiment features (baseline)
- Variant B: With sentiment features (target)

Hyperparameters (from paper):
- Optimizer: Adam (lr=1e-3)
- Loss: BCEWithLogitsLoss
- Batch Size: 256
- Epoch: 10
- Early Stopping: patience=3

Usage:
    python 08_deepfm_training.py

Output:
    - models/deepfm/deepfm_without_sentiment.pt
    - models/deepfm/deepfm_with_sentiment.pt
    - models/deepfm/training_log.json
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config
from deepfm_model import DeepFM, DeepFMDataset, collate_fn


# ============================================================================
# Configuration (from paper)
# ============================================================================

RANDOM_SEED = 42
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
EPOCHS = 10
EMBED_DIM = 16
DNN_HIDDEN_DIMS = [400, 400, 400]
DROPOUT = 0.5
EARLY_STOPPING_PATIENCE = 3


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_metrics(predictions, labels):
    """Calculate evaluation metrics."""
    preds_binary = (predictions >= 0.5).astype(int)
    labels_binary = labels.astype(int)

    metrics = {
        'loss': nn.BCELoss()(torch.tensor(predictions), torch.tensor(labels)).item(),
        'balanced_accuracy': balanced_accuracy_score(labels_binary, preds_binary),
        'roc_auc': roc_auc_score(labels_binary, predictions),
        'pr_auc': average_precision_score(labels_binary, predictions)
    }

    return metrics


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for sparse, dense, labels in tqdm(dataloader, desc="  Training"):
        # Move to device
        sparse = {k: v.to(device) for k, v in sparse.items()}
        if dense is not None:
            dense = {k: v.to(device) for k, v in dense.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(sparse, dense)

        # Loss
        loss = criterion(predictions, labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_labels))
    metrics['loss'] = avg_loss

    return metrics


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sparse, dense, labels in tqdm(dataloader, desc="  Evaluating"):
            sparse = {k: v.to(device) for k, v in sparse.items()}
            if dense is not None:
                dense = {k: v.to(device) for k, v in dense.items()}
            labels = labels.to(device)

            predictions = model(sparse, dense)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_labels))
    metrics['loss'] = avg_loss

    return metrics


def train_deepfm_variant(
    train_data,
    val_data,
    field_config,
    use_sentiment,
    variant_name,
    device,
    models_path
):
    """
    Train a DeepFM variant.

    Args:
        train_data: dict, Training data
        val_data: dict, Validation data
        field_config: dict, Feature configuration
        use_sentiment: bool, Whether to use sentiment features
        variant_name: str, Name of variant
        device: torch.device
        models_path: Path

    Returns:
        dict: Training results
    """
    print(f"\n  Training {variant_name}...")

    # Create datasets
    train_dataset = DeepFMDataset(train_data, use_sentiment=use_sentiment)
    val_dataset = DeepFMDataset(val_data, use_sentiment=use_sentiment)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Create model
    model = DeepFM(
        field_config=field_config,
        embed_dim=EMBED_DIM,
        dnn_hidden_dims=DNN_HIDDEN_DIMS,
        dropout=DROPOUT,
        use_sentiment=use_sentiment
    )
    model.to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    training_log = {'epochs': []}

    for epoch in range(EPOCHS):
        print(f"\n    Epoch {epoch + 1}/{EPOCHS}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_roc_auc': train_metrics['roc_auc'],
            'val_loss': val_metrics['loss'],
            'val_roc_auc': val_metrics['roc_auc'],
            'val_pr_auc': val_metrics['pr_auc'],
            'val_balanced_accuracy': val_metrics['balanced_accuracy']
        }
        training_log['epochs'].append(epoch_log)

        print(f"    Train Loss: {train_metrics['loss']:.4f}, ROC-AUC: {train_metrics['roc_auc']:.4f}")
        print(f"    Val Loss: {val_metrics['loss']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}")

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"    [NEW BEST] Val Loss: {val_metrics['loss']:.4f}")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"    Early stopping triggered")
            break

    # Save best model
    model.load_state_dict(best_model_state)
    model_path = models_path / f'deepfm_{variant_name}.pt'
    torch.save(model.state_dict(), model_path)
    print(f"    Saved model: {model_path}")

    training_log['best_val_loss'] = best_val_loss
    training_log['final_roc_auc'] = training_log['epochs'][-1]['val_roc_auc']

    return {
        'model_path': str(model_path),
        'training_log': training_log,
        'best_val_loss': best_val_loss
    }


# ============================================================================
# Main Training Function
# ============================================================================

def train_deepfm(config):
    """
    Main DeepFM training pipeline.

    Args:
        config: dict, Configuration from setup

    Returns:
        dict: Training results for both variants
    """
    print("\n" + "=" * 60)
    print("DEEPFM TRAINING - MovieLens 1M")
    print("=" * 60 + "\n")

    set_seed(RANDOM_SEED)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Setup paths
    logs_path = Path(config['logs_path'])
    models_path = Path(config['models_path'])

    preprocessed_path = logs_path / 'deepfm_preprocessed'

    # =========================================================================
    # Load Preprocessed Data
    # =========================================================================
    print("[1/4] Loading preprocessed data...")

    train_data = torch.load(preprocessed_path / 'train.pt')
    val_data = torch.load(preprocessed_path / 'val.pt')
    test_data = torch.load(preprocessed_path / 'test.pt')

    with open(preprocessed_path / 'field_config.json', 'r') as f:
        field_config = json.load(f)

    print(f"  Train samples: {len(train_data['labels'])}")
    print(f"  Val samples: {len(val_data['labels'])}")
    print(f"  Test samples: {len(test_data['labels'])}")

    # =========================================================================
    # Train Variant A: Without Sentiment
    # =========================================================================
    print("\n[2/4] Training DeepFM without sentiment...")

    result_a = train_deepfm_variant(
        train_data=train_data,
        val_data=val_data,
        field_config=field_config,
        use_sentiment=False,
        variant_name='without_sentiment',
        device=device,
        models_path=models_path / 'deepfm'
    )

    # =========================================================================
    # Train Variant B: With Sentiment
    # =========================================================================
    print("\n[3/4] Training DeepFM with sentiment...")

    result_b = train_deepfm_variant(
        train_data=train_data,
        val_data=val_data,
        field_config=field_config,
        use_sentiment=True,
        variant_name='with_sentiment',
        device=device,
        models_path=models_path / 'deepfm'
    )

    # =========================================================================
    # Save Training Log
    # =========================================================================
    print("\n[4/4] Saving training log...")

    training_log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'embed_dim': EMBED_DIM,
            'dnn_hidden_dims': DNN_HIDDEN_DIMS,
            'dropout': DROPOUT,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'random_seed': RANDOM_SEED
        },
        'variant_a_without_sentiment': result_a['training_log'],
        'variant_b_with_sentiment': result_b['training_log']
    }

    log_path = models_path / 'deepfm' / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"  Saved: {log_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("\nVariant A (Without Sentiment):")
    print(f"  Best Val Loss: {result_a['best_val_loss']:.4f}")
    print(f"  Final ROC-AUC: {result_a['training_log']['final_roc_auc']:.4f}")

    print("\nVariant B (With Sentiment):")
    print(f"  Best Val Loss: {result_b['best_val_loss']:.4f}")
    print(f"  Final ROC-AUC: {result_b['training_log']['final_roc_auc']:.4f}")

    target_roc_auc = 0.8447
    print(f"\nTarget ROC-AUC (from paper): {target_roc_auc:.4f}")

    return {
        'variant_a': result_a,
        'variant_b': result_b,
        'log_path': str(log_path)
    }


if __name__ == '__main__':
    config = load_config()
    result = train_deepfm(config)

    print("\nTraining Summary:")
    for key, value in result.items():
        print(f"  {key}: {value}")
