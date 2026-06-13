"""
09_deepfm_evaluation.py
========================
DeepFM evaluation on MovieLens 1M test set.

Metrics calculated:
- Loss (BCE)
- Balanced Accuracy
- ROC-AUC
- PR-AUC
- Precision@10
- Recall@10
- MAP
- NDCG

Comparison table: Without vs With Sentiment

Usage:
    python 09_deepfm_evaluation.py

Output:
    - results/deepfm_comparison.json
    - results/deepfm_metrics.png
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config
from deepfm_model import DeepFM, DeepFMDataset, collate_fn


# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 256
EMBED_DIM = 16
DNN_HIDDEN_DIMS = [400, 400, 400]
DROPOUT = 0.5


# ============================================================================
# Metric Functions
# ============================================================================

def ndcg_at_k(y_true, y_pred, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    Args:
        y_true: array, True relevance scores
        y_pred: array, Predicted scores
        k: int, Cutoff position

    Returns:
        float: NDCG@k
    """
    # Get top-k indices
    top_k_idx = np.argsort(y_pred)[-k:][::-1]

    # Calculate DCG
    dcg = 0.0
    for i, idx in enumerate(top_k_idx):
        dcg += y_true[idx] / np.log2(i + 2)

    # Calculate IDCG (ideal DCG)
    ideal_idx = np.argsort(y_true)[-k:][::-1]
    idcg = 0.0
    for i, idx in enumerate(ideal_idx):
        idcg += y_true[idx] / np.log2(i + 2)

    # Return NDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg


def map_at_k(y_true, y_pred, k=10):
    """
    Calculate Mean Average Precision at k.

    Args:
        y_true: array, True binary labels
        y_pred: array, Predicted scores
        k: int, Cutoff position

    Returns:
        float: MAP@k
    """
    top_k_idx = np.argsort(y_pred)[-k:][::-1]
    relevant = y_true[top_k_idx]

    # Calculate AP
    precision_sum = 0.0
    num_relevant = 0
    for i, rel in enumerate(relevant):
        if rel == 1:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)

    if num_relevant == 0:
        return 0.0
    return precision_sum / num_relevant


def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sparse, dense, labels in tqdm(dataloader, desc="  Evaluating"):
            sparse = {k: v.to(device) for k, v in sparse.items()}
            if dense is not None:
                dense = {k: v.to(device) for k, v in dense.items()}
            labels = labels.to(device)

            predictions = model(sparse, dense)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)


def calculate_all_metrics(predictions, labels):
    """Calculate all evaluation metrics."""
    preds_binary = (predictions >= 0.5).astype(int)
    labels_binary = labels.astype(int)

    metrics = {
        'loss': float(nn.BCELoss()(torch.tensor(predictions), torch.tensor(labels)).item()),
        'balanced_accuracy': float(balanced_accuracy_score(labels_binary, preds_binary)),
        'roc_auc': float(roc_auc_score(labels_binary, predictions)),
        'pr_auc': float(average_precision_score(labels_binary, predictions)),
        'precision_at_10': float(precision_score(labels_binary, preds_binary)),
        'recall_at_10': float(recall_score(labels_binary, preds_binary)),
        'map_at_10': float(map_at_k(labels_binary, predictions, k=10)),
        'ndcg_at_10': float(ndcg_at_k(labels_binary, predictions, k=10))
    }

    return metrics


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_deepfm(config):
    """
    Main DeepFM evaluation pipeline.

    Args:
        config: dict, Configuration from setup

    Returns:
        dict: Evaluation results
    """
    print("\n" + "=" * 60)
    print("DEEPFM EVALUATION")
    print("=" * 60 + "\n")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Setup paths
    logs_path = Path(config['logs_path'])
    models_path = Path(config['models_path'])
    results_path = Path(config['results_path'])

    preprocessed_path = logs_path / 'deepfm_preprocessed'
    model_dir = models_path / 'deepfm'

    # =========================================================================
    # Load Data and Config
    # =========================================================================
    print("[1/4] Loading data and configuration...")

    test_data = torch.load(preprocessed_path / 'test.pt')

    with open(preprocessed_path / 'field_config.json', 'r') as f:
        field_config = json.load(f)

    print(f"  Test samples: {len(test_data['labels'])}")

    # =========================================================================
    # Evaluate Variant A: Without Sentiment
    # =========================================================================
    print("\n[2/4] Evaluating DeepFM without sentiment...")

    # Create model
    model_a = DeepFM(
        field_config=field_config,
        embed_dim=EMBED_DIM,
        dnn_hidden_dims=DNN_HIDDEN_DIMS,
        dropout=DROPOUT,
        use_sentiment=False
    )
    model_a.load_state_dict(torch.load(model_dir / 'deepfm_without_sentiment.pt', map_location=device))
    model_a.to(device)

    # Create dataloader
    test_dataset_a = DeepFMDataset(test_data, use_sentiment=False)
    test_loader_a = DataLoader(
        test_dataset_a,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Evaluate
    preds_a, labels_a = evaluate_model(model_a, test_loader_a, device)
    metrics_a = calculate_all_metrics(preds_a, labels_a)

    print(f"  ROC-AUC: {metrics_a['roc_auc']:.4f}")
    print(f"  Balanced Accuracy: {metrics_a['balanced_accuracy']:.4f}")

    # =========================================================================
    # Evaluate Variant B: With Sentiment
    # =========================================================================
    print("\n[3/4] Evaluating DeepFM with sentiment...")

    # Create model
    model_b = DeepFM(
        field_config=field_config,
        embed_dim=EMBED_DIM,
        dnn_hidden_dims=DNN_HIDDEN_DIMS,
        dropout=DROPOUT,
        use_sentiment=True
    )
    model_b.load_state_dict(torch.load(model_dir / 'deepfm_with_sentiment.pt', map_location=device))
    model_b.to(device)

    # Create dataloader
    test_dataset_b = DeepFMDataset(test_data, use_sentiment=True)
    test_loader_b = DataLoader(
        test_dataset_b,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Evaluate
    preds_b, labels_b = evaluate_model(model_b, test_loader_b, device)
    metrics_b = calculate_all_metrics(preds_b, labels_b)

    print(f"  ROC-AUC: {metrics_b['roc_auc']:.4f}")
    print(f"  Balanced Accuracy: {metrics_b['balanced_accuracy']:.4f}")

    # =========================================================================
    # Generate Comparison
    # =========================================================================
    print("\n[4/4] Generating comparison...")

    # Calculate deltas
    deltas = {}
    for key in metrics_a.keys():
        deltas[key] = metrics_b[key] - metrics_a[key]

    # Create comparison table
    comparison = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'variant_a_without_sentiment': metrics_a,
        'variant_b_with_sentiment': metrics_b,
        'deltas': deltas,
        'target_metrics': {
            'loss': 0.5998,
            'balanced_accuracy': 0.7636,
            'roc_auc': 0.8447,
            'pr_auc': 0.8214,
            'precision_at_10': 0.0592,
            'recall_at_10': 0.0541,
            'map_at_10': 0.1398,
            'ndcg_at_10': 0.2096
        }
    }

    # Save comparison
    comparison_path = results_path / 'deepfm_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved: {comparison_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Metric':<25} {'Without Sentiment':>18} {'With Sentiment':>18} {'Delta':>12} {'Target':>12}")
    print("-" * 85)

    metric_names = {
        'loss': 'Loss',
        'balanced_accuracy': 'Balanced Accuracy',
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'precision_at_10': 'Precision@10',
        'recall_at_10': 'Recall@10',
        'map_at_10': 'MAP@10',
        'ndcg_at_10': 'NDCG@10'
    }

    for key, name in metric_names.items():
        target = comparison['target_metrics'][key]
        delta = deltas[key]
        print(f"{name:<25} {metrics_a[key]:>18.4f} {metrics_b[key]:>18.4f} {delta:>+12.4f} {target:>12.4f}")

    # =========================================================================
    # Generate Visualization
    # =========================================================================
    print("\n[Visualizing] Generating metrics comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ROC-AUC comparison
    metrics_to_plot = ['roc_auc', 'pr_auc', 'balanced_accuracy', 'loss']
    titles = ['ROC-AUC', 'PR-AUC', 'Balanced Accuracy', 'Loss']

    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[i // 2, i % 2]
        x = ['Without\nSentiment', 'With\nSentiment', 'Target']
        y = [metrics_a[metric], metrics_b[metric], comparison['target_metrics'][metric]]
        colors = ['#3498db', '#2ecc71', '#e74c3c']

        bars = ax.bar(x, y, color=colors, edgecolor='black', linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')

        # Add value labels
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
 f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_path / 'deepfm_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {results_path / 'deepfm_metrics.png'}")

    # =========================================================================
    # Verification
    # =========================================================================
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    target_roc_auc = 0.8447
    lower_bound = 0.80
    upper_bound = 0.88

    if lower_bound <= metrics_b['roc_auc'] <= upper_bound:
        print(f"✓ ROC-AUC {metrics_b['roc_auc']:.4f} is within target range [{lower_bound:.2f}-{upper_bound:.2f}]")
        verification_status = "PASS"
    else:
        print(f"✗ ROC-AUC {metrics_b['roc_auc']:.4f} is outside target range [{lower_bound:.2f}-{upper_bound:.2f}]")
        verification_status = "FAIL"

    return {
        'variant_a_metrics': metrics_a,
        'variant_b_metrics': metrics_b,
        'deltas': deltas,
        'comparison_path': str(comparison_path),
        'verification_status': verification_status
    }


if __name__ == '__main__':
    config = load_config()
    result = evaluate_deepfm(config)

    print("\nEvaluation Summary:")
    for key, value in result.items():
        if key not in ['variant_a_metrics', 'variant_b_metrics', 'deltas']:
            print(f"  {key}: {value}")
