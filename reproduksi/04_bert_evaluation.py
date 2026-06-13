"""
04_bert_evaluation.py
======================
BERT sentiment classifier evaluation.

Generates:
- Classification report (per-class precision/recall/F1)
- Confusion matrix heatmap (PNG)
- Per-epoch learning curve (PNG)

Usage:
    python 04_bert_evaluation.py

Output:
    - results/bert_classification_report.txt
    - results/bert_confusion_matrix.png
    - results/bert_training_curve.png
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config


# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 16
NUM_LABELS = 5
LABEL_NAMES = ['Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Positive']


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            all_predictions.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    preds = np.argmax(all_predictions, axis=1)

    return preds, all_labels, all_predictions


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('BERT Sentiment Classification - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved confusion matrix: {save_path}")


def plot_learning_curve(training_log, save_path):
    """Generate and save learning curve."""
    epochs = [e['epoch'] for e in training_log['epochs']]
    train_losses = [e['train_loss'] for e in training_log['epochs']]
    val_losses = [e['val_loss'] for e in training_log['epochs']]
    val_f1s = [e['val_macro_f1'] for e in training_log['epochs']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-o', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 curve
    axes[1].plot(epochs, val_f1s, 'g-o', label='Val Macro F1')
    axes[1].axhline(y=0.75, color='r', linestyle='--', label='Target (0.75)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1 Score')
    axes[1].set_title('Validation Macro F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved learning curve: {save_path}")


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_bert(config):
    """
    Main BERT evaluation pipeline.

    Args:
        config: dict, Configuration from setup

    Returns:
        dict: Evaluation results
    """
    print("\n" + "=" * 60)
    print("BERT EVALUATION")
    print("=" * 60 + "\n")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Setup paths
    base_path = Path(config['base_path'])
    logs_path = Path(config['logs_path'])
    models_path = Path(config['models_path'])
    results_path = Path(config['results_path'])

    preprocessed_path = logs_path / 'bert_preprocessed'
    model_path = models_path / 'bert_sentiment'

    # =========================================================================
    # Load Model and Data
    # =========================================================================
    print("[1/4] Loading model and data...")

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(model_path / 'best_model.pt', map_location=device))
    model.to(device)
    model.eval()
    print(f"  Loaded model from: {model_path / 'best_model.pt'}")

    # Load test data
    test_data = torch.load(preprocessed_path / 'test.pt')

    class IMDbDataset:
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings['input_ids'])

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            return item

    test_dataset = IMDbDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # =========================================================================
    # Run Evaluation
    # =========================================================================
    print("\n[2/4] Running evaluation on test set...")

    preds, labels, probs = evaluate_model(model, test_loader, device)

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')

    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    # =========================================================================
    # Generate Classification Report
    # =========================================================================
    print("\n[3/4] Generating classification report...")

    report = classification_report(
        labels, preds,
        target_names=LABEL_NAMES,
        digits=4
    )

    # Save classification report
    report_path = results_path / 'bert_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BERT SENTIMENT CLASSIFICATION - CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Macro F1-Score: {macro_f1:.4f}\n")
        f.write(f"Weighted F1-Score: {weighted_f1:.4f}\n\n")
        f.write("-" * 60 + "\n\n")
        f.write(report)
        f.write("\n\n" + "-" * 60 + "\n")
        f.write("Target from Paper:\n")
        f.write("  Accuracy: 75.81%\n")
        f.write("  Macro F1: 0.75\n")
        f.write(f"\nDeviation from Target:\n")
        f.write(f"  Accuracy: {abs(accuracy - 0.7581) / 0.7581 * 100:.2f}%\n")
        f.write(f"  Macro F1: {abs(macro_f1 - 0.75) / 0.75 * 100:.2f}%\n")

    print(f"  Saved classification report: {report_path}")
    print("\n" + report)

    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    print("\n[4/4] Generating visualizations...")

    # Confusion matrix
    plot_confusion_matrix(
        labels, preds,
        LABEL_NAMES,
        results_path / 'bert_confusion_matrix.png'
    )

    # Learning curve
    training_log_path = model_path / 'training_log.json'
    if training_log_path.exists():
        with open(training_log_path, 'r') as f:
            training_log = json.load(f)
        plot_learning_curve(
            training_log,
            results_path / 'bert_training_curve.png'
        )

    # =========================================================================
    # Verification
    # =========================================================================
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    # Check if within acceptable range (72%-79% accuracy)
    target_accuracy = 0.7581
    lower_bound = 0.72
    upper_bound = 0.79

    if lower_bound <= accuracy <= upper_bound:
        print(f"✓ Accuracy {accuracy*100:.2f}% is within target range [{lower_bound*100:.0f}%-{upper_bound*100:.0f}%]")
        verification_status = "PASS"
    else:
        print(f"✗ Accuracy {accuracy*100:.2f}% is outside target range [{lower_bound*100:.0f}%-{upper_bound*100:.0f}%]")
        verification_status = "FAIL"

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'verification_status': verification_status,
        'report_path': str(report_path),
        'confusion_matrix_path': str(results_path / 'bert_confusion_matrix.png'),
        'learning_curve_path': str(results_path / 'bert_training_curve.png')
    }


if __name__ == '__main__':
    config = load_config()
    result = evaluate_bert(config)

    print("\nEvaluation Summary:")
    for key, value in result.items():
        print(f"  {key}: {value}")
