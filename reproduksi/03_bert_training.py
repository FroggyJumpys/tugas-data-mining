"""
03_bert_training.py
====================
BERT sentiment classifier training.

Fine-tunes bert-base-uncased for 5-class sentiment classification.
Saves best model based on macro F1-score.

Hyperparameters (from paper Table 1):
- Learning Rate: 2e-5
- Max Sequence Length: 256
- Batch Size: 16
- Epoch: 5
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: Linear warmup + linear decay

Usage:
    python 03_bert_training.py

Output:
    - models/bert_sentiment/best_model.pt
    - models/bert_sentiment/tokenizer/
    - models/bert_sentiment/training_log.json
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config


# ============================================================================
# Configuration (from paper)
# ============================================================================

RANDOM_SEED = 42
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_LABELS = 5


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(predictions, labels):
    """Calculate macro F1 and per-class metrics."""
    preds = np.argmax(predictions, axis=1)
    macro_f1 = f1_score(labels, preds, average='macro')
    return macro_f1, preds


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="  Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on dataloader."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            all_predictions.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    macro_f1, preds = calculate_metrics(all_predictions, all_labels)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, macro_f1, preds, all_labels


# ============================================================================
# Main Training Function
# ============================================================================

def train_bert(config):
    """
    Main BERT training pipeline.

    Args:
        config: dict, Configuration from setup

    Returns:
        dict: Training results and paths
    """
    print("\n" + "=" * 60)
    print("BERT TRAINING -5-Class Sentiment Classification")
    print("=" * 60 + "\n")

    # Set seed
    set_seed(RANDOM_SEED)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Setup paths
    base_path = Path(config['base_path'])
    logs_path = Path(config['logs_path'])
    models_path = Path(config['models_path'])

    preprocessed_path = logs_path / 'bert_preprocessed'
    model_path = models_path / 'bert_sentiment'
    model_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Load Preprocessed Data
    # =========================================================================
    print("[1/5] Loading preprocessed data...")

    train_data = torch.load(preprocessed_path / 'train.pt')
    test_data = torch.load(preprocessed_path / 'test.pt')

    print(f"  Train samples: {len(train_data['labels'])}")
    print(f"  Test samples: {len(test_data['labels'])}")

    # Create datasets
    from torch.utils.data import Dataset

    class IMDbDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings['input_ids'])

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            return item

    train_dataset = IMDbDataset(train_data)
    test_dataset = IMDbDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # =========================================================================
    # Load Model
    # =========================================================================
    print("\n[2/5] Loading BERT model (bert-base-uncased)...")
    print(f"  Number of labels: {NUM_LABELS}")

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=NUM_LABELS
    )
    model.to(device)
    print("  Model loaded successfully")

    # =========================================================================
    # Setup Optimizer and Scheduler
    # =========================================================================
    print("\n[3/5] Setting up optimizer and scheduler...")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")

    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n[4/5] Starting training loop...")

    training_log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'weight_decay': WEIGHT_DECAY,
            'warmup_ratio': WARMUP_RATIO,
            'num_labels': NUM_LABELS,
            'random_seed': RANDOM_SEED
        },
        'epochs': []
    }

    best_macro_f1 = 0
    best_epoch = 0

    for epoch in range(EPOCHS):
        print(f"\n  Epoch {epoch + 1}/{EPOCHS}")
        print("  " + "-" * 40)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

        # Evaluate
        val_loss, val_f1, val_preds, val_labels = evaluate(model, test_loader, device)

        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_macro_f1': val_f1
        }
        training_log['epochs'].append(epoch_log)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Macro F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch = epoch + 1

            # Save model
            torch.save(model.state_dict(), model_path / 'best_model.pt')

            # Save tokenizer
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.save_pretrained(model_path / 'tokenizer')

            print(f"  [NEW BEST] Macro F1: {val_f1:.4f}")

    # =========================================================================
    # Save Training Log
    # =========================================================================
    print("\n[5/5] Saving training artifacts...")

    training_log['best_epoch'] = best_epoch
    training_log['best_macro_f1'] = best_macro_f1

    log_path = model_path / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"  Saved training_log.json: {log_path}")
    print(f"  Saved best_model.pt: {model_path / 'best_model.pt'}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    print(f"Target Macro F1: 0.75 (from paper)")
    print(f"Deviation: {abs(best_macro_f1 - 0.75) / 0.75 * 100:.2f}%")

    return {
        'model_path': str(model_path / 'best_model.pt'),
        'tokenizer_path': str(model_path / 'tokenizer'),
        'log_path': str(log_path),
        'best_epoch': best_epoch,
        'best_macro_f1': best_macro_f1
    }


if __name__ == '__main__':
    config = load_config()
    result = train_bert(config)

    print("\nTraining Summary:")
    for key, value in result.items():
        print(f"  {key}: {value}")
