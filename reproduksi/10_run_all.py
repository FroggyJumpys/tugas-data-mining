"""
10_run_all.py
==============
Orchestrator script to run all experiment steps in sequence.

Steps:
1. Setup environment
2. BERT preprocessing
3. BERT training
4. BERT evaluation
5. Generate sentiment features
6. DeepFM preprocessing
7. DeepFM training (both variants)
8. DeepFM evaluation
9. Final comparison

Usage:
    python 10_run_all.py

Or run individual steps:
    python 10_run_all.py --step 3  # Resume from step 3
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))


# ============================================================================
# Helper function for exec with proper __file__ context
# ============================================================================

def exec_script(script_path, mod_name):
    """Execute a Python script and return its globals with __file__ set."""
    exec_globals = {
        '__file__': str(script_path),
        '__name__': mod_name,
        '__package__': None,
    }
    exec(open(script_path).read(), exec_globals)
    return exec_globals


# ============================================================================
# Step Definitions with Dynamic Imports
# ============================================================================

def step_01_setup():
    """Step 1: Setup environment."""
    print("\n" + "=" * 70)
    print("STEP 1: SETUP ENVIRONMENT")
    print("=" * 70)

    exec_globals = exec_script(script_dir / '01_setup_environment.py', 'setup_env')
    config = exec_globals['setup_environment']()
    return config


def step_02_bert_preprocessing(config):
    """Step 2: BERT preprocessing."""
    print("\n" + "=" * 70)
    print("STEP 2: BERT PREPROCESSING")
    print("=" * 70)

    exec_globals = exec_script(script_dir / '02_bert_preprocessing.py', 'bert_prep')
    result = exec_globals['preprocess_bert_data'](config)
    return result


def step_03_bert_training(config):
    """Step 3: BERT training."""
    print("\n" + "=" * 70)
    print("STEP 3: BERT TRAINING")
    print("=" * 70)

    exec_globals = exec_script(script_dir / '03_bert_training.py', 'bert_train')
    result = exec_globals['train_bert'](config)
    return result


def step_04_bert_evaluation(config):
    """Step 4: BERT evaluation."""
    print("\n" + "=" * 70)
    print("STEP 4: BERT EVALUATION")
    print("=" * 70)

    exec_globals = exec_script(script_dir / '04_bert_evaluation.py', 'bert_eval')
    result = exec_globals['evaluate_bert'](config)
    return result


def step_05_sentiment_features(config):
    """Step 5: Generate sentiment features."""
    print("\n" + "=" * 70)
    print("STEP 5: GENERATE SENTIMENT FEATURES")
    print("=" * 70)

    exec_globals = exec_script(script_dir / '05_generate_sentiment_features.py', 'sentiment')
    result = exec_globals['generate_sentiment_features'](config)
    return result


def step_06_deepfm_preprocessing(config):
    """Step 6: DeepFM preprocessing."""
    print("\n" + "=" * 70)
    print("STEP 6: DEEPFM PREPROCESSING")
    print("=" * 70)

    exec_globals = exec_script(script_dir / '06_deepfm_preprocessing.py', 'deepfm_prep')
    result = exec_globals['preprocess_deepfm'](config)
    return result


def step_07_deepfm_training(config):
    """Step 7: DeepFM training."""
    print("\n" + "=" * 70)
    print("STEP 7: DEEPFM TRAINING")
    print("=" * 70)

    # Load model first, then training script
    mod = exec_script(script_dir / '07_deepfm_model.py', 'deepfm_model')
    exec_globals = exec_script(script_dir / '08_deepfm_training.py', 'deepfm_train')
    # Merge model into training globals
    exec_globals.update(mod)
    result = exec_globals['train_deepfm'](config)
    return result


def step_08_deepfm_evaluation(config):
    """Step 8: DeepFM evaluation."""
    print("\n" + "=" * 70)
    print("STEP 8: DEEPFM EVALUATION")
    print("=" * 70)

    # Load model first, then evaluation script
    mod = exec_script(script_dir / '07_deepfm_model.py', 'deepfm_model')
    exec_globals = exec_script(script_dir / '09_deepfm_evaluation.py', 'deepfm_eval')
    # Merge model into evaluation globals
    exec_globals.update(mod)
    result = exec_globals['evaluate_deepfm'](config)
    return result


# ============================================================================
# Main Orchestrator
# ============================================================================

def run_all(start_step=1, end_step=8):
    """
    Run all experiment steps.

    Args:
        start_step: int, Starting step (1-8)
        end_step: int, Ending step (1-8)
    """
    print("\n" + "=" * 70)
    print("REPRODUKSI BASELINE - BERT + DeepFM")
    print("Running Steps: {} to {}".format(start_step, end_step))
    print("=" * 70)

    start_time = datetime.now()
    results = {}

    try:
        # Step 1: Setup
        if start_step <= 1 <= end_step:
            results['step_01'] = step_01_setup()
            config = results['step_01']
        else:
            from config import load_config
            config = load_config()

        # Step 2: BERT Preprocessing
        if start_step <= 2 <= end_step:
            results['step_02'] = step_02_bert_preprocessing(config)

        # Step 3: BERT Training
        if start_step <= 3 <= end_step:
            results['step_03'] = step_03_bert_training(config)

        # Step 4: BERT Evaluation
        if start_step <= 4 <= end_step:
            results['step_04'] = step_04_bert_evaluation(config)

        # Step 5: Sentiment Features
        if start_step <= 5 <= end_step:
            results['step_05'] = step_05_sentiment_features(config)

        # Step 6: DeepFM Preprocessing
        if start_step <= 6 <= end_step:
            results['step_06'] = step_06_deepfm_preprocessing(config)

        # Step 7: DeepFM Training
        if start_step <= 7 <= end_step:
            results['step_07'] = step_07_deepfm_training(config)

        # Step 8: DeepFM Evaluation
        if start_step <= 8 <= end_step:
            results['step_08'] = step_08_deepfm_evaluation(config)

    except Exception as e:
        print(f"\n[ERROR] Step failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # =========================================================================
    # Final Summary
    # =========================================================================
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")

    # Save final results
    base_path = Path(config['base_path'])
    results_path = base_path / 'reproduksi-eksperimen' / 'results'
    results_path.mkdir(parents=True, exist_ok=True)

    final_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': duration.total_seconds(),
        'steps_completed': list(results.keys()),
        'results': {}
    }

    # Add key metrics
    if 'step_04' in results:
        final_summary['results']['bert_accuracy'] = results['step_04'].get('accuracy')
        final_summary['results']['bert_macro_f1'] = results['step_04'].get('macro_f1')
        final_summary['results']['bert_verification'] = results['step_04'].get('verification_status')

    if 'step_08' in results:
        final_summary['results']['deepfm_roc_auc_without'] = results['step_08'].get('variant_a_metrics', {}).get('roc_auc')
        final_summary['results']['deepfm_roc_auc_with'] = results['step_08'].get('variant_b_metrics', {}).get('roc_auc')
        final_summary['results']['deepfm_verification'] = results['step_08'].get('verification_status')

    summary_path = results_path / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    # Print key results
    print("\n" + "-" * 70)
    print("KEY RESULTS")
    print("-" * 70)

    if 'step_04' in results:
        print("\nBERT Sentiment Classification:")
        acc = results['step_04'].get('accuracy', 'N/A')
        f1 = results['step_04'].get('macro_f1', 'N/A')
        status = results['step_04'].get('verification_status', 'N/A')
        print(f"  Accuracy: {acc:.4f}" if isinstance(acc, float) else f"  Accuracy: {acc}")
        print(f"  Macro F1: {f1:.4f}" if isinstance(f1, float) else f"  Macro F1: {f1}")
        print(f"  Target: 0.7581 (75.81%)")
        print(f"  Status: {status}")

    if 'step_08' in results:
        auc_without = results['step_08'].get('variant_a_metrics', {}).get('roc_auc', 'N/A')
        auc_with = results['step_08'].get('variant_b_metrics', {}).get('roc_auc', 'N/A')
        status = results['step_08'].get('verification_status', 'N/A')
        print("\nDeepFM Recommendation:")
        print(f"  ROC-AUC (Without Sentiment): {auc_without:.4f}" if isinstance(auc_without, float) else f"  ROC-AUC (Without Sentiment): {auc_without}")
        print(f"  ROC-AUC (With Sentiment): {auc_with:.4f}" if isinstance(auc_with, float) else f"  ROC-AUC (With Sentiment): {auc_with}")
        print(f"  Target: 0.8447 (84.47%)")
        print(f"  Status: {status}")

    print("\n" + "=" * 70)

    return results


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all experiment steps')
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help='Start from this step (1-8)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=8,
        help='End at this step (1-8)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all steps'
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable steps:")
        print("  1: Setup environment")
        print("  2: BERT preprocessing")
        print("  3: BERT training")
        print("  4: BERT evaluation")
        print("  5: Generate sentiment features")
        print("  6: DeepFM preprocessing")
        print("  7: DeepFM training")
        print("  8: DeepFM evaluation")
        print("\nUsage:")
        print("  python 10_run_all.py              # Run all steps")
        print("  python 10_run_all.py --step 3      # Start from step 3")
        print("  python 10_run_all.py --step 5 --end 7  # Run steps 5-7")
    else:
        run_all(start_step=args.step, end_step=args.end)
