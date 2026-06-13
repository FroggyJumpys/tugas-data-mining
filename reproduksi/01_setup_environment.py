"""
01_setup_environment.py
========================
Setup environment for BERT + DeepFM experiment reproduction.

Handles:
- Environment detection (local vs Google Colab)
- Google Drive mounting for Colab
- Directory structure creation
- Dataset path validation
- System info display

Usage:
    python 01_setup_environment.py

Author: Claude (Reproduksi Baseline)
"""

import os
import sys
import platform
from pathlib import Path
from datetime import datetime


def detect_environment():
    """
    Detect if running on Google Colab or local machine.

    Returns:
        dict: Environment info with keys:
            - is_colab: bool
            - base_path: Path (working directory)
            - has_gpu: bool
            - gpu_name: str or None
    """
    is_colab = False
    has_gpu = False
    gpu_name = None

    # Check if running on Google Colab
    try:
        from google.colab import drive
        is_colab = True
    except ImportError:
        pass

    # Check GPU availability
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        'is_colab': is_colab,
        'has_gpu': has_gpu,
        'gpu_name': gpu_name
    }


def mount_google_drive():
    """
    Mount Google Drive for Colab environment.
    Returns the mount point path.
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return Path('/content/drive/MyDrive')
    except ImportError:
        print("Warning: Not running on Colab, skipping Google Drive mount")
        return None
    except Exception as e:
        print(f"Warning: Could not mount Google Drive: {e}")
        return None


def get_base_path():
    """
    Get the base working directory.
 For Colab: /content/ or Google Drive
    For local: relative to script location
    """
    # Try to determine script location
    script_dir = Path(__file__).parent.resolve()

    # Check if we're in a Colab environment
    if os.path.exists('/content'):
        return Path('/content')

    return script_dir.parent  # Go up from reproduksi-eksperimen/ to src/


def create_directory_structure(base_path):
    """
    Create all necessary directories for the experiment.

    Directory structure:
    ├── dataset/
    │   ├── IMDB/
    │   └── MovieLens_OneM/
    ├── reproduksi-eksperimen/
    │   ├── models/
    │   │   ├── bert_sentiment/
    │   │   └── deepfm/
    │   ├── logs/
    │   │   ├── bert_preprocessed/
    │   │   └── deepfm_preprocessed/
    │   └── results/
    """
    dirs_to_create = [
        base_path / 'reproduksi-eksperimen' / 'models' / 'bert_sentiment',
        base_path / 'reproduksi-eksperimen' / 'models' / 'deepfm',
        base_path / 'reproduksi-eksperimen' / 'logs' / 'bert_preprocessed',
        base_path / 'reproduksi-eksperimen' / 'logs' / 'deepfm_preprocessed',
        base_path / 'reproduksi-eksperimen' / 'results',
    ]

    created = []
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        created.append(d)

    return created


def validate_dataset_paths(base_path):
    """
    Validate that required dataset files exist.

    Expected files:
    - dataset/IMDB/imdb_reviews.csv
    - dataset/MovieLens_OneM/ratings.dat
    - dataset/MovieLens_OneM/users.dat
    - dataset/MovieLens_OneM/movies.dat
    """
    required_files = [
        'dataset/IMDB/imdb_reviews.csv',
        'dataset/MovieLens_OneM/ratings.dat',
        'dataset/MovieLens_OneM/users.dat',
        'dataset/MovieLens_OneM/movies.dat',
    ]

    results = {}
    for file_path in required_files:
        full_path = base_path / file_path
        exists = full_path.exists()
        results[file_path] = {
            'path': str(full_path),
            'exists': exists
        }
        if not exists:
            print(f"  [WARNING] Missing: {file_path}")

    return results


def print_system_info(env_info):
    """Print system information for debugging."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print()
    print(f"Environment: {'Google Colab' if env_info['is_colab'] else 'Local Machine'}")
    print(f"GPU Available: {'Yes' if env_info['has_gpu'] else 'No'}")
    if env_info['gpu_name']:
        print(f"GPU Name: {env_info['gpu_name']}")
    print()

    # Memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM Total: {mem.total / (1024**3):.2f} GB")
        print(f"RAM Available: {mem.available / (1024**3):.2f} GB")
        print(f"RAM Usage: {mem.percent}%")
    except ImportError:
        print("psutil not installed, skipping memory info")
    print()


def setup_environment():
    """
    Main setup function.

    Returns:
        dict: Configuration with all paths and settings
    """
    print("\n" + "=" * 60)
    print("SETUP ENVIRONMENT - BERT + DeepFM Reproduction")
    print("=" * 60 + "\n")

    # Step 1: Detect environment
    print("[1/5] Detecting environment...")
    env_info = detect_environment()

    # Step 2: Mount Google Drive if Colab
    drive_path = None
    if env_info['is_colab']:
        print("[2/5] Mounting Google Drive...")
        drive_path = mount_google_drive()
        if drive_path:
            print(f"  Google Drive mounted at: {drive_path}")
    else:
        print("[2/5] Skipping Google Drive mount (not Colab)")

    # Step 3: Determine base path
    print("[3/5] Determining base path...")
    base_path = get_base_path()
    print(f"  Base path: {base_path}")

    # Step 4: Create directory structure
    print("[4/5] Creating directory structure...")
    created_dirs = create_directory_structure(base_path)
    print(f"  Created {len(created_dirs)} directories")
    for d in created_dirs:
        print(f"    - {d.relative_to(base_path)}")

    # Step 5: Validate datasets
    print("[5/5] Validating dataset paths...")
    dataset_status = validate_dataset_paths(base_path)
    all_exist = all(v['exists'] for v in dataset_status.values())

    if all_exist:
        print("  All required datasets found!")
    else:
        print("  [WARNING] Some datasets are missing!")

    # Print system info
    print_system_info(env_info)

    # Build configuration
    config = {
        'env_info': env_info,
        'drive_path': drive_path,
        'base_path': base_path,
        'dataset_path': base_path / 'dataset',
        'output_path': base_path / 'reproduksi-eksperimen',
        'models_path': base_path / 'reproduksi-eksperimen' / 'models',
        'logs_path': base_path / 'reproduksi-eksperimen' / 'logs',
        'results_path': base_path / 'reproduksi-eksperimen' / 'results',
        'datasets_valid': all_exist,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }

    # Save config for other scripts
    import json
    config_path = base_path / 'reproduksi-eksperimen' / 'config.json'
    # Convert Path objects to strings for JSON serialization
    config_json = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    with open(config_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    # Final status
    print("\n" + "=" * 60)
    if all_exist and env_info['has_gpu']:
        print("STATUS: Ready to run experiments!")
        print("=" * 60 + "\n")
        return config
    elif all_exist and not env_info['has_gpu']:
        print("STATUS: Ready for Google Colab (no local GPU)")
        print("=" * 60 + "\n")
        return config
    else:
        print("STATUS: Setup incomplete - missing datasets")
        print("=" * 60 + "\n")
        return config


if __name__ == '__main__':
    config = setup_environment()

    # Print final paths for verification
    print("\nConfiguration paths:")
    for key, value in config.items():
        if key != 'env_info':
            print(f"  {key}: {value}")
