#!/usr/bin/env python3
"""
File Utils Switcher Script

Switches file_utils.py with either file_version_for_eval.py or file_version_for_training.py based on argument.

Usage:
    python switch_file_utils.py eval
    python switch_file_utils.py training
"""

import argparse
import shutil
from pathlib import Path

# Base directory path (relative from project root)
BASE_DIR = Path("robomimic-mirage/robomimic/utils")
FILE_UTILS_PATH = BASE_DIR / "file_utils.py"
EVAL_VERSION_PATH = BASE_DIR / "file_version_for_eval.py"
TRAINING_VERSION_PATH = BASE_DIR / "file_version_for_training.py"

def copy_version_file(source_path, target_path):
    """Copy version file to target location"""
    if not source_path.exists():
        print(f"✗ Source file not found: {source_path}")
        return False
    
    if not target_path.parent.exists():
        print(f"✗ Target directory not found: {target_path.parent}")
        return False
    
    try:
        shutil.copy2(source_path, target_path)
        print(f"✓ Copied {source_path.name} to {target_path.name}")
        return True
    except Exception as e:
        print(f"✗ Error copying {source_path.name}: {e}")
        return False

def check_file_exists(file_path):
    """Check if a file exists and show its size"""
    if file_path.exists():
        size = file_path.stat().st_size
        print(f"✓ {file_path.name} exists ({size} bytes)")
        return True
    else:
        print(f"✗ {file_path.name} not found")
        return False

def main():
    parser = argparse.ArgumentParser(description='Switch file_utils.py between eval and training versions')
    parser.add_argument('version', choices=['eval', 'train'], 
                       help='Version to switch to: eval or train')
    
    args = parser.parse_args()
    
    # Determine source file
    if args.version == 'eval':
        source_path = EVAL_VERSION_PATH
        print("Switching to eval version of file_utils.py")
    else:
        source_path = TRAINING_VERSION_PATH
        print("Switching to train version of file_utils.py")
    
    print(f"Source file: {source_path}")
    print(f"Target file: {FILE_UTILS_PATH}")
    success = copy_version_file(source_path, FILE_UTILS_PATH)
    
    if success:
        print(f"\n✓ Successfully switched to {args.version} version!")
        print(f"file_utils.py now contains the {args.version} version.")
    else:
        print(f"\n✗ Failed to switch to {args.version} version.")

if __name__ == "__main__":
    main()