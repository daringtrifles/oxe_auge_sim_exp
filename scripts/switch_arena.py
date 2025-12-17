#!/usr/bin/env python3
"""
Arena Switcher Script

Replaces XML files in the arenas directory with files from either 'original_arenas' or 'altered_lighting_arenas'.

Usage:
    python switch_arena.py original
    python switch_arena.py lighting
"""

import argparse
import os
import shutil
from pathlib import Path

# Base directory paths (relative from project root)
BASE_DIR = Path("robosuite/robosuite/models/assets")
ARENAS_DIR = BASE_DIR / "arenas"
ORIGINAL_ARENAS_DIR = BASE_DIR / "original_arenas"  
ALTERED_LIGHTING_ARENAS_DIR = BASE_DIR / "altered_lighting_arenas"



def copy_arena_files(source_dir, target_dir):
    """Copy XML files from source directory to target directory."""
    if not source_dir.exists():
        print(f"✗ Source directory not found: {source_dir}")
        return False
    
    if not target_dir.exists():
        print(f"✗ Target directory not found: {target_dir}")
        return False
    
    # Find all XML files in source directory
    xml_files = list(source_dir.glob("*.xml"))
    
    if not xml_files:
        print(f"✗ No XML files found in {source_dir}")
        return False
    
    copied_count = 0
    for xml_file in xml_files:
        target_file = target_dir / xml_file.name
        try:
            shutil.copy2(xml_file, target_file)
            print(f"✓ Copied {xml_file.name}")
            copied_count += 1
        except Exception as e:
            print(f"✗ Error copying {xml_file.name}: {e}")
    
    return copied_count > 0

def list_xml_files(directory):
    """List all XML files in a directory."""
    if not directory.exists():
        return []
    return list(directory.glob("*.xml"))

def main():
    parser = argparse.ArgumentParser(description='Switch arena XML files in robosuite')
    parser.add_argument('arena_type', choices=['original', 'lighting'], 
                       help='Arena type: original or altered lighting')
    
    args = parser.parse_args()
    
    # Determine source directory
    if args.arena_type == 'original':
        source_dir = ORIGINAL_ARENAS_DIR
        print("Switching to ORIGINAL arenas")
    else:
        source_dir = ALTERED_LIGHTING_ARENAS_DIR
        print("Switching to ALTERED LIGHTING arenas")
    
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {ARENAS_DIR}")
    
    # Check if directories exist
    if not ARENAS_DIR.exists():
        print(f"✗ Target arenas directory not found: {ARENAS_DIR}")
        return
    
    if not source_dir.exists():
        print(f"✗ Source directory not found: {source_dir}")
        print("Available directories:")
        for d in BASE_DIR.iterdir():
            if d.is_dir() and 'arena' in d.name.lower():
                print(f"  - {d}")
        return
    



    
    # Copy files
    print(f"\nCopying XML files from {source_dir.name} to arenas...")
    success = copy_arena_files(source_dir, ARENAS_DIR)
    
    if success:
        print(f"\n✓ Successfully switched to {args.arena_type} arenas!")
        print(f"XML files in {source_dir.name} have been copied to the arenas directory.")
    else:
        print(f"\n ✗ Failed to switch arenas.")
if __name__ == "__main__":
    main()