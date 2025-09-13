#!/usr/bin/env python3
"""
Run script to create dataset and convert to prompts

This script takes a number n as input, creates a dataset with n rows,
and then converts it to prompts.csv (without header).

Usage:
    python run.py --n 1000
    python run.py --n 5000
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Create dataset and convert to prompts')
    parser.add_argument('--n', type=int, required=True, 
                       help='Number of rows to generate in the dataset')
    
    args = parser.parse_args()
    
    print(f"Creating dataset with {args.n} rows and converting to prompts...")
    print("=" * 60)
    
    # Step 1: Create dataset
    dataset_command = f"python3 create_dataset.py --n {args.n}"
    if not run_command(dataset_command, f"Creating dataset_{args.n}.csv"):
        print("Failed to create dataset. Exiting.")
        sys.exit(1)
    
    # Step 2: Convert to prompts
    convert_command = f"python3 convert_to_prompts.py --dataset dataset_{args.n}.csv"
    if not run_command(convert_command, f"Converting dataset_{args.n}.csv to prompts.csv"):
        print("Failed to convert dataset. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ All done!")
    print(f"📁 Created: dataset_{args.n}.csv")
    print(f"📁 Created: prompts.csv")
    print(f"📊 Dataset contains {args.n} rows (after filtering)")
    print("🎯 Ready to use with RAG system!")


if __name__ == "__main__":
    main()