#!/usr/bin/env python3
"""
Dataset to Prompts Converter

This script takes a generated dataset CSV file, removes the header row,
and renames it to prompts.csv for further processing.

Usage:
    python convert_to_prompts.py --dataset dataset_10.csv
    python convert_to_prompts.py --dataset dataset_20.csv
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def convert_to_prompts(dataset_name, input_dir=None, output_dir=None):
    """
    Convert a dataset CSV to prompts.csv by removing the header.
    
    Args:
        dataset_name (str): Name of the dataset file (e.g., 'dataset_10.csv')
        input_dir (str): Directory containing the dataset file
        output_dir (str): Directory to save the prompts.csv file
        
    Returns:
        str: Path to the created prompts.csv file
    """
    # Set default paths
    if input_dir is None:
        input_dir = "../data_chatgpt_paraphraizes"
    
    if output_dir is None:
        output_dir = "../data_chatgpt_paraphraizes"
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    input_path = script_dir / input_dir / dataset_name
    output_path = script_dir / output_dir / "prompts.csv"
    
    print(f"Reading dataset from: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Dataset file not found at {input_path}")
        return None
    
    try:
        # Read the dataset CSV file
        df = pd.read_csv(input_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Save without header (header=False)
        df.to_csv(output_path, index=False, header=False)
        
        print(f"Successfully converted dataset to prompts.csv")
        print(f"Removed header row - file now contains {len(df)} data rows")
        print(f"Prompts file saved to: {output_path}")
        
        # Show sample of the converted file
        print("\nSample of converted prompts.csv:")
        print(df.head(3).to_string())
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Convert dataset CSV to prompts.csv')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Name of the dataset file (e.g., dataset_10.csv)')
    parser.add_argument('--input-dir', type=str,
                       help='Directory containing the dataset file (default: ../data_chatgpt_paraphraizes)')
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save prompts.csv (default: ../data_chatgpt_paraphraizes)')
    
    args = parser.parse_args()
    
    print(f"Converting {args.dataset} to prompts.csv...")
    result = convert_to_prompts(
        dataset_name=args.dataset, 
        input_dir=args.input_dir, 
        output_dir=args.output_dir
    )
    
    if result:
        print(f"\nConversion completed successfully!")
    else:
        print(f"\nConversion failed!")
        exit(1)


if __name__ == "__main__":
    main()