#!/usr/bin/env python3
"""
Dataset Generator for ChatGPT Paraphrases

This script generates a new CSV dataset from the original ChatGPT paraphrases data.
It takes the 'text' column and expands the 'paraphrases' column into separate 
par_1, par_2, par_3, etc. columns.

Usage:
    python create_dataset.py --n 10
"""

import pandas as pd
import ast
import argparse
import os
from pathlib import Path


def parse_paraphrases(paraphrases_str):
    """
    Parse the paraphrases string representation into a list of strings.
    
    Args:
        paraphrases_str (str): String representation of a list of paraphrases
        
    Returns:
        list: List of paraphrase strings
    """
    try:
        # Use ast.literal_eval to safely parse the string representation
        paraphrases_list = ast.literal_eval(paraphrases_str)
        return paraphrases_list if isinstance(paraphrases_list, list) else []
    except (ValueError, SyntaxError):
        # If parsing fails, return empty list
        return []


def create_dataset(n=10, input_file=None, output_file=None):
    """
    Create a dataset with text and paraphrase columns.
    
    Args:
        n (int): Number of rows to generate
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        
    Returns:
        str: Path to the created output file
    """
    # Set default paths
    if input_file is None:
        input_file = "../data_chatgpt_paraphrases/chatgpt_paraphrases.csv"
    
    if output_file is None:
        output_file = f"../data_chatgpt_paraphrases/dataset_{n}.csv"
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    input_path = script_dir / input_file
    output_path = script_dir / output_file
    
    print(f"Reading data from: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Read the original CSV file
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows from original dataset")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None
    
    # Take only the first n rows
    df_sample = df.head(n).copy()
    print(f"Selected {len(df_sample)} rows for processing")
    
    # Process each row to extract paraphrases
    processed_rows = []
    skipped_rows = 0
    
    for idx, row in df_sample.iterrows():
        text = row['text']
        paraphrases_str = row['paraphrases']
        
        # Parse paraphrases
        paraphrases_list = parse_paraphrases(paraphrases_str)
        
        # Skip rows with only one paraphrase (we ignore the first one)
        if len(paraphrases_list) <= 1:
            skipped_rows += 1
            continue
        
        # Ignore the first paraphrase and use the rest
        paraphrases_list = paraphrases_list[1:]
        
        # Create a dictionary for this row
        row_dict = {'text': text}
        
        # Add paraphrase columns (par_1, par_2, etc.)
        for i, paraphrase in enumerate(paraphrases_list, 1):
            row_dict[f'par_{i}'] = paraphrase
        
        processed_rows.append(row_dict)
    
    print(f"Skipped {skipped_rows} rows with only one paraphrase")
    
    # Create new DataFrame
    new_df = pd.DataFrame(processed_rows)
    
    # Fill missing paraphrase columns with empty strings
    max_paraphrases = new_df.filter(regex='^par_').shape[1]
    print(f"Maximum number of paraphrases found: {max_paraphrases}")
    
    # Ensure all rows have the same columns by filling missing values
    for i in range(1, max_paraphrases + 1):
        col_name = f'par_{i}'
        if col_name not in new_df.columns:
            new_df[col_name] = ''
    
    # Reorder columns to have text first, then par_1, par_2, etc.
    par_columns = [f'par_{i}' for i in range(1, max_paraphrases + 1)]
    column_order = ['text'] + par_columns
    new_df = new_df[column_order]
    
    # Save to CSV
    try:
        new_df.to_csv(output_path, index=False)
        print(f"Successfully created dataset with {len(new_df)} rows and {len(new_df.columns)} columns")
        print(f"Columns: {list(new_df.columns)}")
        print(f"Dataset saved to: {output_path}")
        
        # Show sample of the created dataset
        print("\nSample of created dataset:")
        print(new_df.head(2).to_string())
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error saving output file: {e}")
        return None


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Generate dataset from ChatGPT paraphrases')
    parser.add_argument('--n', type=int, default=10, 
                       help='Number of rows to generate (default: 10)')
    parser.add_argument('--input', type=str, 
                       help='Input CSV file path (default: ../data_chatgpt_paraphraizes/chatgpt_paraphrases.csv)')
    parser.add_argument('--output', type=str,
                       help='Output CSV file path (default: ../data_chatgpt_paraphraizes/dataset_{n}.csv)')
    
    args = parser.parse_args()
    
    print(f"Generating dataset with {args.n} rows...")
    result = create_dataset(n=args.n, input_file=args.input, output_file=args.output)
    
    if result:
        print(f"\nDataset generation completed successfully!")
    else:
        print(f"\nDataset generation failed!")
        exit(1)


if __name__ == "__main__":
    main()