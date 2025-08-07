"""
APPS Dataset Cleaning Pipeline

This script cleans the APPS dataset and creates balanced train/eval/test splits
saved as Parquet files for efficient loading.
"""

import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
from typing import Dict, List, Tuple

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APPSDatasetCleaner:
    """Clean and split the APPS dataset into train/eval/test splits."""
    
    def __init__(self, output_dir: str = "data/APPS/cleaned"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_dataset(self) -> pd.DataFrame:
        """Load the raw APPS dataset from HuggingFace."""
        logger.info("Loading raw APPS dataset from HuggingFace...")
        
        dataset = load_dataset("codeparrot/apps", trust_remote_code=True)
        
        # Combine train and test splits
        all_data = []
        
        for split_name, split_data in dataset.items():
            logger.info(f"Processing {split_name} split ({len(split_data)} samples)...")
            
            for item in split_data:
                try:
                    # Parse JSON fields
                    solutions = []
                    if item.get('solutions'):
                        try:
                            solutions = json.loads(item['solutions'])
                        except:
                            solutions = []
                    
                    inputs = []
                    outputs = []
                    if item.get('input_output'):
                        try:
                            io_data = json.loads(item['input_output'])
                            inputs = io_data.get('inputs', [])
                            outputs = io_data.get('outputs', [])
                        except:
                            inputs = []
                            outputs = []
                    
                    # Only include problems with at least one test case
                    if len(inputs) == 0 or len(outputs) == 0:
                        continue
                    
                    # Create cleaned record
                    cleaned_item = {
                        'problem_id': item['problem_id'],
                        'question': item['question'],
                        'difficulty': item.get('difficulty', 'unknown'),
                        'solutions': solutions,
                        'inputs': inputs,
                        'outputs': outputs,
                        'starter_code': item.get('starter_code', ''),
                        'n_test_cases': len(inputs),
                        'n_solutions': len(solutions),
                        'original_split': split_name
                    }
                    
                    all_data.append(cleaned_item)
                    
                except Exception as e:
                    logger.warning(f"Error processing item {item.get('problem_id', 'unknown')}: {e}")
                    continue
        
        df = pd.DataFrame(all_data)
        
        # Diagnostics: Check for NaN/None and non-list values in 'inputs' and 'outputs'
        for col in ['inputs', 'outputs']:
            n_nan = df[col].isnull().sum()
            n_nonlist = (~df[col].apply(lambda x: isinstance(x, list))).sum()
            print(f"[DIAGNOSTIC] {col}: NaN/None count = {n_nan}, non-list count = {n_nonlist}")
        
        # Ensure list columns are properly formatted for Parquet
        list_columns = ['solutions', 'inputs', 'outputs']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else ([] if pd.isnull(x) or x is None else [x]))
        
        # Check for non-string elements inside 'inputs' and 'outputs'
        for col in ['inputs', 'outputs']:
            bad_rows = df[~df[col].apply(lambda lst: all(isinstance(x, str) for x in lst))]
            if not bad_rows.empty:
                print(f"[ERROR] {col} contains non-string elements in the following rows:")
                print(bad_rows[[col, 'problem_id']].head(10))
                print(f"Total problematic rows in {col}: {len(bad_rows)}")
                
                # Show detailed examples of the problematic data
                print(f"\n[EXAMPLES] Detailed examples of problematic {col}:")
                for idx, row in bad_rows.head(5).iterrows():
                    print(f"Problem ID {row['problem_id']}:")
                    for i, item in enumerate(row[col]):
                        print(f"  Item {i}: {item} (type: {type(item).__name__})")
                        if isinstance(item, list):
                            for j, subitem in enumerate(item):
                                print(f"    Subitem {j}: {subitem} (type: {type(subitem).__name__})")
                    print()
        
        # Stringify inputs and outputs to make them Parquet-compatible
        print("\n[FIXING] Stringifying inputs and outputs for Parquet compatibility...")
        for col in ['inputs', 'outputs']:
            df[col] = df[col].apply(lambda lst: [str(item) for item in lst])
            print(f"  Stringified {col}: {len(df)} problems")
        
        # Verify all elements are now strings
        for col in ['inputs', 'outputs']:
            all_strings = df[col].apply(lambda lst: all(isinstance(x, str) for x in lst)).all()
            print(f"  {col} all strings: {all_strings}")
        
        logger.info(f"Loaded {len(df)} valid problems after cleaning")
        
        return df
    
    def create_balanced_splits(self, df: pd.DataFrame, 
                              train_ratio: float = 0.6,
                              eval_ratio: float = 0.2,
                              test_ratio: float = 0.2,
                              random_seed: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Create balanced train/eval/test splits stratified by difficulty.
        
        Args:
            df: DataFrame with cleaned data
            train_ratio: Proportion for training set
            eval_ratio: Proportion for evaluation set  
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'eval', 'test' DataFrames
        """
        logger.info("Creating balanced splits...")
        
        # Verify ratios sum to 1
        total_ratio = train_ratio + eval_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Set random seed
        np.random.seed(random_seed)
        
        splits: Dict[str, List[pd.DataFrame]] = {}
        
        # Split by difficulty to ensure balanced distribution
        for difficulty in df['difficulty'].unique():
            difficulty_df = df[df['difficulty'] == difficulty].copy()
            
            if len(difficulty_df) == 0:
                continue
                
            logger.info(f"Processing {difficulty} difficulty: {len(difficulty_df)} problems")
            
            # First split: train vs (eval + test)
            train_size = int(len(difficulty_df) * train_ratio)
            remaining_size = len(difficulty_df) - train_size
            
            # Second split: eval vs test from remaining
            eval_size = int(remaining_size * (eval_ratio / (eval_ratio + test_ratio)))
            test_size = remaining_size - eval_size
            
            # Create splits
            train_df = difficulty_df.iloc[:train_size]
            eval_df = difficulty_df.iloc[train_size:train_size + eval_size]
            test_df = difficulty_df.iloc[train_size + eval_size:]
            
            # Add to splits
            for split_name, split_df in [('train', train_df), ('eval', eval_df), ('test', test_df)]:
                if split_name not in splits:
                    splits[split_name] = []
                splits[split_name].append(split_df)
        
        # Combine all difficulties for each split
        final_splits = {}
        for split_name, split_dfs in splits.items():
            if split_dfs:
                final_splits[split_name] = pd.concat(split_dfs, ignore_index=True)
                # Shuffle within each split
                final_splits[split_name] = final_splits[split_name].sample(frac=1, random_state=random_seed).reset_index(drop=True)
            else:
                final_splits[split_name] = pd.DataFrame()
        
        # Log split statistics
        for split_name, split_df in final_splits.items():
            logger.info(f"{split_name.capitalize()} split: {len(split_df)} problems")
            if len(split_df) > 0:
                difficulty_counts = split_df['difficulty'].value_counts()
                logger.info(f"  Difficulty distribution: {dict(difficulty_counts)}")
        
        return final_splits
    
    def save_splits(self, splits: Dict[str, pd.DataFrame]):
        """Save splits as Parquet files."""
        logger.info("Saving splits as Parquet files...")
        
        for split_name, split_df in splits.items():
            if len(split_df) == 0:
                logger.warning(f"No data for {split_name} split, skipping...")
                continue
                
            output_file = self.output_dir / f"apps_{split_name}.parquet"
            split_df.to_parquet(output_file, index=False)
            logger.info(f"Saved {len(split_df)} problems to {output_file}")
    
    def print_statistics(self, splits: Dict[str, pd.DataFrame]):
        """Print detailed statistics about the cleaned dataset."""
        print("\n" + "="*60)
        print("CLEANED DATASET STATISTICS")
        print("="*60)
        
        total_problems = sum(len(df) for df in splits.values())
        print(f"Total problems: {total_problems:,}")
        
        for split_name, split_df in splits.items():
            if len(split_df) == 0:
                continue
                
            print(f"\n{split_name.upper()} SPLIT ({len(split_df):,} problems):")
            print("-" * 40)
            
            # Difficulty distribution
            difficulty_counts = split_df['difficulty'].value_counts()
            for difficulty, count in difficulty_counts.items():
                percentage = (count / len(split_df)) * 100
                print(f"  {difficulty:15}: {count:4,} problems ({percentage:5.1f}%)")
            
            # Test case statistics
            test_case_stats = split_df['n_test_cases'].describe()
            print(f"\n  Test cases per problem:")
            print(f"    Mean: {test_case_stats['mean']:.1f}")
            print(f"    Median: {test_case_stats['50%']:.1f}")
            print(f"    Min: {test_case_stats['min']:.0f}")
            print(f"    Max: {test_case_stats['max']:.0f}")
            
            # Solution statistics
            solution_stats = split_df['n_solutions'].describe()
            print(f"\n  Solutions per problem:")
            print(f"    Mean: {solution_stats['mean']:.1f}")
            print(f"    Median: {solution_stats['50%']:.1f}")
            print(f"    Min: {solution_stats['min']:.0f}")
            print(f"    Max: {solution_stats['max']:.0f}")
            
            # Problems with starter code
            has_starter = split_df['starter_code'].str.len() > 0
            starter_count = has_starter.sum()
            starter_pct = (starter_count / len(split_df)) * 100
            print(f"\n  Problems with starter code: {starter_count:,} ({starter_pct:.1f}%)")
    
    def run_cleaning_pipeline(self, random_seed: int = 42):
        """Run the complete cleaning pipeline."""
        logger.info("Starting APPS dataset cleaning pipeline...")
        
        # Step 1: Load and clean raw data
        df = self.load_raw_dataset()
        
        # Step 2: Create balanced splits
        splits = self.create_balanced_splits(df, random_seed=random_seed)
        
        # Step 3: Save splits
        self.save_splits(splits)
        
        # Step 4: Print statistics
        self.print_statistics(splits)
        
        logger.info("Cleaning pipeline completed successfully!")
        
        return splits


def main():
    """Main function to run the cleaning pipeline."""
    cleaner = APPSDatasetCleaner()
    splits = cleaner.run_cleaning_pipeline(random_seed=42)
    
    print(f"\nCleaned dataset saved to: {cleaner.output_dir}")
    print("Files created:")
    for split_name in ['train', 'eval', 'test']:
        file_path = cleaner.output_dir / f"apps_{split_name}.parquet"
        if file_path.exists():
            print(f"  - {file_path}")


if __name__ == "__main__":
    main() 